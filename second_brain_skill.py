"""
╔═══════════════════════════════════════════════════════════════════════╗
║           Second Brain Skill — 个人数字第二大脑引擎                     ║
║                                                                       ║
║  能力：                                                                ║
║    • PDF / Markdown / TXT 批量入库与向量索引                            ║
║    • 两阶段 RAG 检索（稠密召回 + Cross-Encoder 重排序）                  ║
║    • 研报 / 长文档结构化摘要提取                                         ║
║    • 本地私人知识库智能问答（完全离线，数据不出机）                        ║
║                                                                      ║
║  架构：                                                              ║
║    Ingest → Chunk → Embed(all-MiniLM-L6-v2) → ChromaDB               ║
║    Query → Dense Retrieve(Top-20) → Cross-Encoder Rerank(Top-K)      ║
║                                                                  ║
║  使用方式：                                                      ║
║    python second_brain_skill.py ingest <path1> [path2 ...]       ║
║    python second_brain_skill.py query "<question>"               ║
║    python second_brain_skill.py summarize <file> [focus]          ║
║    python second_brain_skill.py status                           ║
║    python second_brain_skill.py interactive                      ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import re
import sys
import json
import hashlib
import logging
import argparse
import textwrap
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime

# ─── 日志 ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SecondBrain")


# ═══════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════

class Config:
    """全局配置，可通过环境变量覆盖"""

    # 路径：默认与脚本同目录
    SKILL_ROOT: Path = Path(__file__).resolve().parent

    # 数据目录
    DATA_DIR:    Path = SKILL_ROOT / "data"
    VECTOR_DIR:  Path = DATA_DIR / "vectors"
    INDEX_DIR:   Path = DATA_DIR / "index"
    REPORT_DIR:  Path = DATA_DIR / "reports"

    # 分割参数
    CHUNK_SIZE:     int = int(os.getenv("SB_CHUNK_SIZE",     "512"))
    CHUNK_OVERLAP:  int = int(os.getenv("SB_CHUNK_OVERLAP",  "64"))

    # 检索参数
    STAGE1_TOP_K:   int = int(os.getenv("SB_STAGE1_TOP_K",   "20"))
    STAGE2_TOP_K:   int = int(os.getenv("SB_STAGE2_TOP_K",   "5"))
    MAX_CONTEXT:    int = int(os.getenv("SB_MAX_CONTEXT",    "3000"))

    # 模型
    EMBED_MODEL:    str = os.getenv("SB_EMBED_MODEL",    "all-MiniLM-L6-v2")
    RERANK_MODEL:   str = os.getenv("SB_RERANK_MODEL",   "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # ChromaDB
    COLLECTION_NAME: str = os.getenv("SB_COLLECTION", "second_brain")

    @classmethod
    def ensure_dirs(cls):
        for d in [cls.DATA_DIR, cls.VECTOR_DIR, cls.INDEX_DIR, cls.REPORT_DIR]:
            d.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# 异常定义
# ═══════════════════════════════════════════════════════════════

class SecondBrainError(Exception):
    """基础异常"""

class DependencyError(SecondBrainError):
    """缺少依赖"""

class ParseError(SecondBrainError):
    """文档解析失败"""

class StoreError(SecondBrainError):
    """向量存储异常"""

class EmptyKnowledgeBase(SecondBrainError):
    """知识库为空"""


# ═══════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════

@dataclass
class Chunk:
    """知识块"""
    chunk_id: str
    text: str
    source: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HitResult:
    """单条检索命中"""
    chunk_id: str
    text: str
    source: str
    chunk_index: int
    distance: Optional[float] = None
    rerank_score: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "source": Path(self.source).name,
            "source_path": self.source,
            "chunk_index": self.chunk_index,
            "distance": self.distance,
            "rerank_score": self.rerank_score,
            "text_preview": self.text[:200] + ("..." if len(self.text) > 200 else ""),
        }

@dataclass
class QueryResult:
    """问答结果"""
    status: str                    # ok | empty | no_hits | error
    question: str
    context: str                   # 组装好的 RAG 上下文
    hits: List[HitResult]
    num_hits: int
    elapsed_ms: float = 0.0
    message: str = ""

@dataclass
class IngestResult:
    """入库结果"""
    status: str
    files_discovered: int
    files_parsed: int
    files_failed: int
    chunks_total: int
    chars_total: int
    failures: List[Dict] = field(default_factory=list)

@dataclass
class SummaryResult:
    """摘要结果"""
    status: str
    file: str
    filename: str
    word_count: int
    focus: str
    summary: str
    para_count: int


# ═══════════════════════════════════════════════════════════════
# 1. 文档解析器
# ═══════════════════════════════════════════════════════════════

class DocumentParser:
    """多格式文档解析器，支持 PDF / Markdown / TXT / JSON"""

    SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt", ".json", ".jsonl"}

    @staticmethod
    def parse_pdf(file_path: str) -> str:
        try:
            import fitz
        except ImportError:
            raise DependencyError(
                "PDF 解析需要 PyMuPDF。运行: pip install PyMuPDF"
            )
        try:
            doc = fitz.open(file_path)
            full_text: List[str] = []
            for page_num, page in enumerate(doc):
                text = page.get_text()
                if text.strip():
                    full_text.append(f"[第{page_num+1}页]\n{text}")
            doc.close()
            return "\n\n".join(full_text)
        except Exception as e:
            raise ParseError(f"PDF 解析失败: {file_path} — {e}")

    @staticmethod
    def parse_markdown(file_path: str) -> str:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="gbk", errors="replace") as f:
                text = f.read()

        # 去图片语法，保留 alt 文本
        text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'[图片: \1]', text)
        # 去链接保留文字
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        # 代码块保留语言标记
        text = re.sub(r'```(\w+)?\n', r'\n[代码块:\1]\n', text)
        text = text.replace("```", "")
        return text

    @staticmethod
    def parse_txt(file_path: str) -> str:
        for enc in ["utf-8", "gbk", "latin-1"]:
            try:
                with open(file_path, "r", encoding=enc, errors="replace") as f:
                    return f.read()
            except Exception:
                continue
        raise ParseError(f"无法解码文件: {file_path}")

    @staticmethod
    def parse_json(file_path: str) -> str:
        """JSON 文件递归展平为可读文本"""
        text = DocumentParser.parse_txt(file_path)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return text  # 就当普通文本

        def flatten(obj, prefix: str = "") -> List[str]:
            lines = []
            if isinstance(obj, dict):
                for k, v in obj.items():
                    lines.extend(flatten(v, f"{prefix}{k}." if prefix else f"{k}: "))
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    lines.extend(flatten(item, f"{prefix}[{i}]. "))
            else:
                lines.append(f"{prefix}{obj}")
            return lines

        return "\n".join(flatten(data))

    PARSERS = {
        ".pdf":      parse_pdf.__func__,
        ".md":       parse_markdown.__func__,
        ".markdown": parse_markdown.__func__,
        ".txt":      parse_txt.__func__,
        ".json":     parse_json.__func__,
        ".jsonl":    parse_txt.__func__,
    }

    @classmethod
    def parse(cls, file_path: str) -> str:
        ext = Path(file_path).suffix.lower()
        parser = cls.PARSERS.get(ext)
        if parser is None:
            raise ParseError(f"不支持的文件格式: {ext}（支持: {cls.SUPPORTED_EXTENSIONS}）")
        return parser(file_path)


# ═══════════════════════════════════════════════════════════════
# 2. 文本分割器
# ═══════════════════════════════════════════════════════════════

class TextSplitter:
    """
    递归文本分割器
    策略：段落 → 句子 → 字符，逐级降级
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = max(chunk_size, 128)
        self.chunk_overlap = min(chunk_overlap, self.chunk_size // 4)

    def split(self, text: str, source: str) -> List[Chunk]:
        if not text.strip():
            return []

        # Level 1: 段落分割
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        chunks: List[Chunk] = []
        current = ""
        chunk_idx = 0
        source_stem = Path(source).stem[:40]  # 截断过长的文件名

        for para in paragraphs:
            combined = (current + "\n\n" + para) if current else para

            if len(combined) <= self.chunk_size:
                current = combined
            else:
                if current:
                    chunks.append(self._make_chunk(current, source, source_stem, chunk_idx))
                    chunk_idx += 1

                # Level 2: 句子分割
                if len(para) > self.chunk_size:
                    sub = self._split_long(para, source, source_stem, chunk_idx)
                    chunks.extend(sub)
                    chunk_idx += len(sub)
                    current = ""
                else:
                    current = para

        if current.strip():
            chunks.append(self._make_chunk(current, source, source_stem, chunk_idx))

        # 追加 overlap 衔接
        self._add_overlap(chunks)

        logger.info("  切分: %d 块", len(chunks))
        return chunks

    @staticmethod
    def _make_chunk(text: str, source: str, stem: str, idx: int) -> Chunk:
        return Chunk(
            chunk_id=f"{stem}_{idx:04d}",
            text=text,
            source=source,
            chunk_index=idx,
        )

    def _split_long(self, text: str, source: str, stem: str, start_idx: int) -> List[Chunk]:
        sentences = re.split(r'(?<=[。！？.!?])\s*', text)
        if len(sentences) <= 1:
            # Level 3: 字符级强制截断
            return [
                self._make_chunk(text[i:i+self.chunk_size], source, stem, start_idx + j)
                for j, i in enumerate(range(0, len(text), self.chunk_size))
            ]

        chunks = []
        current = ""
        for sent in sentences:
            if len(current) + len(sent) <= self.chunk_size:
                current += sent
            else:
                if current:
                    chunks.append(self._make_chunk(current, source, stem, start_idx + len(chunks)))
                current = sent
        if current:
            chunks.append(self._make_chunk(current, source, stem, start_idx + len(chunks)))
        return chunks

    @staticmethod
    def _add_overlap(chunks: List[Chunk]):
        for i in range(1, len(chunks)):
            prev_tail = chunks[i-1].text[-64:]
            if prev_tail and prev_tail not in chunks[i].text:
                chunks[i].text = prev_tail + "\n" + chunks[i].text


# ═══════════════════════════════════════════════════════════════
# 3. 向量存储 (ChromaDB)
# ═══════════════════════════════════════════════════════════════

class VectorStore:
    """基于 ChromaDB 的本地向量存储"""

    def __init__(self, collection_name: str = "second_brain"):
        self.collection_name = collection_name
        self._ensure_chromadb()
        self.embedding_fn = None
        self.client = None
        self.collection = None
        self._init()

    @staticmethod
    def _ensure_chromadb():
        try:
            import chromadb
        except ImportError:
            raise DependencyError(
                "向量存储需要 chromadb。运行: pip install chromadb"
            )

    def _init(self):
        import chromadb
        from chromadb.utils import embedding_functions

        Config.ensure_dirs()

        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=Config.EMBED_MODEL,
            device="cpu",
        )

        self.client = chromadb.PersistentClient(path=str(Config.VECTOR_DIR))

        # 删除旧集合后重建（每次 ingest 全量刷新）
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass

        self.collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: List[Chunk]):
        if not chunks:
            return

        batch_size = 128
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            self.collection.add(
                ids=[c.chunk_id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[
                    {"source": c.source, "chunk_index": c.chunk_index}
                    for c in batch
                ],
            )
        logger.info("  写入向量库: %d 条", len(chunks))

    def search(self, query: str, top_k: int = 20) -> List[HitResult]:
        if self.collection is None:
            return []

        results = self.collection.query(query_texts=[query], n_results=top_k)

        hits = []
        ids       = results.get("ids", [[]])[0]
        docs      = results.get("documents", [[]])[0]
        metas     = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for i in range(len(ids)):
            hits.append(HitResult(
                chunk_id=ids[i],
                text=docs[i] if i < len(docs) else "",
                source=metas[i].get("source", "") if i < len(metas) else "",
                chunk_index=metas[i].get("chunk_index", 0) if i < len(metas) else 0,
                distance=distances[i] if i < len(distances) else None,
            ))

        return hits


# ═══════════════════════════════════════════════════════════════
# 4. Cross-Encoder 重排序器
# ═══════════════════════════════════════════════════════════════

class Reranker:
    """
    Cross-Encoder 重排序
    对 (query, chunk) 拼接后联合编码打分，精度远超双塔模型
    """

    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def _lazy_load(self):
        if self._loaded:
            return
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
        except ImportError:
            raise DependencyError(
                "重排序需要 transformers。运行: pip install transformers torch"
            )
        logger.info("  加载 Cross-Encoder: %s ...", self.MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
        self.model.eval()
        self._loaded = True

    def rerank(self, query: str, hits: List[HitResult], top_k: int = 5) -> List[HitResult]:
        if not hits:
            return []
        if len(hits) <= top_k:
            for h in hits:
                h.rerank_score = None  # 不需要重排
            return hits

        self._lazy_load()
        import torch

        pairs = [(query, h.text) for h in hits]
        scores: List[float] = []

        batch_size = 16
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = self.model(**encoded).logits
                scores.extend(logits.squeeze(-1).tolist())

        for h, s in zip(hits, scores):
            h.rerank_score = round(s, 4)

        hits.sort(key=lambda x: x.rerank_score or 0, reverse=True)
        return hits[:top_k]


# ═══════════════════════════════════════════════════════════════
# 5. 索引管理器
# ═══════════════════════════════════════════════════════════════

class IndexManager:
    """文件索引元数据的读写管理"""

    INDEX_FILE = Config.INDEX_DIR / "index_meta.json"

    def __init__(self):
        self.meta: Dict[str, dict] = {}
        self.load()

    def load(self):
        if self.INDEX_FILE.exists():
            try:
                self.meta = json.loads(self.INDEX_FILE.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.meta = {}
                logger.warning("索引文件损坏，已重置")

    def save(self):
        Config.ensure_dirs()
        self.INDEX_FILE.write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add_file(self, file_path: str, chunks: int, chars: int):
        file_hash = self._hash_file(Path(file_path))
        self.meta[file_path] = {
            "hash": file_hash,
            "chunks": chunks,
            "chars": chars,
            "type": Path(file_path).suffix.lower(),
            "indexed_at": datetime.now().isoformat(),
        }

    def remove_file(self, file_path: str):
        self.meta.pop(file_path, None)

    def get_stats(self) -> Dict:
        file_types = {}
        for m in self.meta.values():
            t = m.get("type", "unknown")
            file_types[t] = file_types.get(t, 0) + 1
        return {
            "indexed_files": len(self.meta),
            "file_types": file_types,
            "total_chunks": sum(m.get("chunks", 0) for m in self.meta.values()),
            "total_chars": sum(m.get("chars", 0) for m in self.meta.values()),
            "total_size_mb": round(
                sum(Path(fp).stat().st_size for fp in self.meta if Path(fp).exists())
                / (1024 * 1024), 2
            ),
        }

    @staticmethod
    def _hash_file(path: Path) -> str:
        if not path.exists():
            return "MISSING"
        return hashlib.md5(path.read_bytes()).hexdigest()


# ═══════════════════════════════════════════════════════════════
# 6. 核心引擎
# ═══════════════════════════════════════════════════════════════

class SecondBrain:
    """个人数字第二大脑"""

    def __init__(self):
        Config.ensure_dirs()
        self.index = IndexManager()
        self.vector_store: Optional[VectorStore] = None
        self.reranker: Optional[Reranker] = None
        self.splitter = TextSplitter(Config.CHUNK_SIZE, Config.CHUNK_OVERLAP)

    # ─── 入库 ────────────────────────────────────────────

    def ingest(self, paths: List[str], reset: bool = True) -> IngestResult:
        """
        导入文件/目录到知识库

        Args:
            paths: 文件或目录路径
            reset: 是否清空重建（默认 True）
        """
        import time
        t0 = time.time()

        logger.info("=" * 50)
        logger.info("开始入库...")

        # 1) 收集文件
        file_list = self._collect_files(paths)
        logger.info("发现 %d 个文件", len(file_list))

        # 2) 初始化向量库
        if reset or self.vector_store is None:
            self.vector_store = VectorStore(Config.COLLECTION_NAME)
            if reset:
                self.index = IndexManager()  # 重置索引

        # 3) 解析 & 切分
        all_chunks: List[Chunk] = []
        parsed = 0
        failed_list: List[Dict] = []

        for fp in file_list:
            try:
                text = DocumentParser.parse(fp)
                chunks = self.splitter.split(text, fp)
                all_chunks.extend(chunks)
                self.index.add_file(fp, len(chunks), len(text))
                parsed += 1
                logger.info("  ✓ %s (%d chunks)", Path(fp).name, len(chunks))
            except Exception as e:
                logger.error("  ✗ %s: %s", Path(fp).name, e)
                failed_list.append({"file": fp, "error": str(e)})

        # 4) 写入向量库
        self.vector_store.add_chunks(all_chunks)
        self.index.save()

        elapsed = (time.time() - t0) * 1000

        result = IngestResult(
            status="ok",
            files_discovered=len(file_list),
            files_parsed=parsed,
            files_failed=len(failed_list),
            chunks_total=len(all_chunks),
            chars_total=sum(len(c.text) for c in all_chunks),
            failures=failed_list,
        )
        logger.info(
            "入库完成: %d 文件 → %d 块, 耗时 %.1fs",
            parsed, len(all_chunks), elapsed / 1000,
        )
        return result

    def _collect_files(self, paths: List[str]) -> List[str]:
        files = []
        for p in paths:
            path = Path(p)
            if not path.exists():
                logger.warning("跳过不存在的路径: %s", p)
                continue
            if path.is_dir():
                for ext in DocumentParser.SUPPORTED_EXTENSIONS:
                    files.extend(str(f) for f in path.rglob(f"*{ext}"))
            elif path.is_file():
                if path.suffix.lower() in DocumentParser.SUPPORTED_EXTENSIONS:
                    files.append(str(path))
                else:
                    logger.warning("跳过不支持的文件: %s", p)
        return sorted(set(files))

    # ─── 检索 ────────────────────────────────────────────

    def retrieve(self, question: str, top_k: int = 5, use_rerank: bool = True) -> List[HitResult]:
        if self.vector_store is None:
            raise EmptyKnowledgeBase("知识库为空，请先执行 ingest")

        # Stage 1: 稠密召回
        stage1 = self.vector_store.search(question, top_k=Config.STAGE1_TOP_K)
        logger.info("  Stage1 召回: %d", len(stage1))

        if not use_rerank or len(stage1) <= top_k:
            return stage1[:top_k]

        # Stage 2: Cross-Encoder 重排序
        if self.reranker is None:
            self.reranker = Reranker()
        reranked = self.reranker.rerank(question, stage1, top_k=top_k)
        logger.info("  Stage2 重排序后: %d", len(reranked))
        return reranked

    def build_context(self, hits: List[HitResult], max_chars: int = 3000) -> str:
        parts = []
        total = 0
        for i, h in enumerate(hits):
            src = Path(h.source).name
            header = f"[{i+1}] 来源: {src}"
            body = f"{header}\n{h.text}\n"
            if total + len(body) > max_chars:
                remain = max_chars - total
                if remain > len(header) + 20:
                    body = body[:remain] + "..."
                    parts.append(body)
                break
            parts.append(body)
            total += len(body)
        return "\n---\n".join(parts)

    # ─── 问答 ────────────────────────────────────────────

    def query(self, question: str, top_k: int = 5) -> QueryResult:
        import time
        t0 = time.time()

        try:
            hits = self.retrieve(question, top_k=top_k, use_rerank=True)
        except EmptyKnowledgeBase as e:
            return QueryResult(
                status="empty", question=question, context="",
                hits=[], num_hits=0, message=str(e),
            )

        if not hits:
            return QueryResult(
                status="no_hits", question=question, context="",
                hits=[], num_hits=0,
                message="未找到相关内容，请尝试换个问法或导入更多文档。",
            )

        context = self.build_context(hits, Config.MAX_CONTEXT)
        elapsed = (time.time() - t0) * 1000

        return QueryResult(
            status="ok",
            question=question,
            context=context,
            hits=hits,
            num_hits=len(hits),
            elapsed_ms=round(elapsed, 1),
        )

    # ─── 摘要 ────────────────────────────────────────────

    def summarize(self, file_path: str, focus: str = "") -> SummaryResult:
        """
        研报/文档摘要提取

        Args:
            file_path: 文件路径
            focus: 关注主题（如"财务数据"、"风险提示"、"行业趋势"）
        """
        text = DocumentParser.parse(file_path)
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 20]

        if not paragraphs:
            return SummaryResult(
                status="empty", file=file_path, filename=Path(file_path).name,
                word_count=len(text), focus=focus, summary="（文档无有效段落）",
                para_count=0,
            )

        # 策略：首部 + 关键词匹配 + 均匀采样
        selected: List[str] = []
        selected.extend(paragraphs[:2])  # 首部

        if focus:
            keywords = re.split(r'[，,、\s]+', focus)
            for para in paragraphs[2:]:
                if any(k in para for k in keywords if k):
                    selected.append(para)
                    if len(selected) >= 6:
                        break

        # 补充均匀采样
        if len(selected) < 5:
            step = max(1, len(paragraphs) // (10 - len(selected)))
            for i in range(0, len(paragraphs), step):
                if paragraphs[i] not in selected:
                    selected.append(paragraphs[i])
                if len(selected) >= 10:
                    break

        summary = "\n\n".join(selected)

        return SummaryResult(
            status="ok",
            file=file_path,
            filename=Path(file_path).name,
            word_count=len(text),
            focus=focus or "全文概览",
            summary=summary[:Config.MAX_CONTEXT],
            para_count=len(paragraphs),
        )

    # ─── 状态 ────────────────────────────────────────────

    def status(self) -> Dict:
        stats = self.index.get_stats()
        stats["vector_dir"] = str(Config.VECTOR_DIR)
        stats["index_path"] = str(IndexManager.INDEX_FILE)
        stats["collection"] = Config.COLLECTION_NAME
        stats["embed_model"] = Config.EMBED_MODEL
        return stats


# ═══════════════════════════════════════════════════════════════
# 7. 交互式终端 (Interactive Shell)
# ═══════════════════════════════════════════════════════════════

def interactive_mode(brain: SecondBrain):
    """交互式问答终端"""
    print("\n" + "=" * 54)
    print("  Second Brain — 交互式问答")
    print("  :help  查看命令   :status  知识库状态   :quit  退出")
    print("=" * 54 + "\n")

    stats = brain.status()
    if stats["indexed_files"] == 0:
        print("⚠ 知识库为空。请先导入文件:")
        print("  python second_brain_skill.py ingest <path>\n")

    while True:
        try:
            user_input = input("🧠 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break

        if not user_input:
            continue

        if user_input.startswith(":"):
            cmd = user_input[1:].lower()
            if cmd == "quit" or cmd == "q":
                print("再见。")
                break
            elif cmd == "help":
                print(textwrap.dedent("""
                命令:
                  :status   查看知识库状态
                  :stats    详细统计
                  :files    列出已索引文件
                  :quit     退出
                  直接输入问题进行问答（如: 这家公司的核心竞争力是什么？）
                """))
            elif cmd == "status" or cmd == "stats":
                print(json.dumps(brain.status(), ensure_ascii=False, indent=2))
            elif cmd == "files":
                for fp in sorted(brain.index.meta.keys()):
                    m = brain.index.meta[fp]
                    print(f"  [{m['type']}] {Path(fp).name}  ({m['chunks']} chunks)")
            else:
                print(f"未知命令: {user_input}")
            continue

        # 问答
        print()
        result = brain.query(user_input)
        if result.status == "ok":
            print(f"（检索到 {result.num_hits} 个相关片段，耗时 {result.elapsed_ms}ms）\n")
            # 在交互模式下只显示上下文，由用户自己给 LLM
            print(result.context)
            print("\n---")
            print("💡 将以上上下文 + 你的问题发给 LLM 即可得到最终答案。")
        else:
            print(f"[{result.status}] {result.message}")
        print()


# ═══════════════════════════════════════════════════════════════
# 8. CLI 入口
# ═══════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Second Brain — 个人数字第二大脑引擎",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        示例:
          %(prog)s ingest ~/Documents/PDFs ~/Notes
          %(prog)s query "核心竞争优势是什么？"
          %(prog)s summarize report.pdf "财务数据与估值"
          %(prog)s status
          %(prog)s interactive
        """),
    )
    sub = parser.add_subparsers(dest="command", help="子命令")

    # ingest
    p_ingest = sub.add_parser("ingest", help="导入文件/目录")
    p_ingest.add_argument("paths", nargs="+", help="文件或目录路径")
    p_ingest.add_argument("--no-reset", action="store_true", help="不清空旧数据（增量追加）")

    # query
    p_query = sub.add_parser("query", help="智能问答")
    p_query.add_argument("question", help="问题")
    p_query.add_argument("-k", "--top-k", type=int, default=5, help="返回 Top-K (默认 5)")
    p_query.add_argument("--no-rerank", action="store_true", help="跳过重排序")
    p_query.add_argument("--json", action="store_true", help="JSON 输出")

    # summarize
    p_sum = sub.add_parser("summarize", help="研报摘要")
    p_sum.add_argument("file", help="文件路径")
    p_sum.add_argument("focus", nargs="?", default="", help="关注主题（可选）")
    p_sum.add_argument("--json", action="store_true", help="JSON 输出")

    # status
    sub.add_parser("status", help="查看知识库状态")

    # interactive
    sub.add_parser("interactive", help="交互式问答终端")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    try:
        brain = SecondBrain()

        if args.command == "ingest":
            result = brain.ingest(args.paths, reset=not args.no_reset)
            print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
            if result.failures:
                print(f"\n⚠ {len(result.failures)} 个文件解析失败:")
                for f in result.failures:
                    print(f"  ✗ {Path(f['file']).name}: {f['error']}")

        elif args.command == "query":
            result = brain.query(args.question, top_k=args.top_k)
            if args.json:
                output = {
                    "status": result.status,
                    "question": result.question,
                    "num_hits": result.num_hits,
                    "elapsed_ms": result.elapsed_ms,
                    "context": result.context,
                    "hits": [h.to_dict() for h in result.hits],
                }
                print(json.dumps(output, ensure_ascii=False, indent=2))
            else:
                print(f"\n问题: {result.question}")
                print(f"状态: {result.status} | 命中: {result.num_hits} | 耗时: {result.elapsed_ms}ms")
                if result.context:
                    print(f"\n{'─'*50}")
                    print(result.context)
                    print("─" * 50)
                if result.message:
                    print(f"\n{result.message}")

        elif args.command == "summarize":
            result = brain.summarize(args.file, args.focus)
            if args.json:
                print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
            else:
                print(f"\n文件: {result.filename}")
                print(f"字数: {result.word_count} | 段落: {result.para_count}")
                print(f"焦点: {result.focus}")
                print(f"\n{'─'*50}")
                print(result.summary)
                print("─" * 50)

        elif args.command == "status":
            print(json.dumps(brain.status(), ensure_ascii=False, indent=2))

        elif args.command == "interactive":
            interactive_mode(brain)

    except DependencyError as e:
        print(f"\n✗ 缺少依赖: {e}", file=sys.stderr)
        print("请运行: python install_deps.py", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ 错误: {e}", file=sys.stderr)
        logger.exception("Unhandled error")
        sys.exit(1)


if __name__ == "__main__":
    main()
