---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 2af059d0d6c7742814b36ef757464387_2456676a6a5411f1a99c5254007bceed
    ReservedCode1: iITXvEHVc/ya5hrRVoSUmY+MdDYTpDL9lNRmPKkFs0fGKjsP7bYP0BSCP+cgs5EazooFW3zhw13PgL4+ibEHMBRSZqVS9yAw4UkXHI9II0n+yj4nJSQB3ktajlhdNCAGEKG05YnjlSKYYKuQM1r4CQ2h26Lg1wRWCpxTasiNcojd+yByXh02EGhqwak=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 2af059d0d6c7742814b36ef757464387_2456676a6a5411f1a99c5254007bceed
    ReservedCode2: iITXvEHVc/ya5hrRVoSUmY+MdDYTpDL9lNRmPKkFs0fGKjsP7bYP0BSCP+cgs5EazooFW3zhw13PgL4+ibEHMBRSZqVS9yAw4UkXHI9II0n+yj4nJSQB3ktajlhdNCAGEKG05YnjlSKYYKuQM1r4CQ2h26Lg1wRWCpxTasiNcojd+yByXh02EGhqwak=
---



# Second Brain Skill 设计文档

> 构建"永不掉线"且完全私有的个人数字第二大脑

---

## 一、总体架构

```
┌─────────────────────────────────────────────────────────┐
│                    Marvis Agent                          │
│  use_skill("second-brain") → 加载本 Skill              │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│                 SecondBrain 引擎                         │
│                                                         │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────┐   │
│  │ Ingest   │──▶│ Chunk +  │──▶│ ChromaDB          │   │
│  │ (解析)   │   │ Embed    │   │ (向量存储,本地)   │   │
│  └──────────┘   └──────────┘   └────────┬─────────┘   │
│                                         │              │
│  ┌──────────────────────────────────────┘              │
│  │                                                      │
│  ▼                                                      │
│  ┌────────────────────┐   ┌─────────────────────────┐  │
│  │ Stage 1: Dense     │──▶│ Stage 2: Cross-Encoder   │  │
│  │ Retrieval (Top-20) │   │ Rerank → Top-K (≤5)      │  │
│  └────────────────────┘   └───────────┬─────────────┘  │
│                                       │                 │
│  ┌────────────────────────────────────┘                 │
│  │                                                      │
│  ▼                                                      │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 上下文组装 → Marvis LLM 生成最终回答              │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 二、核心组件

### 2.1 文档解析层 (`DocumentParser`)

| 格式 | 解析器 | 策略 |
|------|--------|------|
| PDF | PyMuPDF (fitz) | 逐页提取文本，保持阅读顺序 |
| Markdown | 正则清洗 | 去图片链接保留 alt，去链接保留文字，保留标题层级 |
| TXT/JSON | 原生读取 | UTF-8 兜底自动容错 |

### 2.2 文本分割器 (`TextSplitter`)

- **粒度**: 512 tokens / chunk，64 tokens overlap
- **策略**: 递归分割 —— 先按双换行(段落)，超长再按句子
- **overlap 机制**: 相邻 chunk 尾部 64 字符重叠，防止边界截断

### 2.3 向量化 (`VectorStore`)

- **引擎**: ChromaDB (PersistentClient，纯本地)
- **Embedding**: `all-MiniLM-L6-v2` (Sentence-Transformers, 384维, 轻量)
- **相似度**: Cosine
- **存储**: `data/vectors/` 目录持久化

### 2.4 两阶段检索

| 阶段 | 方法 | 候选数 | 说明 |
|------|------|--------|------|
| Stage 1 | 稠密召回 (bi-encoder) | Top-20 | 快速从全库召回相关 chunk |
| Stage 2 | Cross-Encoder 重排序 | Top-K (默认5) | 用 `ms-marco-MiniLM-L-6-v2` 对 (query, chunk) 联合打分 |

**为什么需要两阶段？** 双塔模型 (bi-encoder) 速度快但精度有限；Cross-Encoder 对 query-document 做联合编码，精度显著更高。先用 Stage 1 缩小候选集，再用 Stage 2 精排，兼顾速度与精度。

### 2.5 研报摘要 (`summarize`)

- 自动提取首部关键段落 + 全文均匀采样
- 支持 `focus` 参数聚焦特定主题(如"财务数据"、"风险提示"、"行业趋势")
- 输出结构化 JSON，由 Marvis LLM 做最终精炼

---

## 三、命令接口

```bash
# 1. 导入文档
python second_brain.py ingest /path/to/PDFs /path/to/notes

# 2. 智能问答
python second_brain.py query "这家公司的核心竞争优势是什么？"

# 3. 研报摘要
python second_brain.py summarize report.pdf "财务数据与估值"

# 4. 查看知识库状态
python second_brain.py status
```

---

## 四、使用流程

```
Step 1: 安装依赖
  python install_deps.py

Step 2: 导入知识库
  brain = SecondBrain()
  brain.ingest(["/your/pdf/folder", "/your/notes/folder"])

Step 3: 问答
  result = brain.query("你的问题")
  # result["context"] → 组装好的 RAG 上下文
  # result["hits"]    → Top-K 检索结果详情

Step 4: 单文件摘要
  result = brain.summarize("report.pdf", focus="风险提示")
```

---

## 五、设计原则

1. **完全离线**: 所有模型本地运行 (all-MiniLM-L6-v2, ms-marco-MiniLM-L-6-v2)，无需联网
2. **持久化**: ChromaDB 向量库和文件索引 JSON 均落地磁盘，重启不丢失
3. **增量可扩展**: `ingest(reset=False)` 支持追加导入
4. **隐私优先**: 数据不出本机，不存在第三方 API 调用
5. **轻量**: 两个模型合计 < 200MB，CPU 可运行

---

## 六、文件清单

| 文件 | 说明 |
|------|------|
| `second_brain.py` | 核心引擎 (约 500 行) |
| `install_deps.py` | 依赖安装脚本 |
| `data/vectors/` | ChromaDB 向量持久化目录 |
| `data/index/index_meta.json` | 文件索引元数据 |
| `data/reports/` | 摘要输出目录 |
*（内容由AI生成，仅供参考）*
*（内容由AI生成，仅供参考）*
