"""
Second Brain Skill — 依赖一键安装
运行: python install_deps.py
"""

import subprocess
import sys
import importlib

PACKAGES = {
    "sentence-transformers": "sentence-transformers>=2.2.0",
    "chromadb":              "chromadb>=0.4.0",
    "fitz":                  "PyMuPDF>=1.23.0",
    "torch":                 "torch>=2.0.0",
    "transformers":          "transformers>=4.30.0",
}

def check(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False

def install(pip_name: str):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", pip_name],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

def main():
    print("Second Brain — 依赖检查与安装")
    print("-" * 40)

    missing = []
    for mod, pip_name in PACKAGES.items():
        ok = check(mod)
        print(f"  {'✓' if ok else '✗'} {mod}")
        if not ok:
            missing.append(pip_name)

    if not missing:
        print("\n所有依赖已就绪。")
        return

    print(f"\n即将安装 {len(missing)} 个缺失的包...")
    for pip_name in missing:
        print(f"\n→ 安装 {pip_name} ...")
        install(pip_name)

    print("\n安装完成。运行 python second_brain_skill.py --help 开始使用。")

if __name__ == "__main__":
    main()
