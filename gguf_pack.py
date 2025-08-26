#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
standalone_pack_and_quant.py
- 一键 clone 官方 ggml-org/llama.cpp（若已存在则跳过）
- 在仓库里创建并使用独立虚拟环境（隔离 convert 所需依赖）
- 调用 convert_hf_to_gguf.py 产出 FP16 GGUF
- 若本地已有 bin/llama-quantize 就用它；否则自动用 CMake 构建仓库里的 quant 工具再量化为 Q4_K_M
- 路径硬编码，可按需改动 HF_MODEL_DIR / 输出目录等

运行：
    python3 standalone_pack_and_quant.py
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

# ========= 硬编码区域（按需修改） =========
# HF 合并后的模型目录（前一步 merge_and_unload 的输出）
HF_MODEL_DIR = Path("./gemma3n-neko-lora-merged")

# 输出目录 & 文件名
OUT_DIR_FP16 = Path("./gguf/fp16")
OUT_DIR_Q4KM = Path("./gguf/q4km")
FP16_GGUF = OUT_DIR_FP16 / "gemma3n-neko-lora-fp16.gguf"
Q4KM_GGUF = OUT_DIR_Q4KM / "gemma3n-neko-lora-Q4_K_M.gguf"

# 你本地已有的量化/推理二进制（可选）
PREFERRED_LLAMA_QUANT = Path("./bin/llama-quantize")   # 若不存在则自动构建
PREFERRED_LLAMA_CLI   = Path("./bin/llama-cli")        # 自检用，可无

# 官方仓库与 clone 位置（默认当前目录下的 ./llama.cpp）
REPO_URL  = "https://github.com/ggml-org/llama.cpp.git"
CLONE_DIR = Path("./llama.cpp")
# 可选：固定某个提交/分支；None 表示最新（depth=1 的浅克隆）
GIT_REF: str | None = None  # 例："b6182" 或 "release"；不需要就保持 None

# 在仓库里创建一个只给转换用的 venv
CONVERT_VENV = CLONE_DIR / ".venv-convert"
# =======================================


def sh(cmd, **kwargs):
    """小封装：打印并执行命令"""
    print("➜", " ".join(map(str, cmd)))
    subprocess.check_call([str(c) for c in cmd], **kwargs)


def ensure_git():
    if shutil.which("git") is None:
        raise RuntimeError("未找到 git，请先安装：sudo apt install git （或对应发行版命令）")


def ensure_cmake():
    if shutil.which("cmake") is None:
        raise RuntimeError("未找到 cmake，请先安装：sudo apt install cmake （或对应发行版命令）")


def venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def clone_llama_cpp():
    if CLONE_DIR.exists():
        print(f"[clone] 已存在：{CLONE_DIR}，跳过克隆。")
        return
    print(f"[clone] 克隆仓库到：{CLONE_DIR}")
    sh(["git", "clone", "--depth", "1", REPO_URL, str(CLONE_DIR)])
    if GIT_REF:
        print(f"[git] 切换到 {GIT_REF}")
        sh(["git", "fetch", "origin", GIT_REF], cwd=CLONE_DIR)
        sh(["git", "checkout", GIT_REF], cwd=CLONE_DIR)


def ensure_convert_venv():
    py = sys.executable  # 用当前 Python 创建子 venv
    if not CONVERT_VENV.exists():
        print(f"[venv] 创建虚拟环境：{CONVERT_VENV}")
        sh([py, "-m", "venv", str(CONVERT_VENV)])
    vpy = venv_python(CONVERT_VENV)
    print(f"[venv] 升级 pip")
    sh([vpy, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    req = CLONE_DIR / "requirements" / "requirements-convert_hf_to_gguf.txt"
    if req.exists():
        print("[venv] 安装 convert 所需依赖（官方 requirements）")
        sh([vpy, "-m", "pip", "install", "-r", str(req)])
    else:
        # 少见情况：如果 requirements 文件结构变了，兜底装常见依赖
        print("[venv] 未找到官方 requirements，安装常见依赖兜底…")
        sh([vpy, "-m", "pip", "install",
            "transformers>=4.41.0", "safetensors>=0.4.0",
            "numpy>=1.24", "sentencepiece", "protobuf", "tokenizers"])
    # 确保使用仓库自带 gguf-py（避免 PyPI 版本不匹配）
    print("[venv] 安装仓库版 gguf-py（可编辑模式）")
    gguf_py = CLONE_DIR / "gguf-py"
    if gguf_py.exists():
        sh([vpy, "-m", "pip", "install", "-e", str(gguf_py)])
    else:
        print("[warn] 未找到 gguf-py 目录，但正常情况下应存在。若后续导入失败，请检查仓库结构。")
    # 一些转换分支会 import mistral-common
    print("[venv] 额外安装 mistral-common（若不需要会跳过）")
    sh([vpy, "-m", "pip", "install", "mistral-common[sentencepiece]"],)


def run_convert(hf_model_dir: Path, fp16_out: Path):
    vpy = venv_python(CONVERT_VENV)
    OUT_DIR_FP16.mkdir(parents=True, exist_ok=True)
    convert_py = CLONE_DIR / "convert_hf_to_gguf.py"
    if not convert_py.exists():
        raise RuntimeError(f"未找到官方转换脚本：{convert_py}")
    # 让 Python 能优先找到仓库内的 gguf-py
    env = os.environ.copy()
    # 不要设置 NO_LOCAL_GGUF；保持默认优先本地 gguf
    print("[1/3] 转换 HF -> GGUF(FP16) …")
    sh([vpy, str(convert_py), str(hf_model_dir),
        "--outtype", "f16",
        "--outfile", str(fp16_out)], env=env)
    print(f"[done] 生成 FP16 GGUF：{fp16_out}")


def build_quant_if_needed() -> Path:
    # 1) 优先使用用户给的预编译二进制
    if PREFERRED_LLAMA_QUANT.is_file():
        print(f"[quant] 使用本地量化工具：{PREFERRED_LLAMA_QUANT}")
        return PREFERRED_LLAMA_QUANT

    # 2) 仓库里构建
    print("[quant] 未发现本地量化工具，尝试从仓库构建 …")
    ensure_cmake()
    build_dir = CLONE_DIR / "build"
    build_bin = build_dir / "bin"
    build_dir.mkdir(exist_ok=True)
    # 仅构建 CPU 版本够用（如需 CUDA，可改为 -DLLAMA_CUBLAS=ON）
    sh(["cmake", "-S", str(CLONE_DIR), "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release"])
    sh(["cmake", "--build", str(build_dir), "--config", "Release", "-j"])
    quant = build_bin / "llama-quantize"
    if not quant.is_file():
        raise RuntimeError("构建完成但未找到 llama-quantize，可查看编译输出是否有报错。")
    print(f"[quant] 构建完成：{quant}")
    return quant


def run_quantize(fp16_path: Path, q4km_out: Path, quant_bin: Path):
    OUT_DIR_Q4KM.mkdir(parents=True, exist_ok=True)
    print("[2/3] 量化 FP16 -> Q4_K_M …")
    sh([str(quant_bin), str(fp16_path), str(q4km_out), "Q4_K_M"])
    print(f"[done] 生成 Q4_K_M GGUF：{q4km_out}")


def quick_smoke_test(q4km_path: Path):
    # 选择可用的 llama-cli：优先用户自己的，否则用仓库构建产物
    cli = None
    if PREFERRED_LLAMA_CLI.is_file():
        cli = PREFERRED_LLAMA_CLI
    else:
        build_cli = CLONE_DIR / "build" / "bin" / "llama-cli"
        if build_cli.is_file():
            cli = build_cli
    if not cli:
        print("[selftest] 未找到 llama-cli，自检跳过。")
        return
    print("[3/3] 自检运行（10 token） …")
    try:
        subprocess.run([str(cli), "-m", str(q4km_path),
                        "-p", "你好，做个自我介绍。", "-n", "10"],
                       check=False)
    except Exception:
        pass


def main():
    # 0) 基本检查
    if not HF_MODEL_DIR.is_dir():
        raise RuntimeError(f"未找到 HF 模型目录：{HF_MODEL_DIR.resolve()}")

    ensure_git()
    clone_llama_cpp()
    ensure_convert_venv()

    # 1) 转换 FP16 GGUF
    run_convert(HF_MODEL_DIR, FP16_GGUF)

    # 2) 量化 Q4_K_M（优先用你自己的量化二进制；否则自动构建）
    quant_bin = build_quant_if_needed()
    run_quantize(FP16_GGUF, Q4KM_GGUF, quant_bin)

    # 3) 可选自检
    quick_smoke_test(Q4KM_GGUF)

    print("\n✅ 全部完成！")
    print(f" - FP16 GGUF: {FP16_GGUF.resolve()}")
    print(f" - Q4_K_M   : {Q4KM_GGUF.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERROR]", e)
        sys.exit(1)
