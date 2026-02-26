import os, json, zipfile, shutil, traceback
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# zip 파일은 전달받은 이름을 그대로 사용하되, 현재 작업 디렉터리/절대경로 모두 지원
ZIP = Path("out_q_keep_0_3_26_stratified.zip")
if not ZIP.exists():
    ZIP = Path(r"C:\Users\FORYOUCOM\Downloads\testdata\0213\models\out_q_keep_0_3_26_stratified.zip")

work = Path("../diag")
if work.exists():
    shutil.rmtree(work)
work.mkdir(parents=True, exist_ok=True)

if not ZIP.exists():
    raise FileNotFoundError(f"zip not found: {ZIP}")

with zipfile.ZipFile(ZIP, "r") as z:
    z.extractall(work)

# Auto-detect model folder inside the extracted contents.
model_dir = work / "model"
if not model_dir.exists():
    dirs = [p for p in work.iterdir() if p.is_dir()]
    if len(dirs) == 1:
        model_dir = dirs[0]
    else:
        raise RuntimeError(f"model folder not found. extracted dirs: {dirs}")

print("model_dir:", model_dir, "exists:", model_dir.exists())

# 1) 필수 파일 체크
req = ["config.json", "model.safetensors", "tokenizer.json"]
for f in req:
    p = model_dir / f
    print(f"[REQ] {f} exists={p.exists()} size={p.stat().st_size if p.exists() else None}")

# 2) config/quantization_config 확인
cfg = json.load(open(model_dir/"config.json", "r", encoding="utf-8"))
print("\n[config] model_type:", cfg.get("model_type"))
print("[config] architectures:", cfg.get("architectures"))
print("[config] torch_dtype:", cfg.get("torch_dtype"))
print("[config] max_position_embeddings:", cfg.get("max_position_embeddings"))
print("[config] has quantization_config:", "quantization_config" in cfg)

if "quantization_config" in cfg:
    qc = cfg["quantization_config"]
    print("[quantization_config] quant_method:", qc.get("quant_method"))
    print("[quantization_config] format:", qc.get("format"))
    print("[quantization_config] version:", qc.get("version"))
    # fp8인지/아닌지 단서
    print("[quantization_config] keys:", list(qc.keys())[:20])

# 3) tokenizer 관련 파일 간단 정합성
def jload(name):
    p = model_dir/name
    return json.load(open(p, "r", encoding="utf-8")) if p.exists() else None

tconf = jload("tokenizer_config.json")
stmap = jload("special_tokens_map.json")

print("\n[tokenizer_config.json]", tconf)
print("[special_tokens_map.json]", stmap)

# 4) transformers 로드 + 1-step forward (여기서 죽으면 vLLM 이전에 정합성/포맷 문제)
print("\n[TRANSFORMERS LOAD TEST]")
try:
    tok = AutoTokenizer.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,  # ByteLevel item assignment 오류 회피
    )
    m = AutoModelForCausalLM.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
    m.eval()

    x = tok("hello", return_tensors="pt")
    with torch.no_grad():
        out = m(**x)
    print("✅ transformers load + forward OK | logits:", tuple(out.logits.shape))
except Exception as e:
    print("❌ transformers failed:", repr(e))
    traceback.print_exc(limit=2)

# 5) vLLM 로드(너가 이미 한 테스트)
print("\n[VLLM LOAD TEST]")
try:
    from vllm import LLM
    llm = LLM(model=str(model_dir), disable_log_stats=True)
    print("✅ vLLM load OK")
except Exception as e:
    print("❌ vLLM failed:", repr(e))
