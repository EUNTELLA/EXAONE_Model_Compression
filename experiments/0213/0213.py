#!/usr/bin/env python3
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# 0213 official settings
MODEL_ID = "../base_model"
CALIB_JSONL = "calib_samples_official_256.jsonl"
OUT_DIR = Path("models/out_q_official_keep_0_4_26_30")
KEEP_RANGES = [(0, 4), (26, 30)]  # [start, end)
NUM_SAMPLES, MAX_SEQ_LEN = 256, 256

def keep_layer_ignores(model, keep_ranges):
    keep_idx = {i for s, e in keep_ranges for i in range(s, e)}
    ignores = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            if any(name.startswith(f"model.layers.{i}.") or name.startswith(f"layers.{i}.") for i in keep_idx):
                ignores.append(name)
    return sorted(ignores)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, local_files_only=True)
ds = load_dataset("json", data_files=CALIB_JSONL, split="train")

ignore = ["embed_tokens", "lm_head"] + keep_layer_ignores(model, KEEP_RANGES)
recipe = [GPTQModifier(
    scheme="W4A16",
    targets=["Linear"],
    ignore=ignore,
    block_size=128,
    dampening_frac=0.01,
    actorder="static",
    offload_hessians=False,
)]

oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=MAX_SEQ_LEN, num_calibration_samples=NUM_SAMPLES)
OUT_DIR.mkdir(parents=True, exist_ok=True)
model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)
(OUT_DIR / "recipe.json").write_text(json.dumps({
    "scheme": "W4A16", "targets": "Linear", "ignore": ignore,
    "block_size": 128, "dampening_frac": 0.01, "actorder": "static",
    "keep_layers": "0:4,26:30", "num_samples": 256, "max_seq_len": 256
}, indent=2))
print(f"[OK] saved -> {OUT_DIR}")
