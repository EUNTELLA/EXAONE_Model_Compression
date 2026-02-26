#!/usr/bin/env python3
import json
from pathlib import Path

import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# 0213-based settings with keep 0:3,26 + stratified calib
MODEL_ID = "../base_model"
OUT_DIR = Path("models/out_q_keep_0_3_26_stratified")
KEEP_RANGES = [(0, 4), (26, 27)]  # [start, end)
NUM_SAMPLES, MAX_SEQ_LEN = 256, 256
SEED = 42

DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"
RANGES = [(0, 2048), (2048, 4096), (4096, 6144)]


def keep_layer_ignores(model, keep_ranges):
    keep_idx = {i for s, e in keep_ranges for i in range(s, e)}
    ignores = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            if any(name.startswith(f"model.layers.{i}.") or name.startswith(f"layers.{i}.") for i in keep_idx):
                ignores.append(name)
    return sorted(ignores)


def preprocess(tokenizer):
    def _fn(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["conversations"],
                add_generation_prompt=True,
                tokenize=False,
            )
        }
    return _fn


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, local_files_only=True)

per = NUM_SAMPLES // len(RANGES)
parts = []
for start, end in RANGES:
    part = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[{start}:{end}]")
    part = part.map(preprocess(tokenizer))
    part = part.shuffle(seed=SEED).select(range(per))
    parts.append(part)

# top up if rounding left remainder
if per * len(RANGES) < NUM_SAMPLES:
    extra = NUM_SAMPLES - (per * len(RANGES))
    part = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[{RANGES[0][0]}:{RANGES[0][1]}]")
    part = part.map(preprocess(tokenizer))
    part = part.shuffle(seed=SEED).select(range(extra))
    parts.append(part)

ds = concatenate_datasets(parts).shuffle(seed=SEED)

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
    "keep_layers": "0:3,26", "num_samples": 256, "max_seq_len": 256,
    "calib": "stratified(0-2048,2048-4096,4096-6144)"
}, indent=2))
print(f"[OK] saved -> {OUT_DIR}")
