import os
import json
import shutil
import re

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier

MODEL_ID = "./base_model"
OUT_DIR = "./model"

DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"
SEED = 42

CALIB_POOL_START = 0
CALIB_POOL_END = 4096
EVAL_POOL_START = 5000
EVAL_POOL_END = 9096

NUM_CALIBRATION_SAMPLES = 256
NUM_EVAL_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 512

SCHEME = "W4A16"
TARGETS = ["Linear"]
KEEP_LAYERS = list(range(0, 5)) + [26]
BASE_IGNORE = ["embed_tokens", "lm_head"]

print("[INFO] Loading model/tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
)
model.to("cpu")

print("[INFO] Model/tokenizer loaded (cpu + bf16)")


def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["conversations"],
            add_generation_prompt=True,
            tokenize=False,
        )
    }


print("[INFO] Loading calibration slice...")
calib_ds = load_dataset(
    DATASET_ID,
    split=f"{DATASET_SPLIT}[{CALIB_POOL_START}:{CALIB_POOL_END}]",
)
calib_ds = calib_ds.map(preprocess)
calib_ds = calib_ds.shuffle(seed=SEED).select(range(NUM_CALIBRATION_SAMPLES))

print("[INFO] Loading eval slice...")
eval_ds = load_dataset(
    DATASET_ID,
    split=f"{DATASET_SPLIT}[{EVAL_POOL_START}:{EVAL_POOL_END}]",
)
eval_ds = eval_ds.map(preprocess)
eval_ds = eval_ds.shuffle(seed=SEED).select(range(NUM_EVAL_SAMPLES))


def build_ignore_for_keep_layers(model_obj, keep_layers):
    ignore_set = set(BASE_IGNORE)
    layer_pattern = re.compile(r"(?:^|\\.)(?:layers|h)\\.(\\d+)(?:\\.|$)")

    for module_name, module in model_obj.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue

        match = layer_pattern.search(module_name)
        if match and int(match.group(1)) in keep_layers:
            ignore_set.add(module_name)

    return sorted(ignore_set)


IGNORE = build_ignore_for_keep_layers(model, KEEP_LAYERS)
print(f"[INFO] keep_layers={KEEP_LAYERS}, ignored_linear_modules={len(IGNORE)}")

recipe = [
    AWQModifier(
        scheme=SCHEME,
        targets=TARGETS,
        ignore=IGNORE,
    )
]

print(
    f"[INFO] AWQ start (scheme={SCHEME}, calib={NUM_CALIBRATION_SAMPLES}, eval={NUM_EVAL_SAMPLES}, max_len={MAX_SEQUENCE_LENGTH})"
)

oneshot(
    model=model,
    dataset=calib_ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

print("[INFO] AWQ done")

os.makedirs(OUT_DIR, exist_ok=True)
model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

with open(os.path.join(OUT_DIR, "eval_subset.jsonl"), "w", encoding="utf-8") as f:
    for row in eval_ds:
        f.write(json.dumps({"text": row["text"]}, ensure_ascii=False) + "\n")

zip_name = "baseline_submit"
shutil.make_archive(
    base_name=zip_name,
    format="zip",
    root_dir=".",
    base_dir=OUT_DIR,
)

print(f"[INFO] Saved: {OUT_DIR}, {zip_name}.zip")
