import os
import torch
import shutil
import re

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

MODEL_ID = "../base_model"
OUT_DIR = "./model"

DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"
SEED = 42

# Calibration pools
CALIB_POOL_START = 0
CALIB_POOL_END = 4096

NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512

SCHEME = "W4A16"
TARGETS = ["Linear"]
KEEP_LAYERS = [0, 1, 2, 3, 4, 26]
BASE_IGNORE = ["embed_tokens", "lm_head"]
EXTRA_IGNORE_SUBSTRINGS = []
USE_STRATIFIED_CALIB = False

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


def load_calib_dataset():
    if not USE_STRATIFIED_CALIB:
        ds = load_dataset(
            DATASET_ID,
            split=f"{DATASET_SPLIT}[{CALIB_POOL_START}:{CALIB_POOL_END}]",
        )
        ds = ds.map(preprocess)
        return ds.shuffle(seed=SEED).select(range(NUM_CALIBRATION_SAMPLES))

    # Stratified: sample equally from three ranges to reduce slice bias
    ranges = [
        (0, 2048),
        (2048, 4096),
        (4096, 6144),
    ]
    per = NUM_CALIBRATION_SAMPLES // len(ranges)
    parts = []
    for start, end in ranges:
        part = load_dataset(
            DATASET_ID,
            split=f"{DATASET_SPLIT}[{start}:{end}]",
        )
        part = part.map(preprocess)
        part = part.shuffle(seed=SEED).select(range(per))
        parts.append(part)
    # top up if rounding left remainder
    if len(parts) > 0 and per * len(ranges) < NUM_CALIBRATION_SAMPLES:
        extra = NUM_CALIBRATION_SAMPLES - (per * len(ranges))
        part = load_dataset(
            DATASET_ID,
            split=f"{DATASET_SPLIT}[{ranges[0][0]}:{ranges[0][1]}]",
        )
        part = part.map(preprocess)
        part = part.shuffle(seed=SEED).select(range(extra))
        parts.append(part)

    from datasets import concatenate_datasets
    return concatenate_datasets(parts).shuffle(seed=SEED)


print("[INFO] Loading calibration slice...")
calib_ds = load_calib_dataset()


def build_ignore_for_keep_layers(model_obj, keep_layers):
    ignore_set = set(BASE_IGNORE)
    layer_pattern = re.compile(r"(?:^|\\.)(?:layers|h)\\.(\\d+)(?:\\.|$)")

    for module_name, module in model_obj.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue

        match = layer_pattern.search(module_name)
        if match and int(match.group(1)) in keep_layers:
            ignore_set.add(module_name)
            continue

        if EXTRA_IGNORE_SUBSTRINGS:
            for sub in EXTRA_IGNORE_SUBSTRINGS:
                if sub in module_name:
                    ignore_set.add(module_name)
                    break

    return sorted(ignore_set)


IGNORE = build_ignore_for_keep_layers(model, KEEP_LAYERS)
print(f"[INFO] keep_layers={KEEP_LAYERS}, extra_ignore={EXTRA_IGNORE_SUBSTRINGS}, ignored_linear_modules={len(IGNORE)}")

recipe = [
    GPTQModifier(
        scheme=SCHEME,
        targets=TARGETS,
        ignore=IGNORE,
    )
]

print(
    f"[INFO] GPTQ start (scheme={SCHEME}, calib={NUM_CALIBRATION_SAMPLES}, max_len={MAX_SEQUENCE_LENGTH})"
)

oneshot(
    model=model,
    dataset=calib_ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

print("[INFO] GPTQ done")

os.makedirs(OUT_DIR, exist_ok=True)
model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

zip_name = "submit_ignore_expanded"
shutil.make_archive(
    base_name=zip_name,
    format="zip",
    root_dir=".",
    base_dir=OUT_DIR,
)

print(f"[INFO] Saved: {OUT_DIR}, {zip_name}.zip")

