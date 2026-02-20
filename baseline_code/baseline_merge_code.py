import os
import torch
import shutil

from datasets import concatenate_datasets, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

MODEL_ID = "./base_model"
OUT_DIR = "./model_merged"

PRIMARY_DATASET_ID = "LGAI-EXAONE/MANTA-1M"
PRIMARY_DATASET_SPLIT = "train"

# Replace with a license-compliant Hugging Face dataset for your experiments.
SECONDARY_DATASET_ID = "OpenAssistant/oasst1"
SECONDARY_DATASET_SPLIT = "train"

NUM_CALIBRATION_SAMPLES = 256
PRIMARY_RATIO = 0.7
MAX_SEQUENCE_LENGTH = 512

SCHEME = "W4A16"
TARGETS = ["Linear"]
IGNORE = ["embed_tokens", "lm_head"]
SEED = 42


def normalize_messages(messages):
    normalized = []
    for m in messages:
        role = str(m.get("role", "user")).strip().lower()
        content = m.get("content", "")

        if isinstance(content, list):
            text_parts = [str(x.get("text", "")) for x in content if isinstance(x, dict)]
            content = " ".join([t for t in text_parts if t]).strip()
        else:
            content = str(content).strip()

        if not content:
            continue

        if role not in {"system", "user", "assistant"}:
            role = "user"

        normalized.append({"role": role, "content": content})

    return normalized


def build_text_from_secondary(example, tokenizer):
    if isinstance(example.get("conversations"), list):
        messages = normalize_messages(example["conversations"])
    elif isinstance(example.get("messages"), list):
        messages = normalize_messages(example["messages"])
    else:
        prompt = str(example.get("prompt") or example.get("instruction") or "").strip()
        chosen = str(example.get("chosen") or example.get("response") or example.get("output") or "").strip()
        text = str(example.get("text", "")).strip()

        if prompt and chosen:
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": chosen},
            ]
        elif prompt:
            messages = [{"role": "user", "content": prompt}]
        elif text:
            messages = [{"role": "user", "content": text}]
        else:
            messages = []

    if not messages:
        return ""

    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )


print("[INFO] Loading model and tokenizer...")


# For local experimentation; evaluation server is offline.
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
)

print("[INFO] Model/tokenizer loaded")

primary_count = int(NUM_CALIBRATION_SAMPLES * PRIMARY_RATIO)
secondary_count = NUM_CALIBRATION_SAMPLES - primary_count

print(f"[INFO] Loading primary dataset: {PRIMARY_DATASET_ID}")
primary_ds = load_dataset(
    PRIMARY_DATASET_ID,
    split=f"{PRIMARY_DATASET_SPLIT}[:{primary_count}]",
)

print(f"[INFO] Loading secondary dataset: {SECONDARY_DATASET_ID}")
secondary_ds = load_dataset(
    SECONDARY_DATASET_ID,
    split=SECONDARY_DATASET_SPLIT,
)

if len(secondary_ds) < secondary_count:
    secondary_count = len(secondary_ds)
    primary_count = NUM_CALIBRATION_SAMPLES - secondary_count

primary_ds = primary_ds.select(range(min(len(primary_ds), primary_count)))
secondary_ds = secondary_ds.shuffle(seed=SEED).select(range(min(len(secondary_ds), secondary_count)))

print(f"[INFO] Merging datasets (primary={len(primary_ds)}, secondary={len(secondary_ds)})")

primary_ds = primary_ds.map(
    lambda ex: {
        "text": tokenizer.apply_chat_template(
            ex["conversations"],
            add_generation_prompt=True,
            tokenize=False,
        )
    }
)

secondary_ds = secondary_ds.map(
    lambda ex: {"text": build_text_from_secondary(ex, tokenizer)}
)

primary_ds = primary_ds.remove_columns([c for c in primary_ds.column_names if c != "text"])
secondary_ds = secondary_ds.remove_columns([c for c in secondary_ds.column_names if c != "text"])
secondary_ds = secondary_ds.filter(lambda ex: bool(str(ex["text"]).strip()))

calibration_ds = concatenate_datasets([primary_ds, secondary_ds]).shuffle(seed=SEED)

print(
    f"[INFO] GPTQ start (scheme={SCHEME}, samples={len(calibration_ds)}, max_len={MAX_SEQUENCE_LENGTH})"
)

recipe = [
    GPTQModifier(
        scheme=SCHEME,
        targets=TARGETS,
        ignore=IGNORE,
    )
]

oneshot(
    model=model,
    dataset=calibration_ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=len(calibration_ds),
)

print("[INFO] GPTQ done")

os.makedirs(OUT_DIR, exist_ok=True)
model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

zip_name = "baseline_submit_merged"
shutil.make_archive(
    base_name=zip_name,
    format="zip",
    root_dir=".",
    base_dir=OUT_DIR,
)

print(f"[INFO] Saved: {OUT_DIR}, {zip_name}.zip")
