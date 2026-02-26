#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from datasets import concatenate_datasets, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# 0225: 0213 recipe + actorder=weight + mixed calibration dataset
MODEL_ID = "../base_model"
OFFICIAL_JSONL = "../0213/calib_samples_official_256.jsonl"
DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"

KEEP_RANGES = [(0, 4), (26, 30)]  # [start, end)
NUM_SAMPLES = 256
MAX_SEQ_LEN = 256
SEED = 42


def keep_layer_ignores(model, keep_ranges):
    keep_idx = {i for s, e in keep_ranges for i in range(s, e)}
    ignores = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            if any(
                name.startswith(f"model.layers.{i}.") or name.startswith(f"layers.{i}.")
                for i in keep_idx
            ):
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


def build_mixed_dataset(tokenizer, official_ratio, manta_ratio):
    if official_ratio + manta_ratio != 100:
        raise ValueError("official_ratio + manta_ratio must equal 100")

    n_official = int(round(NUM_SAMPLES * official_ratio / 100.0))
    n_manta = NUM_SAMPLES - n_official

    official_ds = load_dataset("json", data_files=OFFICIAL_JSONL, split="train")
    official_ds = official_ds.shuffle(seed=SEED).select(range(n_official)) if n_official > 0 else None

    manta_ds = None
    if n_manta > 0:
        manta_pool = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[0:8192]")
        manta_pool = manta_pool.map(preprocess(tokenizer))
        manta_ds = manta_pool.shuffle(seed=SEED).select(range(n_manta))

    if official_ds is None:
        mixed = manta_ds
    elif manta_ds is None:
        mixed = official_ds
    else:
        mixed = concatenate_datasets([official_ds, manta_ds]).shuffle(seed=SEED)

    return mixed, n_official, n_manta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-ratio", type=int, required=True, help="Official jsonl ratio (0~100)")
    parser.add_argument("--manta-ratio", type=int, required=True, help="MANTA ratio (0~100)")
    parser.add_argument("--tag", type=str, default="", help="Optional output tag")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )

    ds, n_official, n_manta = build_mixed_dataset(
        tokenizer,
        args.official_ratio,
        args.manta_ratio,
    )

    ignore = ["embed_tokens", "lm_head"] + keep_layer_ignores(model, KEEP_RANGES)

    recipe = [
        GPTQModifier(
            scheme="W4A16",
            targets=["Linear"],
            ignore=ignore,
            block_size=128,
            dampening_frac=0.01,
            actorder="weight",
            offload_hessians=False,
        )
    ]

    suffix = args.tag if args.tag else f"mix_o{args.official_ratio}_m{args.manta_ratio}"
    out_dir = Path(f"models/out_q_0225_{suffix}")

    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQ_LEN,
        num_calibration_samples=NUM_SAMPLES,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, save_compressed=True)
    tokenizer.save_pretrained(out_dir)

    (out_dir / "recipe.json").write_text(
        json.dumps(
            {
                "base": "0213.py",
                "scheme": "W4A16",
                "targets": "Linear",
                "ignore": ignore,
                "block_size": 128,
                "dampening_frac": 0.01,
                "actorder": "weight",
                "keep_layers": "0:4,26:30",
                "num_samples": NUM_SAMPLES,
                "max_seq_len": MAX_SEQ_LEN,
                "calib": {
                    "official_jsonl": OFFICIAL_JSONL,
                    "dataset_id": DATASET_ID,
                    "dataset_split": DATASET_SPLIT,
                    "official_ratio": args.official_ratio,
                    "manta_ratio": args.manta_ratio,
                    "n_official": n_official,
                    "n_manta": n_manta,
                },
            },
            indent=2,
        )
    )

    print(f"[OK] saved -> {out_dir}")


if __name__ == "__main__":
    main()
