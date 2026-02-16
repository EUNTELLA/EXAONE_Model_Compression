import argparse
import time
from statistics import mean

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "대한민국의 수도는 어디인가요? 이유도 간단히 설명해줘.",
    "양자화가 LLM 추론 속도에 미치는 영향을 3가지로 요약해줘.",
    "다음 수열의 규칙을 설명해줘: 2, 4, 8, 16, ...",
    "파이썬에서 리스트와 튜플의 차이를 예시와 함께 알려줘.",
    "인공지능 모델 압축 방법을 비교해줘: pruning, distillation, quantization.",
]


def load_model(model_path: str, dtype: torch.dtype, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


def benchmark(tokenizer, model, device: str, max_new_tokens: int):
    times = []
    new_tokens_list = []

    with torch.no_grad():
        for prompt in PROMPTS:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_len = inputs["input_ids"].shape[-1]

            start = time.perf_counter()
            outputs = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
            elapsed = time.perf_counter() - start

            output_len = outputs.shape[-1]
            new_tokens = max(0, output_len - input_len)

            times.append(elapsed)
            new_tokens_list.append(new_tokens)

    avg_time = mean(times)
    avg_new_tokens = mean(new_tokens_list)
    toks_per_sec = (sum(new_tokens_list) / sum(times)) if sum(times) > 0 else 0.0

    return {
        "avg_time_sec": avg_time,
        "avg_new_tokens": avg_new_tokens,
        "tokens_per_sec": toks_per_sec,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare two quantized models by generation speed.")
    parser.add_argument("--model-a", default="./model_base", help="Path to model A")
    parser.add_argument("--model-b", default="./model_A_calib512", help="Path to model B")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Inference device")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    dtype = torch.bfloat16

    print(f"[INFO] device={args.device}, dtype={dtype}, max_new_tokens={args.max_new_tokens}")

    print(f"[INFO] Loading A: {args.model_a}")
    tok_a, model_a = load_model(args.model_a, dtype, args.device)
    print(f"[INFO] Benchmarking A: {args.model_a}")
    result_a = benchmark(tok_a, model_a, args.device, args.max_new_tokens)

    print(f"[INFO] Loading B: {args.model_b}")
    tok_b, model_b = load_model(args.model_b, dtype, args.device)
    print(f"[INFO] Benchmarking B: {args.model_b}")
    result_b = benchmark(tok_b, model_b, args.device, args.max_new_tokens)

    print("\n=== RESULT ===")
    print(f"A avg_time_sec={result_a['avg_time_sec']:.4f}, avg_new_tokens={result_a['avg_new_tokens']:.2f}, tokens_per_sec={result_a['tokens_per_sec']:.4f}")
    print(f"B avg_time_sec={result_b['avg_time_sec']:.4f}, avg_new_tokens={result_b['avg_new_tokens']:.2f}, tokens_per_sec={result_b['tokens_per_sec']:.4f}")

    if result_a["tokens_per_sec"] > result_b["tokens_per_sec"]:
        print("[WINNER] A is faster")
    elif result_a["tokens_per_sec"] < result_b["tokens_per_sec"]:
        print("[WINNER] B is faster")
    else:
        print("[WINNER] tie")


if __name__ == "__main__":
    main()
