# 0222 평가 로그

## 입력 데이터
- `data/raw/eval_samples_official_512.jsonl`

## 환경
- Google Colab, GPU: Tesla T4 (bf16 미지원으로 fp16 폴백 메시지 확인)
- vLLM 실행 시 `gpu_memory_utilization=0.85`, `trust_remote_code=True`, `enforce_eager=True`

## 평가 결과
| model_name | model_dir | status | submission_formula_score | test_time_sec | ppl_proxy | out_tok_mean | p50 | p90 | p95 | max | length_finish_ratio | finish_reason_counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_anchor | `/content/drive/MyDrive/model/out_q_0225_mix50_50` | success | **0.18919** | 357.796 | 10.280813 | 584.83 | 622 | 964 | 1045 | 2012 | 0.0 | {"stop": 512} |

> 원본 표는 `eval_0222.ipynb` 출력에서 가져왔습니다. 다른 체크포인트는 실행되지 않았거나 표가 저장되지 않았습니다.

## 다음 액션
- `out_q_0225_mix30_70` 등 다른 모델도 동일 노트북에서 실행 후 표를 추가하세요.
- 결과 CSV로 내보내려면 노트북의 `FINAL_CSV` 경로를 설정해 저장하십시오.
