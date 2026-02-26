# 0225 Experiments

Base recipe: same as `0213/0213.py`, with these changes only:
- `actorder`: `static` -> `weight`
- calibration data: mixed (official jsonl + MANTA-1M)

## Presets
- `0225_mix50_50.py`
- `0225_mix70_30.py`
- `0225_mix30_70.py`

## Run
```powershell
cd 0225
python .\0225_mix50_50.py
python .\0225_mix70_30.py
python .\0225_mix30_70.py
```

## Custom ratio
```powershell
python .\recipe_0225_mixed.py --official-ratio 60 --manta-ratio 40 --tag mix60_40
```

## Compare sheet
Fill `results_template.csv` with lm_eval results for quick comparison.

## 결과
- `mix50_50` 모델(`out_q_0225_mix50_50`) 평가 결과: **submission_formula_score 0.18919**, `test_time_sec` 357.796, `ppl_proxy` 10.280813 (Tesla T4, fp16). 세부 토큰 통계: 평균 584.83 / p50 622 / p90 964 / p95 1045 / 최대 2012, `length_finish_ratio` 0.0.  
  - 측정 로그 출처: `experiments/0222/eval_0222.ipynb`.
- `mix70_30`, `mix30_70` 등은 아직 기록 없음 → `eval_0222.ipynb` 혹은 `eval_fast_compare.ipynb`로 실행 후 `results_template.csv`에 추가하세요.
