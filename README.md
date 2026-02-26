# GPTQ 실험 리포지토리 정리본 (2026-02-26)

## 리포 구조
- `src/` : 베이스라인/비교 스크립트(`baseline_*.py`, `awq_code.py`, `compare_models.py`, `error_initialization_test.py`).
- `experiments/` : 날짜별 실험 자산
  - `0213/` : 원본 GPTQ 실험 스크립트와 비교 노트북.
  - `0222/` : 평가 노트북(`eval_0222.ipynb`, `eval_fast_compare.ipynb`)과 샘플 테스트 스크립트.
  - `0225/` : 혼합 캘리브레이션 레시피(`0225_mix*.py`, `recipe_0225_mixed.py`, `results_template.csv`, `README.md`).
- `data/raw/` : 공식·혼합 캘리브레이션/평가 JSONL (`calib_samples_official_256.jsonl`, `eval_samples_official_512.jsonl`).
- `outputs/` : 제출 zip 아카이브 (`submit.zip`, `submit_20240219.zip`).
- `qk_experiments/` : QK 실험 세트(v1, v2)와 각 제출 zip.
- `baseline_submit/`, `submit/`, `diag/.../model/`, `base_model/` : 대용량 모델 가중치 및 설정.

## 주요 실험 요약
- **0213 baseline GPTQ (W4A16, calib512)**  
  - 스킴: W4A16, Linear 타깃, `embed_tokens`, `lm_head` 제외, bf16.  
  - 캘리브레이션: LGAI-EXAONE/MANTA-1M, 샘플 512, max seq 512.  
  - 점수: **0.5886399455** (루트 결과 기록).  
  - 관련 파일: `experiments/0213/0213*.py`, `compare_0213.ipynb`, `models/` 하위 체크포인트.

### 추가 실험 메모
- **0222**  
  - 평가 전용 노트북(`eval_0222.ipynb`, `eval_fast_compare.ipynb`), 테스트 스크립트 `test_0225_mix50`.  
  - 사용 데이터: `data/raw/eval_samples_official_512.jsonl`.
- **0225 혼합 캘리브레이션**  
  - `0225_mix50_50.py`, `0225_mix70_30.py`, `0225_mix30_70.py` 프리셋.  
  - 커스텀 비율: `python recipe_0225_mixed.py --official-ratio 60 --manta-ratio 40 --tag mix60_40`.  
  - 결과 비교용 템플릿: `results_template.csv`. (점수 미기록 상태)
- **QK experiments (v1/v2)**  
  - 다양한 keep 비율/조합 실험 스크립트와 대응 제출 zip 보관.  
  - 세부 점수는 zip/노트북에 미포함; 필요 시 재평가 필요.

## 실행 메모
- 0225 예시:
  ```
  cd experiments/0225
  python .\0225_mix50_50.py
  ```
- 평가 노트북 실행 시 `data/raw` 경로를 맞춰 로딩.

## Git 관리 권장
- `.gitignore`: `data/`, `outputs/`, `models/*.safetensors`, `*.jsonl`, `*.csv`(산출물), `__pycache__/`, `*.ipynb_checkpoints/`.
- `.gitattributes`: `*.safetensors filter=lfs diff=lfs merge=lfs -text` 추가해 대용량 가중치를 LFS로 관리.

## 앞으로 할 일
- 각 실험 점수와 환경(시드, GPU, 패키지 버전)을 `experiments/<날짜>/README.md`에 채워 넣기.
- 공개 레포로 올릴 때는 가중치(`model.safetensors`)와 제출 zip을 LFS 또는 릴리스 자산으로 분리.
