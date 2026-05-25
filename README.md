# EXAONE Model Compression

LG Aimers 8기 모델 경량화 과제를 위한 EXAONE 압축 실험 저장소입니다. `EXAONE-4.0-1.2B` 계열 모델을 대상으로 GPTQ/AWQ 기반 양자화, calibration 데이터 구성, keep layer 전략, Q/K projection 보존 전략을 실험하며, 성능 저하를 줄이면서 모델 크기와 추론 시간을 낮추는 것을 목표로 합니다.

## 프로젝트 목표

이 프로젝트의 핵심 목표는 제한된 온디바이스/서빙 환경에서도 사용할 수 있는 경량 LLM을 만드는 것입니다.

- 원본 EXAONE 모델 대비 모델 크기 축소
- 추론 시간 감소
- 리더보드 성능 점수 유지
- 제출 제한 조건 충족
- calibration 데이터와 layer 보존 전략에 따른 성능 변화 비교

최종 제출 모델은 전체 평가를 제한 시간 안에 완료해야 하며, 제출 압축 파일 크기 제한도 고려해야 합니다.

## 주요 접근 방법

### GPTQ 양자화

기본 실험은 `llmcompressor`의 `GPTQModifier`를 사용합니다.

- quantization scheme: `W4A16`
- target module: `Linear`
- 기본 ignore module: `embed_tokens`, `lm_head`
- calibration dataset: `LGAI-EXAONE/MANTA-1M`
- calibration sample: 주로 256 또는 512
- max sequence length: 256 또는 512

### AWQ 실험

`src/awq_code.py`에서 AWQ 기반 압축 실험을 수행합니다.

- W4A16 설정
- 일부 layer 보존
- calibration/eval slice 분리
- bf16 기반 모델 로드

### Keep Layer 전략

특정 layer는 양자화 대상에서 제외하여 성능 손실을 줄이는 전략을 사용합니다.

대표 설정:

```txt
keep_layers = [0, 1, 2, 3, 4, 26]
```

또는 실험별로 `[0-4, 26]`, `[0-2, 26]`, `[0-1, 28]` 같은 조합을 비교합니다.

### Mixed Calibration

0225 실험에서는 official calibration jsonl과 MANTA-1M 데이터를 섞어 사용합니다.

예시 preset:

- `mix50_50`
- `mix70_30`
- `mix30_70`

혼합 비율을 바꿔 calibration 데이터 분포가 성능과 추론 속도에 미치는 영향을 확인합니다.

### Q/K Projection 보존

`qk_experiments/`에서는 `q_proj`, `k_proj` 계열 module을 추가로 ignore하는 전략을 실험합니다. attention 품질에 민감한 projection을 보존해 성능 하락을 줄이는 방향의 실험입니다.

## 저장소 구조

```txt
.
├── README.md
├── src/
│   ├── baseline_code.py
│   ├── baseline_code_exp.py
│   ├── baseline_merge_code.py
│   ├── awq_code.py
│   ├── compare_models.py
│   └── error_initialization_test.py
├── experiments/
│   ├── 0213/
│   ├── 0222/
│   └── 0225/
├── qk_experiments/
│   ├── qk_experiments_v1/
│   ├── qk_experiments_v2/
│   ├── qk_experiment_eval.ipynb
│   └── qk_experiment_eval_v2.ipynb
├── comparetxt/
└── drawchart.png
```

## 디렉터리 설명

### `src/`

공통 실험 코드와 비교 도구가 들어 있습니다.

- `baseline_code.py`
  - 기본 GPTQ W4A16 양자화 코드입니다.
  - MANTA-1M calibration 데이터를 사용합니다.

- `baseline_code_exp.py`
  - baseline 양자화 후 모델 저장과 zip 생성까지 수행하는 제출용 흐름의 코드입니다.

- `baseline_merge_code.py`
  - MANTA-1M과 보조 데이터셋을 섞어 calibration dataset을 구성하는 실험 코드입니다.

- `awq_code.py`
  - AWQ 기반 압축 실험 코드입니다.
  - keep layer와 eval slice를 함께 다룹니다.

- `compare_models.py`
  - 두 모델의 생성 속도를 간단한 prompt 세트로 비교하는 벤치마크 스크립트입니다.

- `error_initialization_test.py`
  - 초기화 또는 로딩 관련 문제를 확인하기 위한 테스트 코드입니다.

### `experiments/0213/`

초기 baseline GPTQ 실험 폴더입니다.

- W4A16 GPTQ
- MANTA-1M calibration
- `keep_0_3_26` 계열 실험
- baseline 제출 점수 기록
- 비교 노트북 포함

주요 파일:

```txt
0213.py
0213_keep_0_3_26.py
0213_keep_0_3_26_stratified.py
compare_0213.ipynb
README.md
```

### `experiments/0222/`

평가 로그와 빠른 비교 노트북이 들어 있는 폴더입니다.

- vLLM 평가 설정
- Colab/Tesla T4 평가 로그
- `out_q_0225_mix50_50` 평가 결과
- token length, ppl proxy, test time 기록

주요 파일:

```txt
eval_0222.ipynb
eval_fast_compare.ipynb
README.md
```

### `experiments/0225/`

mixed calibration 실험 폴더입니다.

기준은 0213 recipe이며, 주요 변경점은 다음과 같습니다.

- `actorder=static`에서 `actorder=weight`로 변경
- official jsonl과 MANTA-1M calibration 데이터 혼합
- 비율별 preset 제공

주요 파일:

```txt
recipe_0225_mixed.py
0225_mix50_50.py
0225_mix70_30.py
0225_mix30_70.py
README.md
```

### `qk_experiments/`

Q/K projection과 layer keep 조합을 탐색하는 실험 폴더입니다.

- `qk_experiments_v1/`
  - keep layer 조합 중심 실험

- `qk_experiments_v2/`
  - stratified calibration
  - combined ignore 전략
  - expanded ignore 전략

대표 파일:

```txt
exp_keep_0_3_26.py
exp_keep_0_4_25.py
exp_keep_0_4_27.py
exp_keep_0_5_26.py
exp_calib_stratified.py
exp_combined.py
exp_ignore_expanded.py
```

## 실험 결과 요약

현재 README와 실험 문서에 기록된 주요 결과는 다음과 같습니다.

| 순위 | 실험명 | 주요 방법 | LB Score | Size |
| ---: | --- | --- | ---: | ---: |
| 1 | `0213_version1` | W4A16, keep layers `[0-4, 26]`, calib 256 | 0.6162 | 1.14GB |
| 2 | `mixed_actorder_weight` | `act_order=weight`, mixed calibration | 0.6156 | - |
| 3 | `qk_pA_d010` | q/k keep `[0-2, 26]` | 0.6132 | 1.14GB |
| 4 | `qk_pB_d010` | q/k keep `[0-1, 28]` | 0.6127 | 1.14GB |
| 5 | `GPTQ_bs64_act_static` | GPTQ, `bs=64`, `act_order=static` | 0.5901 | - |
| 6 | `calib512` | baseline W4A16, calib 512 | 0.5886 | - |
| 7 | `strat_global` | stratified global sampling | 0.5879 | - |

0222 평가 로그에 기록된 `mix50_50` 모델 평가 결과:

| model | submission_formula_score | test_time_sec | ppl_proxy | avg output tokens |
| --- | ---: | ---: | ---: | ---: |
| `out_q_0225_mix50_50` | 0.18919 | 357.796 | 10.280813 | 584.83 |

## 실행 환경

기록된 기준 환경은 다음과 같습니다.

```txt
OS: Ubuntu 22.04
GPU: NVIDIA L4 22.4GB VRAM
Python: 3.11.1
torch: 2.9.0+cu128
transformers: 4.57.3
vllm: 0.14.1
safetensors: 0.7.0
accelerate: 1.10.1
```

일부 평가는 Google Colab Tesla T4 환경에서도 수행되었습니다.

## 기본 사용법

### 1. 모델 준비

실험 스크립트는 대체로 로컬 경로의 base model을 기준으로 작성되어 있습니다.

```txt
./base_model
../base_model
```

실행 위치에 따라 필요한 경로가 다르므로, 각 스크립트의 `MODEL_ID`를 먼저 확인하세요.

### 2. Baseline GPTQ 실행

루트 기준:

```bash
python src/baseline_code.py
```

제출 zip 생성 흐름까지 확인하려면:

```bash
python src/baseline_code_exp.py
```

### 3. AWQ 실행

```bash
python src/awq_code.py
```

### 4. 0225 mixed calibration 실행

```bash
cd experiments/0225
python 0225_mix50_50.py
python 0225_mix70_30.py
python 0225_mix30_70.py
```

직접 비율을 지정하려면:

```bash
python recipe_0225_mixed.py --official-ratio 60 --manta-ratio 40 --tag mix60_40
```

### 5. 모델 속도 비교

```bash
python src/compare_models.py \
  --model-a ./model_base \
  --model-b ./model_A_calib512 \
  --device cuda \
  --max-new-tokens 128
```

## 평가 방식

평가는 다음 요소를 함께 봅니다.

- 리더보드 성능 점수
- 추론 시간
- 모델 압축 파일 크기
- proxy perplexity
- output token length 통계
- finish reason 분포

평가 관련 노트북:

```txt
experiments/0222/eval_0222.ipynb
experiments/0222/eval_fast_compare.ipynb
qk_experiments/qk_experiment_eval.ipynb
qk_experiments/qk_experiment_eval_v2.ipynb
```

## 실험 흐름 추천

새 실험을 추가할 때는 다음 순서로 진행하는 것이 좋습니다.

1. baseline recipe를 복사해 새 실험 파일 생성
2. keep layer, calibration ratio, actorder, ignore module만 변경
3. 모델 저장 경로를 실험명 기준으로 분리
4. 동일 평가 노트북으로 test time과 score 측정
5. 각 실험 폴더의 README에 결과 기록
6. 필요하면 `comparetxt/`에 metric 로그 저장

## 주의사항

- Hugging Face dataset과 model 접근 권한이 필요할 수 있습니다.
- 일부 스크립트는 `local_files_only=True`를 사용하므로 모델 파일이 로컬에 있어야 합니다.
- 실험마다 `MODEL_ID`, `OUT_DIR`, calibration jsonl 경로가 다릅니다.
- 제출용 zip 생성 시 파일 크기 제한을 확인해야 합니다.
- Colab/T4와 L4 환경은 bf16/fp16 지원 차이가 있으므로 dtype 설정을 확인해야 합니다.

## 현재 문서

프로젝트 문서는 다음 위치에 나뉘어 있습니다.

```txt
README.md
experiments/0213/README.md
experiments/0222/README.md
experiments/0225/README.md
docs/README.md
```

루트 `README.md`는 프로젝트 전체 설명을 담당하고, 날짜별 README는 개별 실험 로그와 실행 방법을 담당합니다.

