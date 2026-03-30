# EXAONE 모델 경량화 해커톤 (LG Aimers 8기)

## 1. 프로젝트 목표 (Project Goal)

본 프로젝트는 LG Aimers 8기 온라인 해커톤 과제입니다. On-Device AI 환경의 제약(지연, 메모리, 비용)을 해결하기 위해, **`EXAONE-4.0-1.2B`** 모델을 대상으로 성능 저하를 최소화하면서 모델 크기를 줄이는 경량화 기법을 탐구하고 적용하는 것을 목표로 합니다.

## 2. 평가 방식 (Evaluation)

리더보드 점수는 아래 두 가지 지표의 비율을 조합하여 산정됩니다.

*   **성능 비율:** 기본 모델(`EXAONE-4.0-1.2B`) 대비 경량화된 모델의 성능
*   **추론 시간 감소 비율:** 기본 모델 대비 토큰당 추론 시간 감소율

최종 제출 모델은 20분 내에 전체 추론을 완료해야 하며, 압축된 `submit.zip` 파일은 10GB를 초과할 수 없습니다.

## 3. 실험 요약 및 결과 (Experiments & Results)
<img width="1189" height="581" style="width: 80%" alt="drawchart" src="https://github.com/user-attachments/assets/5b75a805-41b6-44a5-a1c6-cfaabaf8dcbf" />

각 실험의 성능, 속도, 크기 간의 트레이드오프를 비교하기 위한 종합 결과입니다.

| 순위 | 실험명 (Experiment) | 주요 방법론 (Methodology) | LB Score | Size (GB) |
|:---:|:---|:---|:---:|:---:|
| 1 | `0213_version1` | W4A16, `keep_layers`:[0-4, 26], calib:256 | **0.6162** | 1.14 |
| 2 | `mixed_actorder_weight`| `act_order=weight`, Mixed-Data Calib | **0.6156** | - |
| 3 | `qk_pA_d010` | `qk_keep`:[0-2, 26] | **0.6132** | 1.14 |
| 4 | `qk_pB_d010` | `qk_keep`:[0-1, 28] | **0.6127** | 1.14 |
| 5 | `GPTQ_bs64_act_static`| GPTQ, `bs=64`, `act_order=static` | **0.5901** | - |
| 6 | `calib512` | **Baseline**: W4A16, `calib=512` | **0.5886** | - |
| 7 | `strat_global` | Stratified Global Sampling | **0.5879** | - |

*   **자동 평가 점수 / 처리 속도:** `qk_experiments/qk_experiment_eval.ipynb`의 로컬 벤치마크 결과입니다.
*   **모델 크기:** 제출용 `submit.zip` 파일 기준입니다.

## 4. 재현 환경 (Environment)

해커톤 평가 서버와 동일한 환경을 구성하기 위한 주요 라이브러리 목록입니다.

*   **OS:** Ubuntu 22.04
*   **GPU:** NVIDIA L4 (22.4GiB VRAM)
*   **Python:** 3.11.1
*   **Key Libraries:**
    *   `torch==2.9.0+cu128`
    *   `transformers==4.57.3`
    *   `vllm==0.14.1`
    *   `safetensors==0.7.0`
    *   `accelerate==1.10.1`
