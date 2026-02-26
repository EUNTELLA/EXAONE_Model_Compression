# 0213 Baseline GPTQ (calib512)

## 설정
- 양자화 스킴: W4A16 (Linear 타깃), `embed_tokens`, `lm_head` 무시
- 캘리브레이션 데이터: LGAI-EXAONE/MANTA-1M
- 캘리브레이션 샘플: 512, 최대 시퀀스 길이: 512
- torch dtype: bfloat16
- 주요 스크립트: `0213.py`, `0213_keep_0_3_26*.py`, 비교 노트북 `compare_0213.ipynb`

## 결과
- 제출 점수: **0.5886399455** (루트 `README.md`에 기록된 값)
- 추가 지표: 노트북 출력에 저장된 점수 없음 (재실행 필요)

## 환경
- 실행 하드웨어/라이브러리 버전이 노트북에 기록되지 않음. 재현 시 동일 데이터셋 경로(`data/raw/calib_samples_official_256.jsonl`)와 스크립트 설정을 사용하세요.
