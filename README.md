# GPTQ W4A16 Experiment (calib512)

## Settings
- Scheme: W4A16
- Targets: Linear
- Ignore: embed_tokens, lm_head
- Dataset: LGAI-EXAONE/MANTA-1M
- Calibration Samples: 512
- Max Seq Length: 512
- Torch dtype: bfloat16

## Result
- Score: 0.5886399455

## Notes
- Baseline GPTQ oneshot quantization.
