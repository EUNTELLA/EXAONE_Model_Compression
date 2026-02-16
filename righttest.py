import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "./submit/model"  # 실제 경로로 수정

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

prompt = "대한민국의 수도는 어디인가?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

start = time.time()
output = model.generate(**inputs, max_new_tokens=100)
end = time.time()

print("추론 시간:", end - start)
