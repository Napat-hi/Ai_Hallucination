
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = '/workspace/models/qwen2.5-7b-instruct'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map='auto'
)
model.eval()
print('✅ สำเร็จ!')
print(f'VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB')
