

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch
import time   # ← thêm
import gc

model_name = "/home/clara/phucth/code/project/public_dataset_med/public_data_medical/Qwen2-VL-7B-multi-137k_full_maed_v3"

PROMPT = "Ảnh Chụp Xquang ngực thẳng, bệnh nhân Nam. Cho biết bệnh nhân bị gì?"
# image_path = "/home/clara/phucth/code/project/Clara_medical_v1/infer/demo_clara/image_test/image_bt.png"
image_path = "/home/clara/phucth/code/project/Medical_CLARA/infer/demo_clara/image_test/image(4).png"
min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28

# Load model
model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, device_map="auto")
processor = AutoProcessor.from_pretrained(model_name, min_pixels=min_pixels, max_pixels=max_pixels)
model.eval()

image = Image.open(image_path).convert("RGB").resize((448, 448))

conversation = []

# ========= TURN 1 =========
q1 = "Ảnh Chụp Xquang ngực thẳng, bệnh nhân Nam. Cho biết bệnh nhân bị gì?"
conversation.append({"role": "user", "content": [{"type": "text", "text": q1}, {"type": "image", "image": image}]})

prompt1 = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
inputs1 = processor(text=[prompt1], images=[image], return_tensors="pt").to(model.device)

# ⏱️ Measure time
start_t = time.time()
with torch.no_grad():
    output_ids1 = model.generate(**inputs1, max_new_tokens=512)
end_t = time.time()
latency1 = end_t - start_t

a1 = processor.decode(output_ids1[0][inputs1.input_ids.shape[1]:], skip_special_tokens=True).strip()
conversation.append({"role": "assistant", "content": [{"type": "text", "text": a1}]})

print("====== TURN 1 ======")
print("Q1:", q1)
print("A1:", a1)
print(f"⏱️ Thời gian infer TURN 1: {latency1:.3f} giây")


# ========= TURN 2 =========
q2 = "Kết luận thông tin trên bệnh nhân bị gì?"
conversation.append({"role": "user", "content": [{"type": "text", "text": q2}]})

prompt2 = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
inputs2 = processor(text=[prompt2], images=[image], return_tensors="pt").to(model.device)

start_t = time.time()
with torch.no_grad():
    output_ids2 = model.generate(**inputs2, max_new_tokens=512)
end_t = time.time()
latency2 = end_t - start_t

a2 = processor.decode(output_ids2[0][inputs2.input_ids.shape[1]:], skip_special_tokens=True).strip()
conversation.append({"role": "assistant", "content": [{"type": "text", "text": a2}]})
sumtime = latency1 + latency2
print(f"\n⏱️ Tổng thời gian infer 2 turn: {sumtime:.3f} giây")
print("\n====== TURN 2 ======")
print("Q2:", q2)
print("A2:", a2)
print(f"⏱️ Thời gian infer TURN 2: {latency2:.3f} giây")
