# """
# b1: mở terminal chạy CUDA_VISIBLE_DEVICES=1 vllm serve THP2903/Qwen2vl_7b_instruct_medical_multiturn_full
# b2 : chạy đoạn code này CUDA_VISIBLE_DEVICES=1 python /home/tiennv/phucth/medical/vllm/src/vllm_clara_multi.py


from openai import OpenAI
import base64
from PIL import Image
from io import BytesIO
import time   # <-- thêm import time

def encode_image(img_path):
    img = Image.open(img_path).convert("RGB").resize((448, 448))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# ---- Encode image ----
image_path = "/home/clara/phucth/code/project/Medical_CLARA/infer/demo_clara/image_test/image(4).png"
image_base64 = encode_image(image_path)

# ---- Client ----
client = OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:8006/v1")

# ---- History ----
messages = []

# Turn 1
q1 = "Ảnh chụp X-ray PA (Chụp Xquang tim phổi thẳng) bệnh nhân nam. Cho biết bệnh nhân bị gì?"
messages.append({
    "role": "user",
    "content": [
        {"type": "text", "text": q1},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
    ]
})

model_path = "/home/clara/phucth/code/project/public_dataset_med/public_data_medical/Qwen2-VL-7B-multi-137k_full_maed_v3"

# ---- Measure time for response1 ----
start_t = time.time()
response1 = client.chat.completions.create(
    model=model_path,
    messages=messages,
    temperature=1.0,
    max_tokens=1024
)
end_t = time.time()
latency1 = end_t - start_t

a1 = response1.choices[0].message.content.strip()
print("\n====== TURN 1 ======")
print("Q1:", q1)
print("A1:", a1)
print(f"⏱️ Thời gian infer TURN 1: {latency1:.3f} giây")

# Turn 2
q2 = "Kết luận từ thông tin trên bệnh nhân bị gì?"
messages.append({"role": "assistant", "content": [{"type": "text", "text": a1}]})
messages.append({"role": "user", "content": [{"type": "text", "text": q2}]})

# ---- Measure time for response2 ----
start_t = time.time()
response2 = client.chat.completions.create(
    model=model_path,
    messages=messages,
    temperature=0.7,
    max_tokens=1024
)
end_t = time.time()
latency2 = end_t - start_t

a2 = response2.choices[0].message.content.strip()
sumtime = latency1 + latency2
print(f"\n⏱️ Tổng thời gian infer 2 turn: {sumtime:.3f} giây")
print("\n====== TURN 2 ======")
print("Q2:", q2)
print("A2:", a2)
print(f"⏱️ Thời gian infer TURN 2: {latency2:.3f} giây")
