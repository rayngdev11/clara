"""
code hoàn chỉnh
"""

# import base64
# import json
# import requests

# with open("/home/tiennv/phucth/medical/data_test/data/testcase_medical/image_test/test4.png", "rb") as f:
#     image_b64 = base64.b64encode(f.read()).decode("utf-8")

# payload = {
#     "prompt": "USER: <image>\nẢnh chụp X-ray PA (Chụp Xquang tim phổi thẳng) bệnh nhân nam, 84 tuổi. Cho biết bệnh nhân bị gì?.\nASSISTANT:",
#     "max_tokens": 256,
#     "temperature": 0.1,
#     "multi_modal_data": json.dumps({"image": image_b64})
# }

# res = requests.post(
#     "http://localhost:8000/v2/models/THP2903clara_multiturn/generate",
#     headers={"Content-Type": "application/json"},
#     data=json.dumps(payload)
# )

# print(res.json())
import base64
import json
import requests
from transformers import AutoProcessor
from PIL import Image

# Load processor để dùng apply_chat_template
model_name = "/home/tiennv/phucth/medical/model/clara_multiturn"
processor = AutoProcessor.from_pretrained(model_name)

# Load image và encode base64
image_path = "/home/tiennv/phucth/medical/data_test/data/testcase_medical/image_test/test4.png"
with open(image_path, "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

image_pil = Image.open(image_path).convert("RGB").resize((448, 448))

# Build conversation like you did with transformers
conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_pil},
            {"type": "text", "text": "Ảnh chụp X-ray PA (Chụp Xquang tim phổi thẳng) bệnh nhân nam, 84 tuổi. Cho biết bệnh nhân bị gì?"}
        ]
    }
]

# Convert to prompt
prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

# Payload to Triton
payload = {
    "prompt": prompt,
    "max_tokens": 512,
    "temperature": 0.1,
    "multi_modal_data": json.dumps({"image": image_b64})
}

res = requests.post(
    "http://localhost:8000/v2/models/THP2903clara_multiturn/generate",  # lưu ý tên model phải khớp
    headers={"Content-Type": "application/json"},
    data=json.dumps(payload)
)

print(res.json())





"""
test và check prompt

"""
# from transformers import AutoProcessor
# from PIL import Image

# model_name = "/home/tiennv/phucth/medical/model/clara_multiturn"
# processor = AutoProcessor.from_pretrained(model_name)

# image = Image.open("/home/tiennv/phucth/medical/data_test/data/testcase_medical/image_test/test4.png").convert("RGB").resize((448, 448))

# conversation = [
#     {"role": "user", "content": [{"type": "text", "text": "Ảnh chụp X-ray PA bệnh nhân nam, 84 tuổi. Cho biết bệnh nhân bị gì?"}, {"type": "image", "image": image}]}
# ]

# prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
# print("Prompt:", prompt)





# import base64
# import json
# import requests

# # === Bước 1: Load ảnh và encode base64 ===
# with open("/home/tiennv/phucth/medical/data_test/data/testcase_medical/image_test/test4.png", "rb") as f:
#     image_b64 = base64.b64encode(f.read()).decode("utf-8")

# # === Bước 2: Tạo biến lưu hội thoại ===
# chat_history = []

# # === Bước 3: Hàm gửi câu hỏi ===
# def ask(question, image_b64=None):
#     global chat_history

#     # Nếu là câu đầu tiên → thêm image
#     if len(chat_history) == 0:
#         chat_history.append(f"USER: <image>\n{question}\nASSISTANT:")
#     else:
#         chat_history.append(f"USER: {question}\nASSISTANT:")

#     # Ghép prompt từ toàn bộ lịch sử
#     full_prompt = "\n".join(chat_history)

#     payload = {
#         "prompt": full_prompt,
#         "max_tokens": 256,
#         "temperature": 0.1,
#     }

#     if image_b64 is not None and len(chat_history) == 1:
#         payload["multi_modal_data"] = json.dumps({"image": image_b64})

#     # Gửi request
#     res = requests.post(
#         "http://localhost:8000/v2/models/THP2903Qwen2vl_instruct_medical_2/generate",
#         headers={"Content-Type": "application/json"},
#         data=json.dumps(payload)
#     )

#     # Check lỗi HTTP
#     if res.status_code != 200:
#         print("❌ HTTP Error:", res.status_code)
#         print("📄 Response:", res.text)
#         return "Error"

#     try:
#         res_json = res.json()
#         response = res_json.get("text", "").strip()
#     except Exception as e:
#         print("❌ JSON decode error:", e)
#         print("📄 Raw response:", res.text)
#         return "Error"

#     chat_history.append(response)
#     return response


# # === Bước 4: Gọi lần lượt 2 câu hỏi ===
# q1 = "Ảnh chụp X-ray PA (Chụp Xquang tim phổi thẳng) bệnh nhân nam, 84 tuổi. Cho biết bệnh nhân bị gì?"
# a1 = ask(q1, image_b64=image_b64)

# q2 = "Có dấu hiệu tổn thương lan tỏa ở phổi không?"
# a2 = ask(q2)

# # === Bước 5: In toàn bộ hội thoại ===
# print("Q1:", q1)
# print("A1:", a1)
# print("Q2:", q2)
# print("A2:", a2)
