import base64
from io import BytesIO
from PIL import Image
import numpy as np
from tritonclient.http import InferenceServerClient, InferInput
import time
import torch
torch.cuda.empty_cache()

# ===============================
# Encode ảnh sang base64
# ===============================
def encode_image_to_b64(image_pil):
    buffered = BytesIO()
    image_pil.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# ===============================
# Gọi Triton + đo thời gian
# ===============================
def triton_infer(model_name, prompt_texts, image_b64s=None, url="localhost:8000"):
    client = InferenceServerClient(url=url)

    # Text input
    text_arr = np.array([t.encode("utf-8") for t in prompt_texts], dtype=object)
    text_input = InferInput("text_input", text_arr.shape, "BYTES")
    text_input.set_data_from_numpy(text_arr)

    # Image input
    if image_b64s is None:
        image_b64s = [""] * len(prompt_texts)
    image_arr = np.array([i.encode("utf-8") for i in image_b64s], dtype=object)
    image_input = InferInput("image_input", image_arr.shape, "BYTES")
    image_input.set_data_from_numpy(image_arr)

    # ⏱️ START
    start = time.time()

    response = client.infer(model_name=model_name, inputs=[text_input, image_input])

    # ⏱️ END
    end = time.time()
    latency = end - start

    # Lấy output
    outputs = None
    for candidate in ["text_output", "output_text", "OUTPUT_TEXT"]:
        try:
            outputs = response.as_numpy(candidate)
            if outputs is not None:
                break
        except Exception:
            continue

    if outputs is None:
        raise RuntimeError("Triton returned None output.")

    decoded = [o.decode("utf-8") if isinstance(o, (bytes, bytearray)) else str(o) for o in outputs]

    return decoded, latency

# ===============================
# Multi-turn inference
# ===============================
def multiturn_infer_triton(image, sex, age, view, model_name="vlm", state_dict=None):
    if state_dict is None:
        state_dict = {}

    if image is not None:
        state_dict["last_image"] = image
    elif "last_image" not in state_dict:
        raise ValueError("Bạn cần upload ảnh X-quang lần đầu.")

    image_to_use = state_dict["last_image"]
    history_text = state_dict.get("history", "")

    # ---------- TURN 1 ----------
    prompt_findings = f"Hình X-ray {view} của bệnh nhân {sex}, {age} tuổi. Cho biết các bất thường trong ảnh."
    prompt1 = (history_text + "\n" + prompt_findings).strip()
    image_b64 = encode_image_to_b64(image_to_use)

    out1, lat1 = triton_infer(model_name=model_name, prompt_texts=[prompt1], image_b64s=[image_b64])
    findings = out1[0]

    history_text += f"\nUSER: {prompt_findings}\nMODEL: {findings}"

    # ---------- TURN 2 ----------
    prompt_impression = "Kết luận từ thông tin trên bệnh nhân bị gì?"
    prompt2 = (history_text + "\n" + prompt_impression).strip()

    out2, lat2 = triton_infer(model_name=model_name, prompt_texts=[prompt2], image_b64s=[""])
    impression = out2[0]

    history_text += f"\nUSER: {prompt_impression}\nMODEL: {impression}"
    state_dict["history"] = history_text

    # Gói output
    chat_history = [
        (prompt_findings + f"\n⏱️ Latency TURN 1: {lat1:.3f}s", findings),
        (prompt_impression + f"\n⏱️ Latency TURN 2: {lat2:.3f}s", impression)
    ]
    sumtime = lat1 + lat2
    chat_history.append((f"Tổng thời gian xử lý: {sumtime:.3f}s", ""))


    return chat_history, state_dict

# ===============================
# Test
# ===============================
if __name__ == "__main__":
    image_path = "/home/truongnn/phucth/image_test/image(2).png"
    # image_path = "/home/truongnn/phucth/image_test/test4.png"
    # image_path = "/home/truongnn/phucth/image_test/image_bt.png"
    image = Image.open(image_path).convert("RGB")
    state = {}
    chat, state = multiturn_infer_triton(image=image, sex="nam", age=48, view="PA")

    for i, (user, bot) in enumerate(chat, 1):
        print(f"\n--- TURN {i} ---")
        print("USER:", user)
        print("MODEL:", bot)
