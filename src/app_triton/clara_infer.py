import gradio as gr
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
from io import BytesIO
import base64
from huggingface_hub import login
import torch
torch.cuda.empty_cache()  # giải phóng VRAM cache, không xóa tensors đang dùng

# Đăng nhập Hugging Face (nếu cần)
login(token="")

# Thiết lập thiết bị
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"  # Chạy trên CPU nếu không có GPU
# Danh sách model có thể chọn
MODEL_INFOS = {
    "Clara": {
        "model_id": "THP2903/Qwen2-VL-7B-multi-137k_full_maed",
        "processor_id": "THP2903/Qwen2vl_7b_instruct_medical_multiturn_full"
        # "model_id": "THP2903/Qwen2vl_instruct_medical_2",
        # "processor_id": "THP2903/Qwen2vl_instruct_medical_2"
    },
    "Clara-mini": {
        "model_id": "THP2903/Qwen2vl_7b_instruct_medical_multiturn_full",
        "processor_id": "THP2903/Qwen2vl_7b_instruct_medical_multiturn_full"
    }
}

# # Biến toàn cục lưu model hiện tại
# current_model_name = None
# current_model = None
# current_processor = None
# prev_question_text = None
# turn_count = 0

# # Resize ảnh để inference
# def resize_image(image: Image.Image, max_size: int = 1408) -> Image.Image:
#     if image is None:
#         return None
#     w, h = image.size
#     if max(w, h) <= max_size:
#         return image
#     scale = max_size / max(w, h)
#     new_w, new_h = int(w * scale), int(h * scale)
#     return image.resize((new_w, new_h), resample=Image.BILINEAR)

# # Tách ảnh từ lịch sử hội thoại
# def process_vision_info(chat_history):
#     images = []
#     for message in chat_history:
#         for content in message["content"]:
#             if content["type"] == "image":
#                 images.append(content["image"])
#     return images if images else None, None

# # Hàm chính chạy model Qwen
# def run_qwen_model(user_text, user_img, model_name, history, max_new_tokens=512, temperature=0.7, top_p=0.9, top_k=50):
#     global current_model, current_processor, current_model_name
#     global prev_question_text, turn_count

#     chat_history_qwen = history.get("qwen_history", [])
#     display_history = history.get("display_history", [])

#     # Load model nếu chưa có hoặc khác model cũ
#     if model_name != current_model_name:
#         if current_model is not None:
#             del current_model
#             torch.cuda.empty_cache()
#         current_model = AutoModelForVision2Seq.from_pretrained(
#             MODEL_INFOS[model_name]["model_id"],
#             torch_dtype=torch.bfloat16,
#             low_cpu_mem_usage=True,
#             device_map="auto"
#         )
#         current_processor = AutoProcessor.from_pretrained(MODEL_INFOS[model_name]["processor_id"])
#         current_model_name = model_name

#     # Đếm số lượt hội thoại
#     if prev_question_text is not None and user_text.strip() != prev_question_text.strip():
#         turn_count += 1
#     elif prev_question_text is None:
#         turn_count = 1
#     prev_question_text = user_text.strip()

#     # Gắn system message nếu là lượt đầu
#     if not chat_history_qwen:
#         chat_history_qwen.append({
#             "role": "system",
#             "content": [{"type": "text", "text": "Bạn là một trợ lý bác sĩ. Trả lời chính xác, dễ hiểu."}]
#         })

#     # Chuẩn bị input
#     user_content = [{"type": "text", "text": user_text}]
#     if user_img is not None:
#         user_img = resize_image(user_img, max_size=512)
#         user_content.append({"type": "image", "image": user_img})
#     chat_history_qwen.append({"role": "user", "content": user_content})

#     # Tạo prompt và xử lý ảnh
#     text_prompt = current_processor.apply_chat_template(chat_history_qwen, tokenize=False, add_generation_prompt=True)
#     images, _ = process_vision_info(chat_history_qwen)
#     inputs = current_processor(text=[text_prompt], images=images, return_tensors="pt", padding=True).to(device)

#     with torch.no_grad():
#         output_ids = current_model.generate(
#             **inputs,
#             max_new_tokens=int(max_new_tokens),
#             temperature=float(temperature),
#             top_p=float(top_p),
#             top_k=int(top_k),
#             eos_token_id=current_processor.tokenizer.eos_token_id
#         )

#     input_len = inputs.input_ids.shape[1]
#     generated_ids = output_ids[0][input_len:]
#     output_text = current_processor.decode(generated_ids, skip_special_tokens=True)

#     # Trả lời từ model
#     chat_history_qwen.append({
#         "role": "assistant",
#         "content": [{"type": "text", "text": output_text}]
#     })

#     # Chỉ hiển thị text (❌ KHÔNG hiển thị ảnh)
#     user_message = user_text
#     display_history.append((user_message, output_text))

#     # Cập nhật lịch sử
#     history["qwen_history"] = chat_history_qwen
#     history["display_history"] = display_history

#     return display_history, history

# # ===== Gradio Interface =====
# with gr.Blocks() as demo:
#     gr.Markdown("## 🩻 Trợ lý X-ray AI - Qwen Medical")

#     with gr.Row():
#         model_selector = gr.Dropdown(choices=list(MODEL_INFOS.keys()), value="Clara", label="Chọn model")
#         submit_btn = gr.Button("Gửi")

#     with gr.Row():
#         user_text = gr.Textbox(label="Nhập câu hỏi", placeholder="Ví dụ: Đây là ảnh X-quang tim phổi, bệnh nhân bị gì?")
#         image_input = gr.Image(type="pil", label="Upload ảnh X-ray")

#     chatbot_display = gr.Chatbot(label="Kết quả tư vấn", height=600)
#     state = gr.State({"qwen_history": [], "display_history": []})

#     submit_btn.click(
#         fn=run_qwen_model,
#         inputs=[user_text, image_input, model_selector, state],
#         outputs=[chatbot_display, state]
#     )

# demo.launch()


current_model_name = None
current_model = None
current_processor = None

def resize_image(image: Image.Image, max_size: int = 448) -> Image.Image:
    if image is None:
        return None
    w, h = image.size
    if max(w, h) <= max_size:
        return image
    scale = max_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    return image.resize((new_w, new_h), resample=Image.BILINEAR)

def process_vision_info(chat_history):
    images = []
    for message in chat_history:
        for content in message["content"]:
            if content["type"] == "image":
                images.append(content["image"])
    return images if images else None, None
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoConfig
from transformers import Qwen2VLProcessor, Qwen2VLForConditionalGeneration
# Gọi model một lượt    _ (  CODE ĐÚNG KO ĐƯỢC CHỈNH SỬA THÊM)
def run_qwen_model(user_text, user_img, model_name, history, max_new_tokens=512, temperature=0.7, top_p=1.0, top_k=30):  # 512 -> 256     0.7 0.9 50 
    global current_model, current_processor, current_model_name

    chat_history_qwen = history.get("qwen_history", []) 

    # if model_name != current_model_name:
    #     if current_model is not None:
    #         del current_model
    #         torch.cuda.empty_cache()
    #     current_model = AutoModelForVision2Seq.from_pretrained(
    #         MODEL_INFOS[model_name]["model_id"],
    #         torch_dtype=torch.bfloat16,  # torch_dtype=torch.float16

    #         low_cpu_mem_usage=True,
    #         device_map="auto"   # sequential
    #     )
    #     current_processor = AutoProcessor.from_pretrained(MODEL_INFOS[model_name]["processor_id"])
    #     current_model_name = model_name
    

    if model_name != current_model_name:
        if current_model is not None:
            del current_model
            torch.cuda.empty_cache()

        current_model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_INFOS[model_name]["model_id"],
            torch_dtype=torch.bfloat16,
            device_map="auto"
            # device_map="cpu"  
        )

        current_processor = Qwen2VLProcessor.from_pretrained(
            MODEL_INFOS[model_name]["processor_id"],
            use_fast=True
        )

        current_model_name = model_name

    # Thêm system message nếu lần đầu
    if not chat_history_qwen:
        chat_history_qwen.append({
            "role": "system",
            "content": [{"type": "text", "text": "Bạn là một trợ lý bác sĩ. Trả lời chính xác, dễ hiểu."}]
        })

    # Chuẩn bị nội dung người dùng
    user_content = [{"type": "text", "text": user_text}]
    if user_img is not None:
        user_img = resize_image(user_img, max_size=448)
        user_content.append({"type": "image", "image": user_img})
    chat_history_qwen.append({"role": "user", "content": user_content})

    text_prompt = current_processor.apply_chat_template(chat_history_qwen, tokenize=False, add_generation_prompt=True)
    images, _ = process_vision_info(chat_history_qwen)
    inputs = current_processor(text=[text_prompt], images=images, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        output_ids = current_model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            eos_token_id=current_processor.tokenizer.eos_token_id
        )

    input_len = inputs.input_ids.shape[1]
    generated_ids = output_ids[0][input_len:]
    output_text = current_processor.decode(generated_ids, skip_special_tokens=True)

    # Ghi lại phản hồi
    chat_history_qwen.append({
        "role": "assistant",
        "content": [{"type": "text", "text": output_text}]
    })

    # Cập nhật lịch sử
    history["qwen_history"] = chat_history_qwen
    return output_text, history


# def run_qwen_model(
#     user_text,
#     user_img,
#     model_name,
#     history,
#     max_new_tokens=512,
#     temperature=0.9,
#     top_p=1.0,
#     top_k=0,
#     deterministic=False  # ✅ thêm tùy chọn cố định output
# ):
#     global current_model, current_processor, current_model_name

#     chat_history_qwen = history.get("qwen_history", [])

#     # Tải model nếu cần
#     if model_name != current_model_name:
#         if current_model is not None:
#             del current_model
#             torch.cuda.empty_cache()

#         current_model = Qwen2VLForConditionalGeneration.from_pretrained(
#             MODEL_INFOS[model_name]["model_id"],
#             torch_dtype=torch.bfloat16,
#             device_map="auto"
#         )

#         current_processor = Qwen2VLProcessor.from_pretrained(
#             MODEL_INFOS[model_name]["processor_id"],
#             use_fast=True
#         )

#         current_model_name = model_name

#     # Thêm system prompt nếu lần đầu
#     if not chat_history_qwen:
#         chat_history_qwen.append({
#             "role": "system",
#             "content": [{"type": "text", "text": "Bạn là một trợ lý bác sĩ. Trả lời chính xác, dễ hiểu."}]
#         })

#     # Tạo user message
#     user_content = [{"type": "text", "text": user_text}]
#     if user_img is not None:
#         user_img = resize_image(user_img, max_size=448)
#         user_content.append({"type": "image", "image": user_img})

#     chat_history_qwen.append({"role": "user", "content": user_content})

#     # Tạo prompt đầu vào
#     text_prompt = current_processor.apply_chat_template(
#         chat_history_qwen,
#         tokenize=False,
#         add_generation_prompt=True
#     )
#     images, _ = process_vision_info(chat_history_qwen)

#     inputs = current_processor(
#         text=[text_prompt],
#         images=images,
#         return_tensors="pt",
#         padding=True
#     ).to(device)

#     # ✅ Cố định seed nếu yêu cầu output ổn định
#     if deterministic:
#         torch.manual_seed(42)

#     # ✅ Tùy chỉnh decoding strategy
#     gen_kwargs = {
#         "max_new_tokens": int(max_new_tokens),
#         "eos_token_id": current_processor.tokenizer.eos_token_id,
#     }

#     if deterministic:
#         gen_kwargs.update({
#             "do_sample": False,  # dùng greedy decoding để cố định output
#         })
#     else:
#         gen_kwargs.update({
#             "do_sample": True,
#             "temperature": float(temperature),
#             "top_p": float(top_p),
#             "top_k": int(top_k),
#         })

#     # Generate
#     with torch.no_grad():
#         output_ids = current_model.generate(**inputs, **gen_kwargs)

#     input_len = inputs.input_ids.shape[1]
#     generated_ids = output_ids[0][input_len:]
#     output_text = current_processor.decode(generated_ids, skip_special_tokens=True)

#     # Lưu kết quả vào history
#     chat_history_qwen.append({
#         "role": "assistant",
#         "content": [{"type": "text", "text": output_text}]
#     })

#     history["qwen_history"] = chat_history_qwen
#     return output_text, history


def to_bullet_list(text):
    lines = text.strip().split("\n")
    items = "\n".join(f"• {line.lstrip('-').strip()}" for line in lines if line.strip())
    return items

def multiturn_infer(image, model_name, state_dict, sex, age, view):
    # ✅ Sửa lỗi: nếu reset thì state_dict có thể là list → ép về dict
    if not isinstance(state_dict, dict):
        state_dict = {}

    # Tạo prompt động
    view_full = f"X-ray {view}" if view else "X-ray"
    sex_text = f"bệnh nhân {sex.lower()}" if sex else "bệnh nhân"
    age_text = f", {age} tuổi" if age else ""

    # Prompt 1: Findings
    prompt_findings = f"Ảnh chụp {view_full} {sex_text}{age_text}. Cho biết bệnh nhân bị gì?"
    prompt_impression = "Kết luận từ thông tin trên bệnh nhân bị gì?"

    # Xử lý ảnh
    last_image = state_dict.get("last_image", None)
    if image is None:
        if last_image is None:
            warning = "⚠️ Bạn cần upload ảnh X-quang ở lần đầu."
            return [(warning, "")], state_dict, None
        else:
            image_to_use = last_image
    else:
        image_to_use = image
        state_dict["last_image"] = image

    findings_response, state_dict = run_qwen_model(prompt_findings, image_to_use, model_name, state_dict)
    impression_response, state_dict = run_qwen_model(prompt_impression, image_to_use, model_name, state_dict)
    final_response = f"""🖼️ **Hình ảnh cho thấy:**\n{to_bullet_list(findings_response)}\n\n🔍 **Chẩn đoán:**\n{to_bullet_list(impression_response)}"""

    return [(prompt_findings, final_response)], state_dict, None

import base64
import numpy as np
from io import BytesIO
from PIL import Image
from tritonclient.http import InferenceServerClient, InferInput

# =========================================================
# Helper: Encode image to base64
# =========================================================
def encode_image_to_b64(image_pil):
    buffered = BytesIO()
    image_pil.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")


# =========================================================
# Helper: Triton inference
# =========================================================
def triton_infer(model_name, text_list, image_b64_list=None, url="localhost:8000"):
    """
    Robust Triton infer helper:
    - Gọi client
    - Ghi log outputs trả về để debug tên output
    - Thử đọc `text_output` trước vì code test của bạn dùng 'text_output'
    - Nếu không có, in ra response.get_response().outputs để bạn biết tên đúng
    """
    client = InferenceServerClient(url=url)

    # TEXT INPUT
    text_arr = np.array([t.encode("utf-8") for t in text_list], dtype=object)
    text_input = InferInput("text_input", text_arr.shape, "BYTES")
    text_input.set_data_from_numpy(text_arr)

    # IMAGE INPUT
    if image_b64_list is None:
        image_b64_list = [""] * len(text_list)
    image_arr = np.array([i.encode("utf-8") for i in image_b64_list], dtype=object)
    image_input = InferInput("image_input", image_arr.shape, "BYTES")
    image_input.set_data_from_numpy(image_arr)

    # SEND
    response = client.infer(model_name=model_name, inputs=[text_input, image_input])

    # --- DEBUG: in ra cấu trúc outputs trả về từ Triton ---
    try:
        raw_resp = response.get_response()
        print(">>> Triton response outputs:", raw_resp.outputs)
    except Exception:
        # some client versions may not expose get_response the same way
        pass

    # THỬ đọc theo tên mà bạn đã dùng trong script test: "text_output"
    outputs = response.as_numpy("text_output")
    if outputs is None:
        # Thử một vài tên phổ biến (fallback)
        for candidate in ["output_text", "OUTPUT_TEXT", "output_0", "OUTPUT", "response"]:
            try:
                outputs = response.as_numpy(candidate)
                if outputs is not None:
                    print(f">>> Using output name '{candidate}'")
                    break
            except Exception:
                outputs = None

    if outputs is None:
        # Nếu vẫn None: in debug thêm và raise
        print("❌ Triton returned None for all tried output names. Inspect server logs & model config.pbtxt.")
        # in raw response bytes (if any)
        try:
            print("Raw response:", response.get_response())
        except Exception:
            pass
        raise RuntimeError("Triton returned None output — check model output name or server error")

    # Decode outputs (bytes -> str)
    decoded = []
    for o in outputs:
        if isinstance(o, (bytes, bytearray)):
            decoded.append(o.decode("utf-8"))
        else:
            decoded.append(str(o))

    return decoded




# =========================================================
# MAIN FUNCTION: MULTI-TURN INFERENCE USING TRITON
# =========================================================
def multiturn_infer_triton(image, state_dict, sex, age, view):
    # Handle None state_dict
    if state_dict is None:
        state_dict = {}

    # USER INPUT FORMAT
    prompt_findings = f"Hình X-ray {view} của bệnh nhân {sex}, {age} tuổi. Cho biết các bất thường trong ảnh."
    prompt_impression = "Kết luận từ thông tin trên bệnh nhân bị gì?"

    # ------------- TURN 1: FINDINGS (send image) -------------
    image_b64 = encode_image_to_b64(image)
    out1 = triton_infer(
        model_name="vlm",
        text_list=[prompt_findings],
        image_b64_list=[image_b64]
    )
    findings = out1[0]

    # ------------- TURN 2: IMPRESSION (no image) -------------
    out2 = triton_infer(
        model_name="vlm",
        text_list=[prompt_impression],
        image_b64_list=[""]  # no image for turn 2
    )
    impression = out2[0]

    # COMBINE FINAL
    final_response = (
        f"**Findings:** {findings}\n\n"
        f"**Impression:** {impression}"
    )

    # FORMAT CHAT HISTORY FOR GRADIO
    chat_history = [
        (prompt_findings, findings),
        (prompt_impression, impression),
    ]

    return chat_history, state_dict

"""
hàm chuẩn về singleturn_infer
"""

def singleturn_infer(image, model_name, state_dict, sex, age, view):
    # Nếu reset -> ép về dict
    if not isinstance(state_dict, dict):
        state_dict = {}

    # Tạo prompt duy nhất
    view_full = f"X-ray {view}" if view else "X-ray"
    sex_text = f"bệnh nhân {sex.lower()}" if sex else "bệnh nhân"
    age_text = f", {age} tuổi" if age else ""

    prompt_findings = f"Ảnh chụp {view_full} {sex_text}{age_text}. Cho biết bệnh nhân bị gì?"
    # prompt_findings = f"This is a {view_full} X-ray of a {sex_text}{age_text} patient. What is the diagnosis?"

    # Dùng ảnh mới hoặc lấy lại ảnh cũ
    last_image = state_dict.get("last_image", None)
    if image is None:
        if last_image is None:
            warning = "⚠️ Bạn cần upload ảnh X-quang ở lần đầu."
            return [(warning, "")], state_dict, None
        else:
            image_to_use = last_image
    else:
        image_to_use = image
        state_dict["last_image"] = image

    # Chỉ chạy 1 lượt
    response_text, state_dict = run_qwen_model(prompt_findings, image_to_use, model_name, state_dict)
    final_response = f"""🔍 **Kết quả phân tích:**\n{to_bullet_list(response_text)}"""

    return [(prompt_findings, final_response)], state_dict, None


"""
hàm test singleturn_infer
"""
# def singleturn_infer(image, model_name, state_dict, sex, age, view,
#                      max_tokens=512, temperature=0.9, top_p=1, top_k=0):
#     # Nếu reset -> ép về dict
#     if not isinstance(state_dict, dict):
#         state_dict = {}

#     # Tạo prompt duy nhất
#     view_full = f"X-ray {view}" if view else "X-ray"
#     sex_text = f"bệnh nhân {sex.lower()}" if sex else "bệnh nhân"
#     age_text = f", {age} tuổi" if age else ""

#     prompt_findings = f"Ảnh chụp {view_full} {sex_text}{age_text}. Cho biết bệnh nhân bị gì?"

#     # Dùng ảnh mới hoặc lấy lại ảnh cũ
#     last_image = state_dict.get("last_image", None)
#     if image is None:
#         if last_image is None:
#             warning = "⚠️ Bạn cần upload ảnh X-quang ở lần đầu."
#             return [(warning, "")], state_dict, None
#         else:
#             image_to_use = last_image
#     else:
#         image_to_use = image
#         state_dict["last_image"] = image

#     # Chạy mô hình Qwen
#     response_text, state_dict = run_qwen_model(
#         user_text=prompt_findings,
#         user_img=image_to_use,
#         model_name=model_name,
#         history=state_dict,
#         max_new_tokens=max_tokens,
#         temperature=temperature,
#         top_p=top_p,
#         top_k=top_k,
#         deterministic=True
#     )

#     final_response = f"""🔍 **Kết quả phân tích:**\n{to_bullet_list(response_text)}"""
#     return [(prompt_findings, final_response)], state_dict, None


# ---------------------------------------------------------
# TEST HERE
# ---------------------------------------------------------
if __name__ == "__main__":
    # 1. Load local image (đổi path ảnh của bạn ở đây)
    # image_path = "/home/truongnn/phucth/image_test/image_bt.png"
    image_path = "/home/truongnn/phucth/image_test/image(4).png"

    image = Image.open(image_path).convert("RGB")

    # 2. Call multi-turn inference
    chat, _ = multiturn_infer_triton(
        image=image,
        state_dict={},
        sex="nam",
        age=60,
        view="PA"
    )

    # 3. Print result
    print("\n===== MULTITURN OUTPUT =====")
    for i, (user, bot) in enumerate(chat, 1):
        print(f"\n--- TURN {i} ---")
        print("USER:", user)
        print("MODEL:", bot)