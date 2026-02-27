import gradio as gr
from clara_infer import multiturn_infer as clara_infer
# from clara_infer import singleturn_infer as clara_infer
# from clara_lora import multiturn_infer as clara_infer
# from gemini_api import gemini_multiturn_infer as gemini_infer
# from gpt_api import gpt_multiturn_infer as gpt_infer
import gradio as gr
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
from io import BytesIO
import base64
from huggingface_hub import login
from check_xray import is_chest_xray
# ==== Khởi tạo biến global ====



# Đăng nhập Hugging Face (nếu cần)
login(token="")

# Thiết lập thiết bị
device = "cuda" if torch.cuda.is_available() else "cpu"

# Danh sách model có thể chọn
MODEL_INFOS = {
    "Clara": {
        "model_id": "THP2903/Qwen2-VL-7B-multi-137k_full_maed_v3",
        "processor_id": "THP2903/Qwen2-VL-7B-multi-137k_full_maed_v3"
        # "model_id": "THP2903/Qwen2vl_instruct_medical_2",
        # "processor_id": "THP2903/Qwen2vl_instruct_medical_2"
    },
    "Clara-mini": {
        "model_id": "THP2903/Qwen2-VL-7B-multi-137k_full_maed_v3",
        "processor_id": "THP2903/Qwen2-VL-7B-multi-137k_full_maed_v3"
    }
}

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
from io import BytesIO
import base64
from huggingface_hub import login
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch

app = FastAPI(title="Qwen2VL Medical API")
# Đăng nhập Hugging Face (nếu cần)
login(token="")


# Thiết lập thiết bị
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"  # Chạy trên CPU nếu không có GPU
# Danh sách model có thể chọn
# MODEL_INFOS = {
#     "Clara": {
#         "model_id": "THP2903/Qwen2-VL-7B-multi-137k_full_maed",
#         "processor_id": "Qwen/Qwen2-VL-7B-Instruct"
#         # "model_id": "THP2903/Qwen2vl_instruct_medical_2",
#         # "processor_id": "THP2903/Qwen2vl_instruct_medical_2"
#     },
#     "Clara-mini": {
#         "model_id": "THP2903/Qwen2vl_7b_instruct_medical_multiturn_full",
#         "processor_id": "THP2903/Qwen2vl_7b_instruct_medical_multiturn_full"
#     }
# }
from huggingface_hub import hf_hub_download
MAE_CKPT = hf_hub_download(
    repo_id="THPBi/mae_med",
    filename="loss=0.02.ckpt",
    token="",
    subfolder="files/output_ptln/sample-epoch=060-valid"
)
# MODEL_INFOS = {
#     "Clara": {
#         "model_id": "THP2903/Qwen2-VL-7B-multi-137k_full_maed_v3",
#         "processor_id": "Qwen/Qwen2-VL-7B-Instruct"
#     },
#     "Clara-custom": {
#         "base_model": "Qwen/Qwen2-VL-7B-Instruct",
#         "lora_path": "THP2903/Clara-7B-multi-137k-mae",
#         "processor_id": "Qwen/Qwen2-VL-7B-Instruct",
#         "mae_ckpt": MAE_CKPT,
#     }
# }

# ====== Hàm kiểm tra ảnh X-quang trước khi gọi model ======
# def safe_clara_infer(image, model_selector, state, sex, age, view):
#     if image is None:
#         return [[None, "❌ Bạn cần upload ảnh X-quang."]], state, None
#     if not is_chest_xray(image):
#         return [[None, "❌ Đây không phải ảnh X-quang ngực."]], state, None
#     return clara_infer(image, model_selector, state, sex, age, view)
# def safe_clara_infer(image, model_selector, state, sex, age, view):
#     if image is None:
#         return [[None, "❌ Bạn cần upload ảnh X-quang."]], state

#     if not is_chest_xray(image):
#         return [[None, "❌ Đây không phải ảnh X-quang ngực."]], state

#     # Đảm bảo có "llm_history" trong state
#     if "llm_history" not in state:
#         state["llm_history"] = []

#     # Ghi log vào LLM history
#     state["llm_history"].append({"role": "user", "content": "Phân tích ảnh X-quang"})

#     # Gọi model chính
#     response, state, _ = clara_infer(image, model_selector, state, sex, age, view)

#     # Ghi lại kết quả vào LLM history
#     result_text = response[0][1] if response else "Không có kết quả"
#     state["llm_history"].append({"role": "assistant", "content": result_text})

#     return response, state

# def safe_clara_infer(image, model_selector, state, sex, age, view):
#     if image is None:
#         return [[None, "❌ Bạn cần upload ảnh X-quang."]], state

#     if not is_chest_xray(image):
#         return [[None, "❌ Đây không phải ảnh X-quang ngực."]], state

#     # ✅ Ép lại state về dict nếu bị truyền sai kiểu
#     if not isinstance(state, dict):
#         state = {"qwen_history": [], "last_image": None, "llm_history": []}
#     elif "llm_history" not in state:
#         state["llm_history"] = []

#     # 🧠 Log yêu cầu phân tích ảnh vào lịch sử chat
#     state["llm_history"].append({"role": "user", "content": "Phân tích ảnh X-quang"})

#     # 🩻 Gọi model chính Clara
#     response, state, _ = clara_infer(image, model_selector, state, sex, age, view)

#     # 💬 Ghi lại phản hồi từ mô hình vào lịch sử chat
#     result_text = response[0][1] if response else "Không có kết quả"
#     state["llm_history"].append({"role": "assistant", "content": result_text})

#     return response, state

def safe_clara_infer(image, model_selector, state, sex, age, view):
    if not isinstance(state, dict):
        state = {"qwen_history": [], "last_image": None, "llm_history": []}

    if image is None:
        return [[None, "❌ Bạn cần upload ảnh X-quang."]], state

    if not is_chest_xray(image):
        return [[None, "❌ Đây không phải ảnh X-quang ngực."]], state

    # Ghi log vào LLM history
    state["llm_history"].append({"role": "user", "content": "Phân tích ảnh X-quang"})

    # Gọi model chính
    response, state, _ = clara_infer(image, model_selector, state, sex, age, view)

    # Ghi lại kết quả vào LLM history
    result_text = response[0][1] if response else "Không có kết quả"
    state["llm_history"].append({"role": "assistant", "content": result_text})

    return response, state



# def safe_gemini_infer(image, sex, age, view, state):
#     if image is None:
#         return [[None, "❌ Bạn cần upload ảnh X-quang."]], state, None
#     if not is_chest_xray(image):
#         return [[None, "❌ Đây không phải ảnh X-quang ngực."]], state, None
#     return gemini_infer(image, sex, age, view, state)

# def safe_gpt_infer(image, sex, age, view, state):
#     if image is None:
#         return [[None, "❌ Bạn cần upload ảnh X-quang."]], state, None
#     if not is_chest_xray(image):
#         return [[None, "❌ Đây không phải ảnh X-quang ngực."]], state, None
#     return gpt_infer(image, sex, age, view, state)





# ===== Gradio UI =====
# with gr.Blocks() as demo:
#     gr.Markdown("## Trợ lý Clara - Phân tích ảnh X-quang ngực")

#     # INPUT CHUNG
#     with gr.Row():
#         sex_input = gr.Dropdown(choices=["Nam", "Nữ"], label="Giới tính")
#         age_input = gr.Textbox(label="Tuổi", placeholder="VD: 45")
#         view_input = gr.Dropdown(choices=["PA", "Lateral", "AP"], label="Góc chụp", value="PA")
#     image_input = gr.Image(type="pil", label="Upload ảnh X-quang")
    
#     # State lưu ảnh và lịch sử model
#     shared_image = gr.State(None)         # lưu ảnh dùng chung cho 2 tab
#     state_clara = gr.State({"qwen_history": [], "last_image": None})
#     state_gemini = gr.State([])
#     state_gpt = gr.State([])

#     # Khi người dùng upload ảnh → cập nhật shared_image
#     def update_shared_image(image):
#         return image

#     image_input.change(fn=update_shared_image, inputs=image_input, outputs=shared_image)
    
#     # TABS
#         # === CỘT TRÁI: INPUT ===
        
#     with gr.Tabs():
#         with gr.Tab("Clara (Qwen)"):
#             clara_model_selector = gr.Dropdown(choices=["Clara", "Clara-mini"], value="Clara", label="Chọn model Clara")
#             submit_c = gr.Button("Phân tích với Clara")
#             chatbot_c = gr.Chatbot(label="Kết quả từ Clara", height=500)

#             # submit_c.click(
#             #     fn=clara_infer,
#             #     inputs=[shared_image, clara_model_selector, state_clara, sex_input, age_input, view_input],
#             #     outputs=[chatbot_c, state_clara, image_input]  # vẫn reset image nếu bạn muốn, có thể bỏ nếu không
#             # )
#             submit_c.click(
#                 fn=safe_clara_infer,
#                 inputs=[shared_image, clara_model_selector, state_clara, sex_input, age_input, view_input],
#                 outputs=[chatbot_c, state_clara, image_input]
#             )

#         with gr.Tab("Gemini"):
#             submit_g = gr.Button("Phân tích với Gemini")
#             output_g = gr.Textbox(label="Kết quả từ Gemini", lines=10)

#             # submit_g.click(
#             #     fn=gemini_infer,
#             #     inputs=[shared_image, sex_input, age_input, view_input, state_gemini],
#             #     outputs=[output_g, state_gemini]
#             # )
#             submit_g.click(
#                 fn=safe_gemini_infer,
#                 inputs=[shared_image, sex_input, age_input, view_input, state_gemini],
#                 outputs=[output_g, state_gemini]
#             )

            
#         with gr.Tab("GPT-4o"):
#             submit_g = gr.Button("Phân tích với GPT-4o")
#             output_g = gr.Textbox(label="Kết quả từ GPT-4o", lines=10)

#             # submit_g.click(
#             #     fn=gpt_infer,
#             #     inputs=[shared_image, sex_input, age_input, view_input, state_gpt],
#             #     outputs=[output_g, state_gpt]
#             # )
            
#             submit_g.click(
#                 fn=safe_gpt_infer,
#                 inputs=[shared_image, sex_input, age_input, view_input, state_gpt],
#                 outputs=[output_g, state_gpt]
#             )

#         # NÚT RESET TOÀN BỘ
#     reset_btn = gr.Button("🗑️ Reset tất cả", variant="stop")

#     def reset_all():
#         return (
#             None,  # image_input
#             None,  # sex_input
#             "",    # age_input
#             "PA",  # view_input
#             [],    # state_clara
#             [],    # state_gemini
#             [],    # state_gpt
#             None,  # clara chatbot
#             "",    # gemini output
#             ""     # gpt output
#         )

#     reset_btn.click(
#         fn=reset_all,
#         inputs=[],
#         outputs=[
#             image_input,
#             sex_input,
#             age_input,
#             view_input,
#             state_clara,
#             state_gemini,
#             state_gpt,
#             chatbot_c,
#             output_g,
#             output_g  # GPT và Gemini dùng chung output_g, nếu khác thì sửa lại
#         ]
#     )
# Load model/tokenizer 1 lần


# ****************************************************************

# def clean_qwen_response(text: str) -> str:
#     # Cắt phần nằm giữa <|im_start|>assistant ... <|im_end|>
#     if "<|im_start|>assistant" in text:
#         text = text.split("<|im_start|>assistant")[-1]
#     if "<|im_end|>" in text:
#         text = text.split("<|im_end|>")[0]
#     return text.strip()
# # def truncate_response(text, max_lines=5):
# #     lines = text.split("\n")
# #     truncated = "\n".join(lines[:max_lines])
# #     if len(lines) > max_lines:
# #         truncated += "\n👉 [Câu trả lời dài – xem thêm bên dưới]"
# #     return truncated


# SYSTEM_PROMPT = 'You are a language model that answers medical questions using the provided context. Use step-by-step reasoning with **thought**, **action**, and **observation**. At the end, summarize your reasoning inside <think>...</think> and provide the final answer in Vietnamese. Always reply entirely in Vietnamese using proper medical terms.'
# # device_map = "cuda" if torch.cuda.is_available() else "cpu"
# from transformers import AutoModelForCausalLM, AutoTokenizer
# llm_model = AutoModelForCausalLM.from_pretrained(
#     "ChaosAiVision/DeepSeek-R1-0528-Qwen3-8B-vi-sft-medical-9k",
#     torch_dtype="auto",
#     device_map="auto"
# )
# llm_tokenizer = AutoTokenizer.from_pretrained('ChaosAiVision/DeepSeek-R1-0528-Qwen3-8B-vi-sft-medical-9k') # Qwen/Qwen3-1.7B
# def llm_chat_infer(user_input, history):
#     history = history or []
#     chat_list = [{"role": "system", "content": SYSTEM_PROMPT}] + history
#     chat_list.append({"role": "user", "content": user_input})

#     # Build prompt
#     text_prompt = llm_tokenizer.apply_chat_template(
#         chat_list,
#         tokenize=False,
#         add_generation_prompt=True,
#         # enable_thinking=True
#     )

#     inputs = llm_tokenizer(text_prompt, return_tensors="pt").to("cuda")
#     outputs = llm_model.generate(
#         input_ids=inputs.input_ids,
#         attention_mask=inputs.attention_mask,
#         max_new_tokens=1024,
#         temperature=0.7,
#         top_p=0.9,
#         do_sample=True
#     )
#     response_text = llm_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
#     response_text = clean_qwen_response(response_text)

#     history.append({"role": "user", "content": user_input})
#     history.append({"role": "assistant", "content": response_text})
#     # print(">>> Prompt:", text_prompt)
#     # print(">>> LLM response:", response_text)

#     return response_text, history
# ****************************************************************

# def llm_gr_chat(user_input, state):
#     if not isinstance(state, dict):
#         state = {"qwen_history": [], "last_image": None, "llm_history": []}

#     history = state.get("llm_history", [])
#     response, updated_history = llm_chat_infer(user_input, history)
#     state["llm_history"] = updated_history

#     # Chuyển thành dạng (user, assistant) để Chatbot hiển thị
#     chat_ui = []
#     for i in range(1, len(updated_history), 2):  # từng cặp user - assistant
#         user_msg = updated_history[i - 1]["content"]
#         assistant_msg = updated_history[i]["content"]
#         chat_ui.append((user_msg, assistant_msg))

#     return chat_ui, state

# def llm_gr_chat(user_input, state):
#     if not isinstance(state, dict):
#         state = {"qwen_history": [], "last_image": None, "llm_history": []}

#     history = state.get("llm_history", [])
#     response, updated_history = llm_chat_infer(user_input, history)
#     state["llm_history"] = updated_history

#     # Lấy toàn bộ hội thoại đang có
#     chat_ui = []
#     for i in range(1, len(updated_history), 2):
#         u = updated_history[i - 1].get("content", "")
#         a = updated_history[i].get("content", "")
#         chat_ui.append((u, a))

#     # ✅ Cẩn thận: nếu không có câu trả lời thì vẫn nên trả ít nhất 1 dòng
    
#     chat_ui = [(str(u), str(a)) for u, a in chat_ui]
#     if not chat_ui:
#         chat_ui = [(user_input, "⚠️ LLM không trả lời được.")]
#     print(">>> Chat UI:", chat_ui)
#     print(">>> Type:", type(chat_ui), "Length:", len(chat_ui))
#     print(">>> Sample:", chat_ui[:1])


#     return gr.update(value=chat_ui, visible=True), state
    # return chat_ui, state


    # ****************************************************************

# def llm_gr_chat(user_msg, state):
#     if not isinstance(state, dict):
#         state = {"qwen_history": [], "last_image": None, "llm_history": []}

#     history = state["llm_history"]

#     # Gọi model thật để trả lời
#     reply, updated_history = llm_chat_infer(user_msg, history)

#     state["llm_history"] = updated_history

#     # Chuyển OpenAI-style history → Gradio-compatible
#     chat_ui = []
#     for i in range(1, len(updated_history), 2):
#         u = updated_history[i - 1].get("content", "")
#         a = updated_history[i].get("content", "")
#         chat_ui.append([u, a])

#     if not chat_ui:
#         chat_ui = [[user_msg, "⚠️ LLM không trả lời được."]]

#     return gr.update(value=chat_ui, visible=True), state
# state = gr.State({"llm_history": [], "full_history_ui": ""})
# import re

# def strip_think_tags(text):
#     # Xóa đoạn nằm trong <think>...</think>
#     return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

# def llm_gr_chat_1(user_msg, state):
#     history = state["llm_history"]

#     full_reply, updated_history = llm_chat_infer(user_msg, history)
#     # updated_history.append({"role": "user", "content": user_msg})
#     # updated_history.append({"role": "assistant", "content": full_reply})
#     state["llm_history"] = updated_history

#     # short_reply = truncate_response(full_reply, max_lines=4)
#     full_cleaned = strip_think_tags(full_reply)
    
#     # Tạo UI rút gọn cho Chatbot
#     chat_ui = []
#     for i in range(1, len(updated_history), 2):
#         u = updated_history[i-1]["content"]
#         a = updated_history[i]["content"]
#         # a_short = truncate_response(a, max_lines=4)
#         chat_ui.append([u, a])

#     # Gộp full để xem lại
#     full_history_ui = ""
#     for i in range(1, len(updated_history), 2):
#         q = updated_history[i-1]["content"]
#         a = updated_history[i]["content"]
#         full_history_ui += f"👨‍💻 User: {q}\n\n 🤖 Clara: {a}\n\n ----------------------------------------\n\n"

#     state["full_history_ui"] = full_history_ui
#     full_history_cleared = strip_think_tags(full_history_ui)

#     return gr.update(value=full_cleaned), gr.update(value=full_history_cleared), state, "" #gr.update(value=chat_ui), 

# ****************************************************************




# def chat_with_llm(user_msg, state):
#     if "llm_history" not in state or not state["llm_history"]:
#         return state.get("llm_history", []), state

#     state["llm_history"].append({"role": "user", "content": user_msg})
    
#     text_prompt = tokenizer.apply_chat_template(
#         state["llm_history"], tokenize=False, add_generation_prompt=True, enable_thinking=True
#     )
    
#     inputs = tokenizer(text_prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).to("cuda")
#     output_ids = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=1024)
#     response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
#     response = response.split("<|assistant|>\n")[-1].strip()

#     state["llm_history"].append({"role": "assistant", "content": response})
    
#     messages = [(msg["content"], None) if msg["role"] == "user" else (None, msg["content"]) for msg in state["llm_history"] if msg["role"] != "system"]
#     return messages, state

# demo.launch(share=True)

# with gr.Blocks() as demo:
#     gr.Markdown("## Pythera Clara (Clinical Language Analytics and Reasoning AI)")

#     shared_image = gr.State(None)   
#     # state_clara = gr.State({"qwen_history": [], "last_image": None})
#     state_clara = gr.State({"qwen_history": [], "last_image": None, "llm_history": []})

#     state_gemini = gr.State([])
#     state_gpt = gr.State([])

#     with gr.Row():
#         # ==== CỘT TRÁI ====
#         with gr.Column(scale=1):
#             sex_input = gr.Dropdown(choices=["Nam", "Nữ"], label="Giới tính")
#             age_input = gr.Textbox(label="Tuổi", placeholder="VD: 45")
#             view_input = gr.Dropdown(choices=["PA", "Lateral", "AP"], label="Góc chụp", value="PA")
#             image_input = gr.Image(type="pil", label="Upload ảnh X-quang")

#             # ==== TEST CASE ====
#             examples = [
#                 ["image_test/test.png", "Nam", "81", "PA"],
#                 ["image_test/test2.png", "Nữ", "75", "AP"],
#                 ["image_test/test1.png", "Nam", "62", "PA"],
#                 ["image_test/test3.png", "Nam", "48", "PA"],
#                 ["image_test/test4.png", "Nam", "84", "PA"],
#                 ["image_test/test_1.png", "Nam", "53", "PA"],
#                 ["image_test/test_2.png", "Nam", "35", "PA"],
#                 ["image_test/test_3.png", "Nam", "79", "PA"],
#                 ["image_test/test_4.png", "Nữ", "33", "PA"],
#                 # ["/home/tiennv/phucth/medical/data_test/data/testcase_medical/image_test/test_5.png", "Nam", "55", "Lateral"],

#             ]

#             gr.Examples(
#                 examples=examples,
#                 inputs=[image_input, sex_input, age_input, view_input],
#                 label="🧪 Chọn Test Case mẫu"
#             )

#             # ==== RESET ====
#             reset_btn = gr.Button("🗑️ Reset tất cả", variant="stop")

#         # ==== CỘT PHẢI ====
#         with gr.Column(scale=2):
#             with gr.Tabs():
#                 with gr.Tab("Clara"):
#                     clara_model_selector = gr.Dropdown(
#                         choices=["Clara-custom"],  # , "Clara-mini"
#                         value="Clara",
#                         label="model Clara"
#                     )
#                     submit_c = gr.Button("Phân tích với Clara")
#                     # chatbot_c = gr.Chatbot(label="Kết quả từ Clara", height=500)
#                     # # chatbot_llm = gr.Chatbot(label="Trợ lý bác sĩ (LLM)", height=400)
#                     # chatbot_ui = gr.Chatbot(label="Trợ lý chẩn đoán tổng hợp", height=600)

#                     # llm_input = gr.Textbox(label="Nhập câu hỏi", placeholder="Ví dụ: Bệnh nhân có cần nhập viện không?")
#                     # # llm_send = gr.Button("Gửi câu hỏi đến LLM")
#                     # # llm_input = gr.Textbox(label="Nhập câu hỏi", placeholder="Ví dụ: ...")
                    
#                     # llm_send = gr.Button("Gửi câu hỏi đến LLM")
#                     chatbot_ui = gr.Chatbot(label="Trợ lý", height=400, render_markdown=True)
#                     # chatbot_llm = gr.Chatbot(label="💬 Trợ lý LLM", height=400, render_markdown=True)
#                     llm_input = gr.Textbox(label="Nhập câu hỏi")
#                     # llm_full_reply = gr.Textbox(label="Câu trả lời chi tiết", lines=10, interactive=False)
#                     llm_full_reply = gr.Markdown(label="📄 Câu trả lời chi tiết")

#                     # llm_full_history = gr.Textbox(label="🧾 Lịch sử đầy đủ", lines=20, interactive=False)
#                     with gr.Accordion("🧾 Xem lịch sử đầy đủ", open=False):
#                         llm_full_history = gr.Markdown(visible=True)  # Hoặc Textbox nếu bạn không dùng markdown

                    
#                     # full_textbox = gr.Textbox(label="Nội dung chi tiết")
#                     state_clara = gr.State({"llm_history": []})
#                     submit_c.click(
#                         fn=safe_clara_infer,
#                         inputs=[shared_image, clara_model_selector, state_clara, sex_input, age_input, view_input],
#                         outputs=[chatbot_ui, state_clara]
#                     )
#                     # llm_input.submit(
#                     #     fn=llm_gr_chat,
#                     #     inputs=[llm_input, state_clara],
#                     #     outputs=[chatbot_ui, state_clara]
#                     # )


#                     # Gửi khi nhấn nút
#                     # llm_send.click(fn=llm_gr_chat, inputs=[llm_input, state_clara], outputs=[chatbot_ui, state_clara])
#                     # Gửi khi Enter


#                     # llm_input.submit(fn=llm_gr_chat_1, inputs=[llm_input, state_clara], outputs=[chatbot_ui,full_textbox, state_clara])

#                     llm_input.submit(fn=llm_gr_chat_1, 
#                         inputs=[llm_input, state_clara], 
#                         outputs=[llm_full_reply, llm_full_history, state_clara, llm_input]) #chatbot_llm, 


#                 # with gr.Tab("Gemini"):
#                 #     submit_gemini = gr.Button("Phân tích với Gemini")
#                 #     output_gemini = gr.Textbox(label="Kết quả từ Gemini", lines=10)

#                 #     submit_gemini.click(
#                 #         fn=safe_gemini_infer,
#                 #         inputs=[shared_image, sex_input, age_input, view_input, state_gemini],
#                 #         outputs=[output_gemini, state_gemini]
#                 #     )

#                 # with gr.Tab("GPT-4o"):
#                 #     submit_gpt = gr.Button("Phân tích với GPT-4o")
#                 #     output_gpt = gr.Textbox(label="Kết quả từ GPT-4o", lines=10)

#                 #     submit_gpt.click(
#                 #         fn=safe_gpt_infer,
#                 #         inputs=[shared_image, sex_input, age_input, view_input, state_gpt],
#                 #         outputs=[output_gpt, state_gpt]
#                 #     )

#     # === CẬP NHẬT ẢNH DÙNG CHUNG ===
#     def update_shared_image(image):
#         return image

#     image_input.change(fn=update_shared_image, inputs=image_input, outputs=shared_image)

#     # === RESET TOÀN BỘ ===
#     # def reset_all():
#     #     return (
#     #         None,       # image_input
#     #         None,       # sex_input
#     #         "",         # age_input
#     #         "PA",       # view_input
#     #         {"llm_history": []},  # state_clara
#     #         {"gemini_history": []},  # state_gemini
#     #         {"gpt_history": []},     # state_gpt
#     #         [],         # chatbot_ui
#     #         "",         # output_gemini
#     #         "",         # output_gpt
#     #         "",         # llm_full_reply
#     #         ""          # llm_full_history
#     #     )
#     # reset_btn.click(
#     #     fn=reset_all,
#     #     inputs=[],
#     #     outputs=[
#     #         image_input,
#     #         sex_input,
#     #         age_input,
#     #         view_input,
#     #         state_clara,
#     #         state_gemini,
#     #         state_gpt,
#     #         chatbot_ui,
#     #         output_gemini,
#     #         output_gpt,
#     #         llm_full_reply,
#     #         llm_full_history
#     #     ]
#     # )
#     def reset_all():
#         return (
#             None,        # image_input
#             None,        # sex_input
#             "",          # age_input
#             "PA",        # view_input
#             {"llm_history": []},  # state_clara
#             [],          # state_gemini
#             [],          # state_gpt
#             [],          # chatbot_ui (Trợ lý chẩn đoán)
#             "",          # llm_full_reply
#             "",          # llm_full_history
#             # "",          # output_gemini
#             # "",          # output_gpt
#         )




#     reset_btn.click(
#         fn=reset_all,
#         inputs=[],
#         outputs=[
#             image_input,       # 1
#             sex_input,         # 2
#             age_input,         # 3
#             view_input,        # 4
#             state_clara,       # 5
#             state_gemini,      # 6
#             state_gpt,         # 7
#             chatbot_ui,        # 8 (nếu bạn dùng chatbot chẩn đoán tổng hợp)
#             llm_full_reply,    # 9
#             llm_full_history,  # 10
#             # output_gemini,     # 11
#             # output_gpt         # 12
#         ]
#     )




# # demo.launch(share=True)
# demo.launch(
#     share=True,
#     allowed_paths=["/home/clara/phucth/code/project/Medical_CLARA/infer/demo_clara/image_test"]
# )


import gradio as gr

with gr.Blocks() as demo:
    gr.Markdown("## Pythera Clara (Clinical Language Analytics and Reasoning AI)")

    # ==== State dùng chung ====
    shared_image = gr.State(None)
    state_clara = gr.State({"qwen_history": [], "last_image": None, "llm_history": []})
    state_gemini = gr.State([])
    state_gpt = gr.State([])

    with gr.Row():
        # ==== CỘT TRÁI: Form upload + thông tin bệnh nhân ====
        with gr.Column(scale=1):
            sex_input = gr.Dropdown(choices=["Nam", "Nữ"], label="Giới tính")
            age_input = gr.Textbox(label="Tuổi", placeholder="VD: 45")
            view_input = gr.Dropdown(choices=["PA", "Lateral", "AP"], label="Góc chụp", value="PA")
            image_input = gr.Image(type="pil", label="Upload ảnh X-quang")

            # ==== Test case mẫu ====
            examples = [
                ["image_test/test.png", "Nam", "81", "PA"],
                ["image_test/test2.png", "Nữ", "75", "AP"],
                ["image_test/test1.png", "Nam", "62", "PA"],
                ["image_test/test3.png", "Nam", "48", "PA"],
                ["image_test/test4.png", "Nam", "84", "PA"],
                ["image_test/test_1.png", "Nam", "53", "PA"],
                ["image_test/test_2.png", "Nam", "35", "PA"],
                ["image_test/test_3.png", "Nam", "79", "PA"],
                ["image_test/test_4.png", "Nữ", "33", "PA"],
            ]

            gr.Examples(
                examples=examples,
                inputs=[image_input, sex_input, age_input, view_input],
                label="🧪 Chọn Test Case mẫu"
            )

        # ==== CỘT PHẢI: Clara Tab ====
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("Clara"):
                    # ==== Dropdown chọn model ====
                    clara_model_selector = gr.Dropdown(
                        choices=["Clara"],
                        value="Clara",   # chọn sẵn
                        label="Model Clara"
                    )

                    # ==== Row nút Submit + Reset nằm cạnh nhau ====
                    with gr.Row():
                        submit_c = gr.Button("Phân tích với Clara")
                        reset_btn = gr.Button("🗑️ Reset tất cả", variant="stop")

                    # ==== Chatbot hiển thị kết quả ====
                    chatbot_ui = gr.Chatbot(label="Trợ lý", height=400, render_markdown=True)
                    llm_input = gr.Textbox(label="Nhập câu hỏi")
                    llm_full_reply = gr.Markdown(label="📄 Câu trả lời chi tiết")
                    with gr.Accordion("🧾 Xem lịch sử đầy đủ", open=False):
                        llm_full_history = gr.Markdown(visible=True)

                    state_clara = gr.State({"llm_history": []})

                    # ==== Submit click ====
                    submit_c.click(
                        fn=safe_clara_infer,
                        inputs=[shared_image, clara_model_selector, state_clara, sex_input, age_input, view_input],
                        outputs=[chatbot_ui, state_clara]
                    )

                    # ==== Reset click ====
                    def reset_all():
                        return (
                            None,        # image_input
                            None,        # sex_input
                            "",          # age_input
                            "PA",        # view_input
                            {"llm_history": []},  # state_clara
                            # [],          # state_gemini
                            # [],          # state_gpt
                            # [],          # chatbot_ui
                            # "",          # llm_full_reply
                            # "",          # llm_full_history
                        )

                    reset_btn.click(
                        fn=reset_all,
                        inputs=[],
                        outputs=[
                            image_input,
                            sex_input,
                            age_input,
                            view_input,
                            state_clara,
                            # state_gemini,
                            # state_gpt,
                            # chatbot_ui,
                            # llm_full_reply,
                            # llm_full_history
                        ]
                    )

                    # # ==== LLM input submit (Enter) ====
                    # llm_input.submit(
                    #     fn=llm_gr_chat_1,
                    #     inputs=[llm_input, state_clara],
                    #     outputs=[llm_full_reply, llm_full_history, state_clara, llm_input]
                    # )

    # === Cập nhật ảnh dùng chung ===
    def update_shared_image(image):
        return image

    image_input.change(fn=update_shared_image, inputs=image_input, outputs=shared_image)

# === Launch demo ===
demo.launch(
    share=True,
    allowed_paths=["/home/clara/phucth/code/project/Medical_CLARA/infer/demo_clara/image_test"]
)
