# vllm_infer.py
from openai import OpenAI
import base64
import io

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8001/v1")


def encode_pil_image(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return img_base64

def call_qwen2vl_vllm(pil_image, sex, age, view):
    img_base64 = encode_pil_image(pil_image)

    view_text = f"X-ray {view}" if view else "X-ray"
    prompt = f"Ảnh chụp {view_text} bệnh nhân {sex.lower()}, {age} tuổi. Cho biết bệnh nhân bị gì?"

    response = client.chat.completions.create(
        model="THP2903/Qwen2vl_7b_instruct_medical_multiturn_full",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                    },
                    {"type": "text", "text": prompt}
                ]
            }
        ],
        temperature=0.2,
        max_tokens=1024,
    )

    return response.choices[0].message.content
# app.py
import gradio as gr
from check_xray import is_chest_xray

# def safe_clara_infer(image, model_selector, state, sex, age, view):
#     if image is None:
#         return [[None, "❌ Bạn cần upload ảnh X-quang."]], state, None
#     if not is_chest_xray(image):
#         return [[None, "❌ Đây không phải ảnh X-quang ngực."]], state, None
#     return clara_infer(image, model_selector, state, sex, age, view)


def safe_clara_infer_vllm(image, model_selector, state, sex, age, view):
    # if image is None:
    #     return [[None, "❌ Bạn cần upload ảnh X-quang."]], state, None
    # if not is_chest_xray(image):
    #     return [[None, "❌ Đây không phải ảnh X-quang ngực."]], state, None

    try:
        output_text = call_qwen2vl_vllm(image, sex, age, view)
    except Exception as e:
        return [[None, f"❌ Lỗi khi gọi vLLM: {str(e)}"]], state, None

    prompt = f"Ảnh chụp X-ray {view}, bệnh nhân {sex}, {age} tuổi. Cho biết bệnh nhân bị gì?"
    return [[prompt, output_text]], state, None

with gr.Blocks() as demo:
    gr.Markdown("## 🧠 AI Clara - Phân tích ảnh X-quang")

    shared_image = gr.State(None)
    state_clara = gr.State({})

    with gr.Row():
        with gr.Column(scale=1):
            sex_input = gr.Dropdown(choices=["Nam", "Nữ"], label="Giới tính")
            age_input = gr.Textbox(label="Tuổi", placeholder="VD: 45")
            view_input = gr.Dropdown(choices=["PA", "Lateral", "AP"], label="Góc chụp", value="PA")
            image_input = gr.Image(type="pil", label="Upload ảnh X-quang")

            examples = [
                ["image_test/test1.png", "Nam", "62", "PA"],
                ["image_test/test2.png", "Nữ", "75", "AP"],
                ["image_test/test3.png", "Nam", "48", "PA"],
            ]
            gr.Examples(
                examples=examples,
                inputs=[image_input, sex_input, age_input, view_input],
                label="🧪 Test Case mẫu"
            )

            reset_btn = gr.Button("🗑️ Reset")

        with gr.Column(scale=2):
            clara_model_selector = gr.Dropdown(
                choices=["Clara"],
                value="Clara",
                label="Chọn model"
            )
            submit_c = gr.Button("Phân tích với Clara")
            chatbot_c = gr.Chatbot(label="Kết quả từ Clara", height=500)

            submit_c.click(
                fn=safe_clara_infer_vllm,
                inputs=[shared_image, clara_model_selector, state_clara, sex_input, age_input, view_input],
                outputs=[chatbot_c, state_clara, image_input]
            )

    def update_shared_image(image):
        return image

    image_input.change(fn=update_shared_image, inputs=image_input, outputs=shared_image)

    def reset_all():
        return (
            None, None, "", "PA", {}, None
        )

    reset_btn.click(
        fn=reset_all,
        inputs=[],
        outputs=[image_input, sex_input, age_input, view_input, state_clara, chatbot_c]
    )

demo.launch(share=True)
