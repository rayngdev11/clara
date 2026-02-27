import gradio as gr
import requests
import json
from PIL import Image
import base64
from io import BytesIO
from check_xray import is_chest_xray  # Giữ nguyên function check của bạn

# API endpoint
API_BASE_URL = "http://127.0.0.1:8000"  # Port FastAPI của bạn

def safe_clara_infer(image, sex, age):
    """Gọi API FastAPI thay vì model trực tiếp"""
    if image is None:
        return [[None, "❌ Bạn cần upload ảnh X-quang."]]
    
    # Check ảnh X-quang
    if not is_chest_xray(image):
        return [[None, "❌ Đây không phải ảnh X-quang ngực."]]
    
    try:
        # Convert PIL Image to bytes
        img_buffer = BytesIO()
        image.save(img_buffer, format='PNG')
        image_bytes = img_buffer.getvalue()
        
        # Prepare form data cho API
        files = {'image': ('xray.png', image_bytes, 'image/png')}
        # data = {
        #     'sex': sex or None,
        #     'age': age or None
        # }
        data = {
            'sex': sex if sex in ["Nam","Nữ"] else None,
            'age': age if age.strip() != "" else None
        }

        
        # Gọi API
        response = requests.post(
            f"{API_BASE_URL}/predict",
            files=files,
            data=data,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Format kết quả cho chatbot
            patient_info = result.get('patient_info', {})
            turn1 = result.get('turn1', {})
            turn2 = result.get('turn2', {})
            
            # Tạo message đẹp
            messages = []
            
            # Message thông tin bệnh nhân
            patient_str = f"**Thông tin BN:** {patient_info.get('sex', 'N/A')}, {patient_info.get('age', 'N/A')}"
            messages.append([None, patient_str])
            
            # Turn 1
            messages.append([turn1.get('question', ''), turn1.get('answer', '')])
            
            # Turn 2
            if turn2:
                messages.append([turn2.get('question', ''), f"**🎯 KẾT LUẬN:**\n{turn2.get('answer', '')}"])
                messages.append([None, f"⏱️ **Tổng thời gian:** {result.get('total_latency', 0)}s"])
            
            return messages
        
        else:
            return [[None, f"❌ Lỗi API: {response.status_code} - {response.text}"]]
            
    except requests.exceptions.ConnectionError:
        return [[None, "❌ Không kết nối được API FastAPI. Kiểm tra server port 8000"]]
    except Exception as e:
        return [[None, f"❌ Lỗi: {str(e)}"]]

def check_api_status():
    """Check API status"""
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if resp.status_code == 200:
            return f"✅ **API OK** - {resp.json()}"
        else:
            return f"❌ **API Error** - Status: {resp.status_code}"
    except:
        return "❌ **API Offline** - Kiểm tra FastAPI port 8000"

# ===== GRADIO INTERFACE =====
with gr.Blocks(
    title="CLARA - Vision-Language AI for Chest X-ray Caption Automation",
    theme=gr.themes.Soft()
) as demo:
    gr.Markdown("""
    # **Clara** 
    *Clinical Language Analytics and Reasoning AI*
    
    **Upload ảnh X-quang tim phổi → Nhập thông tin (NẾU CÓ) → Nhận kết luận tự động**
    """)
    
    gr.Markdown("---")
    
    with gr.Row():
        # === CỘT TRÁI: INPUT ===
        with gr.Column(scale=1):
            gr.Markdown("### 📋 Thông tin bệnh nhân")
            
            with gr.Row():
                sex_input = gr.Dropdown(
                    choices=["Nam", "Nữ"], 
                    label="Giới tính", 
                    value=None,              # KHÔNG đặt mặc định
                    allow_custom_value=True  # Cho phép rỗng hoặc giá trị tùy ý
                )
                age_input = gr.Textbox(
                    label="Tuổi", 
                    placeholder="VD: 45", 
                    max_lines=1,
                    value=""
                )
            

            
            image_input = gr.Image(
                type="pil", 
                label="Upload ảnh X-quang tim phổi",
                height=300
            )
            
            # Examples
            examples = [
                ["image_test/image_bt.png", "Nam", "81"],
                ["image_test/image(2).png", "Nữ", "75"],
                ["image_test/image(3).png", "", ""],
                ["image_test/image(5).png", "", ""],
                ["image_test/test4.png", "", ""],
                ["image_test/123.7575738677987.1789116863351791.png","",""]
            ]
            
            gr.Examples(
                examples=examples,
                inputs=[image_input, sex_input, age_input],
                label="🧪 Test cases mẫu"
            )
        
        # === CỘT PHẢI: OUTPUT ===
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("🎯 Kết quả phân tích"):  # Bỏ icon
                    chatbot_ui = gr.Chatbot(
                        label="Kết quả AI", 
                        height=500,
                        show_label=True
                    )
                    
                    with gr.Row():
                        submit_btn = gr.Button("Phân tích ngay", variant="primary")
                        clear_btn = gr.Button("Xóa tất cả", variant="secondary")
            
            with gr.Tab("API Status"):  # Bỏ icon
                status_output = gr.Markdown("**Kiểm tra:** `curl http://127.0.0.1:8000/health`")
                check_btn = gr.Button("🔍 Check API")
                check_btn.click(check_api_status, outputs=status_output)
    
    # === EVENTS ===
    submit_btn.click(
        fn=safe_clara_infer,
        inputs=[image_input, sex_input, age_input],
        outputs=chatbot_ui
    )
    
    clear_btn.click(
        lambda: (None, "", "", []),
        outputs=[image_input, sex_input, age_input, chatbot_ui]
    )
    
    # # Enter submit
    # image_input.change(
    #     fn=safe_clara_infer,
    #     inputs=[image_input, sex_input, age_input], 
    #     outputs=chatbot_ui
    # )

# ===== LAUNCH =====
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_api=True,
        allowed_paths=["image_test"]  # Thư mục chứa test images
    )
