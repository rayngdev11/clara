import gradio as gr
import requests
import json
from PIL import Image
from io import BytesIO
from datetime import datetime
import os
from check_xray import is_chest_xray

# API endpoint
API_BASE_URL = "http://127.0.0.1:8000"

# File lưu kết quả đánh giá
EVAL_FILE = "evaluation_results.jsonl"

# Biến global để lưu kết quả hiện tại
current_result = {}

def safe_clara_infer(image, sex, age):
    """Gọi API FastAPI"""
    global current_result
    current_result = {}  # Reset
    
    if image is None:
        return [[None, "❌ Bạn cần upload ảnh X-quang."]], "", "", ""
    
    if not is_chest_xray(image):
        return [[None, "❌ Đây không phải ảnh X-quang ngực."]], "", "", ""
    
    try:
        img_buffer = BytesIO()
        image.save(img_buffer, format='PNG')
        image_bytes = img_buffer.getvalue()
        
        files = {'image': ('xray.png', image_bytes, 'image/png')}
        data = {
            'sex': sex if sex in ["Nam","Nữ"] else None,
            'age': age if age.strip() != "" else None
        }
        
        response = requests.post(
            f"{API_BASE_URL}/predict",
            files=files,
            data=data,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Lấy answer từ turn1 và turn2
            turn1_answer = result.get('turn1', {}).get('answer', '')
            turn2_answer = result.get('turn2', {}).get('answer', '')
            
            # Xử lý turn2_answer: bỏ "🎯 KẾT LUẬN:\n" nếu có
            if turn2_answer.startswith("**🎯 KẾT LUẬN:**\n"):
                turn2_answer = turn2_answer.replace("**🎯 KẾT LUẬN:**\n", "")
            elif turn2_answer.startswith("🎯 KẾT LUẬN:\n"):
                turn2_answer = turn2_answer.replace("🎯 KẾT LUẬN:\n", "")
            
            # Lưu kết quả vào biến global (CHỈ LƯU ANSWER THUẦN, KHÔNG CÓ PREFIX)
            current_result = {
                'patient_info': result.get('patient_info', {}),
                'turn1_generated': turn1_answer.strip(),
                'turn2_generated': turn2_answer.strip(),
                'total_latency': result.get('total_latency', 0),
                'timestamp': datetime.now().isoformat()
            }
            
            # Format messages cho chatbot UI
            messages = []
            patient_info = result.get('patient_info', {})
            turn1 = result.get('turn1', {})
            turn2 = result.get('turn2', {})
            
            patient_str = f"**Thông tin BN:** {patient_info.get('sex', 'N/A')}, {patient_info.get('age', 'N/A')}"
            messages.append([None, patient_str])
            messages.append([turn1.get('question', ''), turn1_answer])
            
            if turn2:
                messages.append([turn2.get('question', ''), f"**🎯 KẾT LUẬN:**\n{turn2_answer}"])
                messages.append([None, f"⏱️ **Tổng thời gian:** {result.get('total_latency', 0)}s"])
            
            # Trả về messages + auto-fill các ô ground_truth (KHÔNG CÓ PREFIX)
            return (
                messages,
                current_result['turn1_generated'],  # Auto-fill turn1 - CHỈ CÓ ANSWER
                current_result['turn2_generated'],  # Auto-fill turn2 - CHỈ CÓ ANSWER
                "5"  # Default score
            )
        
        else:
            return [[None, f"❌ Lỗi API: {response.status_code}"]], "", "", ""
            
    except requests.exceptions.ConnectionError:
        return [[None, "❌ Không kết nối được API FastAPI"]], "", "", ""
    except Exception as e:
        return [[None, f"❌ Lỗi: {str(e)}"]], "", "", ""

def save_evaluation(gt_turn1, gt_turn2, score):
    """Lưu đánh giá của bác sĩ vào file JSONL"""
    global current_result
    
    if not current_result:
        return "❌ Chưa có kết quả nào để lưu. Vui lòng phân tích ảnh trước!"
    
    try:
        score = int(score)
        if not (1 <= score <= 5):
            return "❌ Score phải từ 1-5"
    except:
        return "❌ Score không hợp lệ"
    
    if not gt_turn1.strip() or not gt_turn2.strip():
        return "❌ Vui lòng nhập đầy đủ ground truth cho cả 2 turn"
    
    # Tạo 2 dòng JSONL (1 cho mỗi turn)
    records = [
        {
            "sample_idx": get_next_sample_idx(),
            "answer_idx": 1,
            "generated": current_result['turn1_generated'],
            "ground_truth": gt_turn1.strip(),
            "score": score
        },
        {
            "sample_idx": get_next_sample_idx(),
            "answer_idx": 2,
            "generated": current_result['turn2_generated'],
            "ground_truth": gt_turn2.strip(),
            "score": score
        }
    ]
    
    # Ghi vào file JSONL
    with open(EVAL_FILE, 'a', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    return f"✅ Đã lưu đánh giá thành công vào {EVAL_FILE}\n📊 Sample #{records[0]['sample_idx']}"

def get_next_sample_idx():
    """Lấy sample_idx tiếp theo"""
    if not os.path.exists(EVAL_FILE):
        return 0
    
    try:
        with open(EVAL_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if not lines:
                return 0
            last_line = json.loads(lines[-1])
            return last_line.get('sample_idx', 0) + 1
    except:
        return 0

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

def view_evaluations():
    """Xem các đánh giá đã lưu"""
    if not os.path.exists(EVAL_FILE):
        return "Chưa có đánh giá nào được lưu"
    
    try:
        with open(EVAL_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if not lines:
                return "File rỗng"
            
            # Hiển thị 10 dòng cuối
            recent = lines[-20:]
            output = f"**📊 {len(lines)} dòng trong file {EVAL_FILE}**\n\n"
            output += "**20 đánh giá gần nhất:**\n```json\n"
            output += ''.join(recent)
            output += "```"
            return output
    except Exception as e:
        return f"Lỗi đọc file: {str(e)}"

# ===== GRADIO INTERFACE =====
with gr.Blocks(
    title="CLARA - Vision-Language AI for Chest X-ray Caption Automation",
    theme=gr.themes.Soft()
) as demo:
    gr.Markdown("""
    # **Clara** 
    *Clinical Language Analytics and Reasoning AI*
    
    **Upload ảnh X-quang → Phân tích → Bác sĩ đánh giá → Lưu JSONL**
    """)
    
    gr.Markdown("---")
    
    with gr.Row():
        # === CỘT TRÁI: INPUT ===
        with gr.Column(scale=1):
            gr.Markdown("### Thông tin bệnh nhân")
            
            with gr.Row():
                sex_input = gr.Dropdown(
                    choices=["Nam", "Nữ"], 
                    label="Giới tính", 
                    value=None,
                    allow_custom_value=True
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
            
            submit_btn = gr.Button("Phân tích ngay", variant="primary", size="lg")
            
            # Examples
            examples = [
                ["image_test/image_bt.png", "Nam", "81"],
                ["image_test/image(2).png", "Nữ", "75"],
                ["image_test/image(3).png", "", ""],
            ]
            
            gr.Examples(
                examples=examples,
                inputs=[image_input, sex_input, age_input],
                label="Test cases mẫu"
            )
        
        # === CỘT PHẢI: OUTPUT ===
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("Kết quả AI"):
                    chatbot_ui = gr.Chatbot(
                        label="Kết quả phân tích", 
                        height=400,
                        show_label=True
                    )
                    
                    gr.Markdown("---")
                    gr.Markdown("### 👨‍⚕️ Đánh giá của Bác sĩ")
                    
                    with gr.Row():
                        with gr.Column(scale=3):
                            gt_turn1 = gr.Textbox(
                                label="Ground Truth - Turn 1",
                                placeholder="VD: - BÓNG TIM TO\n- CÓ MÁY TẠO NHỊP",
                                lines=3
                            )
                        with gr.Column(scale=3):
                            gt_turn2 = gr.Textbox(
                                label="Ground Truth - Turn 2",
                                placeholder="VD: - QUÃNG ĐỒNG MẠCH CHỦ RỘNG...",
                                lines=3
                            )
                        with gr.Column(scale=1):
                            score_input = gr.Dropdown(
                                choices=["1", "2", "3", "4", "5"],
                                label="Score",
                                value="5",
                                info="1=Sai hoàn toàn, 5=Hoàn hảo"
                            )
                    
                    with gr.Row():
                        save_btn = gr.Button("Lưu đánh giá", variant="primary")
                        clear_btn = gr.Button("Xóa tất cả", variant="secondary")
                    
                    save_status = gr.Markdown("")
                
                with gr.Tab("Xem đánh giá đã lưu"):
                    eval_display = gr.Markdown("Chưa có dữ liệu")
                    refresh_btn = gr.Button("Refresh")
                    refresh_btn.click(view_evaluations, outputs=eval_display)
                
                with gr.Tab("API Status"):
                    status_output = gr.Markdown("**Kiểm tra:** `curl http://127.0.0.1:8000/health`")
                    check_btn = gr.Button("🔍 Check API")
                    check_btn.click(check_api_status, outputs=status_output)
    
    # === EVENTS ===
    submit_btn.click(
        fn=safe_clara_infer,
        inputs=[image_input, sex_input, age_input],
        outputs=[chatbot_ui, gt_turn1, gt_turn2, score_input]
    )
    
    save_btn.click(
        fn=save_evaluation,
        inputs=[gt_turn1, gt_turn2, score_input],
        outputs=save_status
    )
    
    clear_btn.click(
        lambda: (None, "", "", [], "", "", "5", ""),
        outputs=[image_input, sex_input, age_input, chatbot_ui, gt_turn1, gt_turn2, score_input, save_status]
    )

# ===== LAUNCH =====
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=True,
        show_api=True,
        allowed_paths=["image_test"]
    )