# from fastapi import FastAPI, File, UploadFile, Form
# from fastapi.responses import JSONResponse
# from openai import OpenAI
# import base64
# from PIL import Image
# from io import BytesIO
# import time
# from typing import Optional

# app = FastAPI(title="Medical VLM API")

# # ---- Client connect to vLLM ----
# client = OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:8006/v1")
# model_path = "/home/clara/phucth/code/project/public_dataset_med/public_data_medical/Qwen2-VL-7B-multi-137k_full_maed_v3"

# def encode_image_from_bytes(image_bytes):
#     """Encode image từ bytes"""
#     img = Image.open(BytesIO(image_bytes)).convert("RGB").resize((448, 448))
#     buffer = BytesIO()
#     img.save(buffer, format="PNG")
#     return base64.b64encode(buffer.getvalue()).decode("utf-8")

# @app.post("/predict")
# async def predict(
#     image: UploadFile = File(...),
#     question1: str = Form(...),
#     question2: Optional[str] = Form(None)
# ):
#     """
#     API nhận ảnh và câu hỏi, trả về kết quả từ VLM
    
#     - image: file ảnh X-ray
#     - question1: câu hỏi đầu tiên
#     - question2: câu hỏi thứ 2 (optional)
#     """
#     try:
#         # Đọc và encode image
#         image_bytes = await image.read()
#         image_base64 = encode_image_from_bytes(image_bytes)
        
#         # ---- TURN 1 ----
#         messages = []
#         messages.append({
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": question1},
#                 {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
#             ]
#         })
        
#         start_t1 = time.time()
#         response1 = client.chat.completions.create(
#             model=model_path,
#             messages=messages,
#             temperature=1.0,
#             max_tokens=1024
#         )
#         latency1 = time.time() - start_t1
        
#         answer1 = response1.choices[0].message.content.strip()
        
#         result = {
#             "turn1": {
#                 "question": question1,
#                 "answer": answer1,
#                 "latency": round(latency1, 3)
#             }
#         }
        
#         # ---- TURN 2 (nếu có) ----
#         if question2:
#             messages.append({"role": "assistant", "content": [{"type": "text", "text": answer1}]})
#             messages.append({"role": "user", "content": [{"type": "text", "text": question2}]})
            
#             start_t2 = time.time()
#             response2 = client.chat.completions.create(
#                 model=model_path,
#                 messages=messages,
#                 temperature=0.7,
#                 max_tokens=1024
#             )
#             latency2 = time.time() - start_t2
            
#             answer2 = response2.choices[0].message.content.strip()
            
#             result["turn2"] = {
#                 "question": question2,
#                 "answer": answer2,
#                 "latency": round(latency2, 3)
#             }
#             result["total_latency"] = round(latency1 + latency2, 3)
        
#         return JSONResponse(content=result)
    
#     except Exception as e:
#         return JSONResponse(
#             status_code=500,
#             content={"error": str(e)}
#         )

# @app.get("/health")
# async def health_check():
#     """Check xem API có sống không"""
#     return {"status": "ok", "message": "Medical VLM API is running"}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from openai import OpenAI
import base64
from PIL import Image
from io import BytesIO
import time
from typing import Optional

app = FastAPI(title="Medical VLM API")

client = OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:8006/v1")
model_path = "/home/clara/phucth/code/project/public_dataset_med/public_data_medical/Qwen2-VL-7B-multi-137k_full_maed_v3"  # tốt
# model_path = "/home/clara/phucth/code/project/public_dataset_med/public_data_medical/Qwen2-VL-7B-multi-137k_full_maed_v2"  # chưa ok
# model_path = "/home/clara/phucth/code/project/public_dataset_med/public_data_medical/Qwen2-VL-7B-multi-137k_full_maed"  # tốt


def encode_image_from_bytes(image_bytes):
    """Encode image từ bytes"""
    img = Image.open(BytesIO(image_bytes)).convert("RGB").resize((448, 448))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    sex: Optional[str] = Form(None),      # Nam/Nữ, optional
    age: Optional[str] = Form(None)       # Tuổi, optional
):
    """
    API nhận ảnh X-ray và thông tin bệnh nhân (optional), trả về kết quả từ VLM
    
    - image: file ảnh X-ray
    - sex: giới tính bệnh nhân (Nam/Nữ) - optional
    - age: tuổi bệnh nhân - optional
    """
    try:
        # Đọc và encode image
        image_bytes = await image.read()
        image_base64 = encode_image_from_bytes(image_bytes)
        
        # ---- XÂY DỰNG CÂU HỎI 1 THEO TEMPLATE ----
        patient_info = []
        if sex:
            patient_info.append(f"{sex}")
        if age:
            patient_info.append(f"{age} tuổi")
        
        patient_str = ", ".join(patient_info) #  if patient_info else "không rõ thông tin"
        question1 = f"Ảnh chụp X-ray Chụp Xquang tim phổi thẳng bệnh nhân {patient_str}. Cho biết bệnh nhân bị gì?"
        
        # ---- TURN 1 ----
        messages = []
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": question1},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ]
        })
        
        start_t1 = time.time()
        response1 = client.chat.completions.create(
            model=model_path,
            messages=messages,
            temperature=0.2,
            # do_sample=False,
            max_tokens=1024
        )
        latency1 = time.time() - start_t1
        
        answer1 = response1.choices[0].message.content.strip()
        
        result = {
            "patient_info": {
                "sex": sex ,
                "age": age 
            },
            "turn1": {
                "question": question1,
                "answer": answer1,
                "latency": round(latency1, 3)
            }
        }
        
        # ---- TURN 2 (CÂU HỎI 2 SET CỨNG) ----
        question2 = "Kết luận từ thông tin trên bệnh nhân bị gì?"
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer1}]})
        messages.append({"role": "user", "content": [{"type": "text", "text": question2}]})
        
        start_t2 = time.time()
        response2 = client.chat.completions.create(
            model=model_path,
            messages=messages,
            temperature=0.2,
            # do_sample=False,
            # temperature=0.1,
            max_tokens=1024
        )
        latency2 = time.time() - start_t2
        
        answer2 = response2.choices[0].message.content.strip()
        
        result["turn2"] = {
            "question": question2,
            "answer": answer2,
            "latency": round(latency2, 3)
        }
        result["total_latency"] = round(latency1 + latency2, 3)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/health")
async def health_check():
    """Check xem API có sống không"""
    return {"status": "ok", "message": "Medical VLM API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
