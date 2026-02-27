# import os
# import base64
# import torch
# import numpy as np
# from io import BytesIO
# from PIL import Image
# import triton_python_backend_utils as pb_utils
# from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
# from qwen_vl_utils import process_vision_info
# from PIL import Image
# import torch
# import torchvision.transforms as T

# def preprocess_image(image: Image.Image):
#     # Resize ảnh về 448x448
#     transform = T.Compose([
#         T.Resize((448, 448)),
#         T.ToTensor(),  # convert to [0,1]
#         T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])  # giả định model dùng [-1,1]
#     ])
#     return transform(image).unsqueeze(0)  # shape (1, 3, 448, 448)

# class TritonPythonModel:
#     def initialize(self, args):
#         # Chọn device
#         self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
#         # Path model trong container
#         # MODEL_PATH = os.getenv("MODEL_PATH", "THP2903/Qwen2-VL-7B-multi-137k_full_maed_v3")
#         MODEL_PATH = os.getenv("MODEL_PATH", "/models/vlm/Qwen2-VL-7B-multi-137k_full_maed_v3")


#         # Load model offline
#         self.model = Qwen2VLForConditionalGeneration.from_pretrained(
#             MODEL_PATH,
#             local_files_only=True,
#             torch_dtype=torch.bfloat16,
#             device_map="auto"  # map GPU nếu có
#         )
#         self.model.to(self.device)

#         # Load processor
#         self.processor = AutoProcessor.from_pretrained(MODEL_PATH)

#     def execute(self, requests):
#         responses = []

#         for request in requests:
#             # Lấy input từ Triton
#             text = pb_utils.get_input_tensor_by_name(request, "text_input").as_numpy()[0].decode('utf-8')
#             img_b64 = pb_utils.get_input_tensor_by_name(request, "image_input").as_numpy()[0].decode('utf-8')

#             # Decode base64 image
#             # image = Image.open(BytesIO(base64.b64decode(img_b64)))
#             image = preprocess_image(Image.open(BytesIO(base64.b64decode(img_b64))).convert("RGB"))


#             # Chuẩn bị messages cho Qwen
#             messages = [
#                 {
#                     "role": "user",
#                     "content": [
#                         {"type": "image", "image": image},
#                         {"type": "text", "text": text},
#                     ],
#                 }
#             ]

#             # Chuẩn bị input cho model
#             chat_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#             image_inputs, video_inputs = process_vision_info(messages)
#             inputs = self.processor(
#                 text=[chat_text],
#                 images=image_inputs,
#                 videos=video_inputs,
#                 padding=True,
#                 return_tensors="pt",
#             )
#             inputs = {k: v.to(self.device) for k, v in inputs.items()}

#             # Generate output
#             with torch.inference_mode():
#                 generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
#                 generated_ids_trimmed = [
#                     out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
#                 ]
#                 output_text = self.processor.batch_decode(
#                     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
#                 )[0]

#             # Tạo response cho Triton
#             out_tensor = pb_utils.Tensor(
#                 "text_output", np.array([output_text.encode()], dtype=np.bytes_)
#             )
#             responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

#         return responses


import os
import base64
import torch
import numpy as np
from io import BytesIO
from PIL import Image
import triton_python_backend_utils as pb_utils
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

class TritonPythonModel:
    def initialize(self, args):
        # Device
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        # Model path
        MODEL_PATH = os.getenv("MODEL_PATH", "/models/vlm/Qwen2-VL-7B-multi-137k_full_maed_v3")

        # Load model
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            torch_dtype=torch.bfloat16,  # fp16 giống vLLM
            device_map="auto"
        )
        self.model.to(self.device)

        # Load processor
        self.processor = AutoProcessor.from_pretrained(MODEL_PATH)

        # Lưu multi-turn history
        # self.chat_history = []

    def execute(self, requests):
        responses = []

        for request in requests:
            # Lấy input từ Triton
            text = pb_utils.get_input_tensor_by_name(request, "text_input").as_numpy()[0].decode('utf-8')
            img_b64 = pb_utils.get_input_tensor_by_name(request, "image_input").as_numpy()[0].decode('utf-8')

            # Decode ảnh nếu có, resize 448x448
            if img_b64.strip() == "":
                image = None
            else:
                image = Image.open(BytesIO(base64.b64decode(img_b64))).convert("RGB").resize((448, 448))

            # Tạo message Turn mới
            # messages = self.chat_history.copy()
            msg_content = [{"type": "text", "text": text}]
            if image:
                msg_content.insert(0, {"type": "image", "image": image})
            messages = [{"role": "user", "content": msg_content}]

            messages.append({"role": "user", "content": msg_content})

            # Áp dụng prompt template
            chat_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Process ảnh/video
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[chat_text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate output
            with torch.inference_mode():
                generated_ids = self.model.generate(**inputs, max_new_tokens=1024, temperature=0)
                # Trim prompt part
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]

            # Update history multi-turn
            # self.chat_history.append({"role": "assistant", "content": [{"type": "text", "text": output_text}]})

            # Response cho Triton
            out_tensor = pb_utils.Tensor(
                "text_output", np.array([output_text.encode()], dtype=np.bytes_)
            )
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

        return responses



# import torch
# import random
# import numpy as np

# seed = 42
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# np.random.seed(seed)
# random.seed(seed)

# # Force deterministic cudnn
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# ---------------------------------
# import os
# import base64
# import torch
# import numpy as np
# from io import BytesIO
# from PIL import Image
# import triton_python_backend_utils as pb_utils
# from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
# from qwen_vl_utils import process_vision_info

# class TritonPythonModel:
#     def initialize(self, args):
#         # Chọn device
#         self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
#         # Path model trong container
#         MODEL_PATH = os.getenv("MODEL_PATH", "/models/vlm/Qwen2-VL-7B-multi-137k_full_maed_v3")

#         # Load model
#         self.model = Qwen2VLForConditionalGeneration.from_pretrained(
#             MODEL_PATH,
#             local_files_only=True,
#             torch_dtype=torch.bfloat16,
#             device_map=None  
#         )
#         self.model.to(self.device)
#         self.model.eval()

#         # Load processor
#         self.processor = AutoProcessor.from_pretrained(MODEL_PATH)

#         # Multi-turn history
#         self.chat_history = []

#     def execute(self, requests):
#         responses = []

#         for request in requests:
#             # Lấy input từ Triton
#             text = pb_utils.get_input_tensor_by_name(request, "text_input").as_numpy()[0].decode("utf-8")
#             img_b64 = pb_utils.get_input_tensor_by_name(request, "image_input").as_numpy()[0].decode("utf-8")

#             # Decode ảnh nếu có
#             image = None
#             if img_b64.strip() != "":
#                 image = Image.open(BytesIO(base64.b64decode(img_b64))).convert("RGB").resize((448, 448))

#             # Tạo messages multi-turn
#             messages = self.chat_history.copy()
#             msg_content = [{"type": "text", "text": text}]
#             if image:
#                 msg_content.insert(0, {"type": "image", "image": image})
#             messages.append({"role": "user", "content": msg_content})

#             # Áp dụng prompt template
#             chat_text = self.processor.apply_chat_template(
#                 messages,
#                 tokenize=False,
#                 add_generation_prompt=True
#             )

#             # Process ảnh/video
#             image_inputs, video_inputs = process_vision_info(messages)
#             inputs = self.processor(
#                 text=[chat_text],
#                 images=image_inputs,
#                 videos=video_inputs,
#                 padding=True,
#                 return_tensors="pt"
#             )
#             inputs = {k: v.to(self.device) for k, v in inputs.items()}

#             # Generate output với do_sample=True
#             with torch.inference_mode():
#                 generated_ids = self.model.generate(
#                     **inputs,
#                     max_new_tokens=1024,
#                     temperature=0,
#                     # top_p=1.0,
#                     do_sample=False,
#                     # deterministic=True, 
#                 )
#                 # Trim prompt part
#                 generated_ids_trimmed = [
#                     out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
#                 ]
#                 output_text = self.processor.batch_decode(
#                     generated_ids_trimmed,
#                     skip_special_tokens=True,
#                     clean_up_tokenization_spaces=False
#                 )[0]

#             # Update multi-turn history
#             self.chat_history.append({
#                 "role": "assistant",
#                 "content": [{"type": "text", "text": output_text}]
#             })

#             # Response cho Triton
#             out_tensor = pb_utils.Tensor(
#                 "text_output", np.array([output_text.encode()], dtype=np.bytes_)
#             )
#             responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))
#         print("History =", self.chat_history)

#         return responses
