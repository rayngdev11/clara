
# import base64
# from PIL import Image
# import io
# import json
# import logging
# import numpy as np
# import argparse
# from typing import AsyncGenerator, Dict, List
# from pytriton.proxy.types import Request
# from pytriton.triton import Triton, TritonConfig, Tensor, ModelConfig
# from vllm import LLM
# from vllm.sampling_params import SamplingParams

# LOGGER = logging.getLogger("examples.vllm.server")
# from huggingface_hub import HfApi, login
# login(token="")




# llm = None


# async def _generate_for_request(request: Request) -> AsyncGenerator[Dict[str, np.ndarray], None]:
#     """Generate completion for the request."""

#     # user_prompt = request.data.get("prompt")[0].decode('utf-8')
    
#     # Multimodal data is JSON string and it must be decoded into Python objects
#     multi_modal_data = None
#     multi_modal_data_input = request.data.get("multi_modal_data", None) 
#     LOGGER.debug(multi_modal_data_input)
#     if multi_modal_data_input is not None:
#         multi_modal_data_loaded = json.loads(multi_modal_data_input.tolist()[0].decode("utf-8"))

#         # Image field must be decoded using base64 and Pillow
#         image_data_b64 = multi_modal_data_loaded.get("image", None)

#     sampling_params = {name: value.item() for name, value in request.data.items() if name not in ("prompt", "multi_modal_data")}
#     sampling_params = SamplingParams(**sampling_params)

#     # chat template compatible for specific model => here for Qwen/Qwen2-VL-2B-Instruct
#     messages = [{"role": "user", "content": []}]
#     image = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data_b64}"}}
#     messages[0]["content"].append(image)
#     messages[0]["content"].append({"type": "text", "text": user_prompt})

#     outputs = llm.chat(messages, sampling_params)

#     for o in outputs:
#         generated_text = o.outputs[0].text
#         yield {"text": np.char.encode(np.array([generated_text])[None, ...], "utf-8")}
#         return


# async def generate_fn(requests: List[Request]) -> AsyncGenerator[List[Dict[str, np.ndarray]], None]:
#     assert len(requests) == 1, "expected single request because triton batching is disabled"
#     request = requests[0]
#     async for response in _generate_for_request(request):
#         yield [response]  # ensure that the response is a list of responses of len 1, same as requests


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", type=str, default=None)
#     parser.add_argument("--host", type=str, default=None)
#     parser.add_argument("--port", type=int, default=8000)
#     parser.add_argument("--verbose", action="store_true")
#     args = parser.parse_args()

#     log_level = logging.DEBUG if args.verbose else logging.INFO
#     logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")


#     llm = LLM(args.model)

#     with Triton(config=TritonConfig(http_address=args.host, http_port=args.port)) as triton:
#         triton.bind(
#             model_name=args.model.replace("-", "").replace("/", ""),
#             infer_func=generate_fn,
#             inputs=[
#                 Tensor(name="prompt", dtype=bytes, shape=(1,)),
#                 Tensor(name="n", dtype=np.int32, shape=(1,), optional=True),
#                 Tensor(name="best_of", dtype=np.int32, shape=(1,), optional=True),
#                 Tensor(name="temperature", dtype=np.float32, shape=(1,), optional=True),
#                 Tensor(name="top_p", dtype=np.float32, shape=(1,), optional=True),
#                 Tensor(name="max_tokens", dtype=np.int32, shape=(1,), optional=True),
#                 Tensor(name="ignore_eos", dtype=np.bool_, shape=(1,), optional=True),
#                 # Mulimodal data fields is string so you should base string with JSON as argument here
#                 Tensor(name="multi_modal_data", dtype=bytes, shape=(1,), optional=True),
#             ],
#             outputs=[Tensor(name="text", dtype=bytes, shape=(-1, 1))],
#             config=ModelConfig(batching=False, max_batch_size=128, decoupled=True),
#             strict=True,
#         )
#         triton.serve()


import base64
import json
import logging
import numpy as np
import argparse
from typing import AsyncGenerator, Dict, List

from pytriton.proxy.types import Request
from pytriton.triton import Triton, TritonConfig, Tensor, ModelConfig
from vllm import LLM
from vllm.sampling_params import SamplingParams
from huggingface_hub import login
from PIL import Image
import io
from transformers import AutoProcessor
import torch
processor = None


# Đăng nhập nếu model yêu cầu
login(token="")

# Logger
LOGGER = logging.getLogger("vllm.triton_server")

llm = None  # model instance


async def _generate_for_request(request: Request) -> AsyncGenerator[Dict[str, np.ndarray], None]:
    prompt_raw = request.data.get("prompt")[0].decode("utf-8")

    # Default empty values
    image_data_b64 = None
    mm_inputs = None

    # Decode base64 from `multi_modal_data`
    multi_modal_data_input = request.data.get("multi_modal_data", None)
    if multi_modal_data_input is not None:
        multi_modal_data_loaded = json.loads(multi_modal_data_input.tolist()[0].decode("utf-8"))
        image_data_b64 = multi_modal_data_loaded.get("image", None)

    # Prepare image if needed
    if "<image>" in prompt_raw and image_data_b64 is not None:
        # Decode base64 -> PIL -> Resize
        # image_pil = Image.open(io.BytesIO(base64.b64decode(image_data_b64))).convert("RGB").resize((448, 448))

        # # Process image to tensor
        # mm_inputs = processor(image_pil, return_tensors="pt").pixel_values.to("cuda")  # or model.device
        # Giữ lại base64
        image_url = {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_data_b64}"}
        }

    # Sampling parameters
    sampling_params = {name: value.item() for name, value in request.data.items()
                       if name not in ("prompt", "multi_modal_data")}
    sampling_params = SamplingParams(**sampling_params)

    # Convert prompt to messages
    if "<image>" in prompt_raw and image_data_b64 is not None:
        # text_parts = prompt_raw.split("<image>")
        # messages = [{"role": "user", "content": []}]
        # if text_parts[0].strip():
        #     messages[0]["content"].append({"type": "text", "text": text_parts[0].strip()})
        # messages[0]["content"].append({"type": "image", "image": image_pil})
        # if len(text_parts) > 1 and text_parts[1].strip():
        #     messages[0]["content"].append({"type": "text", "text": text_parts[1].strip()})
        text_parts = prompt_raw.split("<image>")
        messages = [{"role": "user", "content": []}]
        if text_parts[0].strip():
            messages[0]["content"].append({"type": "text", "text": text_parts[0].strip()})
        messages[0]["content"].append(image_url)
        if len(text_parts) > 1 and text_parts[1].strip():
            messages[0]["content"].append({"type": "text", "text": text_parts[1].strip()})

    else:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt_raw}]}]

    LOGGER.debug(f"Messages: {messages}")

    # outputs = llm.chat(messages, sampling_params, mm_inputs=[mm_inputs] if mm_inputs is not None else None)
    outputs = llm.chat(messages, sampling_params)

    for o in outputs:
        generated_text = o.outputs[0].text
        yield {
            "text": np.char.encode(np.array([generated_text])[None, ...], "utf-8")
        }
        return



async def generate_fn(requests: List[Request]) -> AsyncGenerator[List[Dict[str, np.ndarray]], None]:
    assert len(requests) == 1, "Batching disabled. Expected a single request."
    request = requests[0]
    async for response in _generate_for_request(request):
        yield [response]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s: %(message)s"
    )

    # Load model
    llm = LLM(model=args.model)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model, min_pixels=256*28*28, max_pixels=1280*28*28)

    # Start Triton server
    with Triton(config=TritonConfig(http_address=args.host, http_port=args.port)) as triton:
        triton.bind(
            model_name=args.model.replace("-", "").replace("/", ""),
            infer_func=generate_fn,
            inputs=[
                Tensor(name="prompt", dtype=bytes, shape=(1,)),
                Tensor(name="n", dtype=np.int32, shape=(1,), optional=True),
                Tensor(name="best_of", dtype=np.int32, shape=(1,), optional=True),
                Tensor(name="temperature", dtype=np.float32, shape=(1,), optional=True),
                Tensor(name="top_p", dtype=np.float32, shape=(1,), optional=True),
                Tensor(name="max_tokens", dtype=np.int32, shape=(1,), optional=True),
                Tensor(name="ignore_eos", dtype=np.bool_, shape=(1,), optional=True),
                Tensor(name="multi_modal_data", dtype=bytes, shape=(1,), optional=True),
            ],
            outputs=[Tensor(name="text", dtype=bytes, shape=(-1, 1))],
            config=ModelConfig(batching=False, max_batch_size=128, decoupled=True),
            strict=True,
        )
        triton.serve()
