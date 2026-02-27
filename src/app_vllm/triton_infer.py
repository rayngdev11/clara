import numpy as np
import tritonclient.grpc as grpcclient
import io

TRITON_URL = "localhost:8001"  # GRPC port
MODEL_NAME = "THP2903clara_multiturn"
MODEL_VERSION = ""

def preprocess_image(image):
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    return buf.getvalue()

def infer_from_triton(image, sex, age, view):
    client = grpcclient.InferenceServerClient(url=TRITON_URL)

    image_bytes = preprocess_image(image)

    inputs = [
        grpcclient.InferInput("image", [1], "BYTES"),
        grpcclient.InferInput("sex", [1], "BYTES"),
        grpcclient.InferInput("age", [1], "BYTES"),
        grpcclient.InferInput("view", [1], "BYTES"),
    ]
    inputs[0].set_data_from_numpy(np.array([image_bytes], dtype=object))
    inputs[1].set_data_from_numpy(np.array([sex], dtype=object))
    inputs[2].set_data_from_numpy(np.array([age], dtype=object))
    inputs[3].set_data_from_numpy(np.array([view], dtype=object))

    outputs = [
        grpcclient.InferRequestedOutput("response")
    ]

    results = client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)
    output_text = results.as_numpy("response")[0].decode("utf-8")
    return output_text
