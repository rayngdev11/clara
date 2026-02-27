# check_xray.py

from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import torchxrayvision as xrv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")  # Chạy trên CPU nếu không có GPU
# Load segmentation model 1 lần duy nhất
seg_model = xrv.baseline_models.chestx_det.PSPNet().to(device)
seg_model.eval()

lung_label_map = {'Left Lung': 5, 'Right Lung': 6}

def is_chest_xray(image: Image.Image, lung_threshold_ratio=0.01):
    img = np.array(image.convert("L")).astype(np.float32)
    img = xrv.datasets.normalize(img, np.max(img))
    img = img[None, ...]

    transform = transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(512)
    ])
    img = transform(img)
    img = torch.from_numpy(img).float().to(device)

    with torch.no_grad():
        output = seg_model(img[None, ...])[0].cpu().numpy()

    left = output[lung_label_map['Left Lung']] > 0.5
    right = output[lung_label_map['Right Lung']] > 0.5

    lung_pixels = np.sum(left) + np.sum(right)
    total_pixels = output.shape[1] * output.shape[2]
    ratio = lung_pixels / total_pixels

    return ratio > lung_threshold_ratio
