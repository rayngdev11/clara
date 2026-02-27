# # CUDA_VISIBLE_DEVICES=0 python gradio_app.py
# #!/bin/bash

# export CUDA_VISIBLE_DEVICES=0

export TMPDIR=/home/clara/tmp
mkdir -p /home/clara/tmp
# Kiểm tra nếu người dùng không truyền GPU ID vào
if [ -z "$1" ]; then
    echo "❌ Vui lòng truyền số GPU vào. Ví dụ: ./app.sh 1 hoặc ./app.sh 0,1"
    exit 1
fi

# GPU ID do người dùng nhập
GPU_IDS=$1

# Chạy Gradio app với GPU được chọn
CUDA_VISIBLE_DEVICES=$GPU_IDS python gradio_app.py

# CUDA_VISIBLE_DEVICES=0 python gradio_app.py
#!/bin/bash

# export TMPDIR=/home/datnvt/tmp
# mkdir -p "$TMPDIR"

# # Kiểm tra GPU ID
# if [ -z "$1" ]; then
#     echo "❌ Vui lòng truyền số GPU vào. Ví dụ: ./app.sh 1 hoặc ./app.sh 0,1"
#     exit 1
# fi

# GPU_IDS=$1

# # Clear PyTorch CUDA cache trước khi chạy app
# echo "🧹 Clearing PyTorch CUDA cache..."
# python -c "
# import torch
# if torch.cuda.is_available():
#     torch.cuda.empty_cache()
#     torch.cuda.reset_peak_memory_stats()
# print('✅ Cache cleared')
# "

# # Tùy chọn: nâng giới hạn số lượng file mở (nếu lỗi liên quan đến `.so`)
# ulimit -n 65535

# # Tùy chọn: preload thư viện nếu cần
# # export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so

# # Chạy Gradio app
# echo "🚀 Running Gradio app on GPU $GPU_IDS"
# CUDA_VISIBLE_DEVICES=$GPU_IDS python gradio_app.py
