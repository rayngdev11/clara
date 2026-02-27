# find /home/tiennv/.conda/envs/medical -name "libpython3.10.so*"
# export LD_LIBRARY_PATH=/home/tiennv/.conda/envs/medical/lib:$LD_LIBRARY_PATH
# echo 'export LD_LIBRARY_PATH=/home/tiennv/.conda/envs/medical/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
# source ~/.bashrc
# ldd $(which python) | grep libpython
# mkdir -p /dev/shm/tmp /dev/shm/egg_cache


# python /home/tiennv/phucth/medical/vllm/src/triton_vllm.py \
# --model THP2903/clara_multiturn

python /home/tiennv/phucth/medical/vllm/src/triton_vllm.py \
  --model THP2903/clara_multiturn \
  --host 0.0.0.0 \
  --port 8000 \
  --verbose