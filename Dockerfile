FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

ENV PYTHONUNBUFFERED=1 MODEL_DIR=/models/qwen3-4b

WORKDIR /app
COPY requirements.txt /app/requirements.txt
# None of these depend on torch; transformers goes in with --no-deps so pip can never swap the ROCm torch for a CUDA one.
RUN pip install --no-cache-dir -r /app/requirements.txt \
 && pip install --no-cache-dir --no-deps transformers \
 && python3 -c "import torch, transformers; v=torch.__version__; print(v, transformers.__version__); assert '+rocm' in v, v"

RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-4B-Instruct-2507', local_dir='/models/qwen3-4b', allow_patterns=['*.json','*.safetensors','*.txt','*.model','merges.txt','vocab.json'])"

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
COPY app/ /app/
RUN mkdir -p /app/output

CMD ["python3", "/app/server.py"]
