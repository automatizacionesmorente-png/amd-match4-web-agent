FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

ARG MODEL_REPO=Qwen/Qwen2.5-VL-7B-Instruct
ENV PYTHONUNBUFFERED=1 MODEL_DIR=/models/vlm PIP_BREAK_SYSTEM_PACKAGES=1 MC3_INDEX_DIR=/app/index

WORKDIR /app
COPY requirements.txt /app/requirements.txt
# Nothing in requirements.txt depends on torch; transformers goes in with --no-deps so pip can never swap the ROCm torch for a CUDA one.
RUN pip install --no-cache-dir -r /app/requirements.txt \
 && pip install --no-cache-dir --no-deps "transformers>=4.56,<5" \
 && python3 -c "import torch, numpy, pypdf, openpyxl, PIL; from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration; from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor; import transformers; v=torch.__version__; print(v, transformers.__version__); assert '+rocm' in v, v"

RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL_REPO}', local_dir='/models/vlm', allow_patterns=['*.json','*.safetensors','*.txt'])" \
 && python3 -c "from transformers import AutoConfig, AutoTokenizer; from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor; AutoTokenizer.from_pretrained('/models/vlm'); Qwen2VLImageProcessor.from_pretrained('/models/vlm'); print(AutoConfig.from_pretrained('/models/vlm').model_type)"

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
COPY app/ /app/
RUN mkdir -p /app/corpus /app/output /app/index

# The model loads while the harness copies the corpus in; --index and every --query talk to it over a unix socket.
CMD ["python3", "/app/server.py"]
