FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

ENV PYTHONUNBUFFERED=1 MODEL_DIR=/models/qwen3-4b PIP_BREAK_SYSTEM_PACKAGES=1

WORKDIR /app
COPY requirements.txt /app/requirements.txt
# Nothing in requirements.txt depends on torch; transformers goes in with --no-deps so pip can never swap the ROCm torch for a CUDA one.
RUN pip install --no-cache-dir -r /app/requirements.txt \
 && pip install --no-cache-dir --no-deps "transformers>=4.56,<5" \
 && python3 -c "import torch, numpy; from transformers import AutoModelForCausalLM, AutoTokenizer; import transformers; v=torch.__version__; print(v, transformers.__version__); assert '+rocm' in v, v"

RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-4B-Instruct-2507', local_dir='/models/qwen3-4b', allow_patterns=['*.json','*.safetensors','*.txt'])" \
 && rm -rf /models/qwen3-4b/.cache /root/.cache \
 && python3 -c "from transformers import AutoConfig, AutoTokenizer; AutoTokenizer.from_pretrained('/models/qwen3-4b'); print(AutoConfig.from_pretrained('/models/qwen3-4b').model_type)"

# The bigger model the server picks on a GPU with room for it; the 4B above stays as the fallback.
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-14B', local_dir='/models/qwen3-14b', allow_patterns=['*.json','*.safetensors','*.txt'])" \
 && rm -rf /models/qwen3-14b/.cache /root/.cache \
 && python3 -c "from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('/models/qwen3-14b'); s=t.apply_chat_template([{'role':'user','content':'hi'}], tokenize=False, add_generation_prompt=True, enable_thinking=False); assert '<think>' in s and '</think>' in s, s; print('14B template OK')"

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
COPY app/ /app/
RUN mkdir -p /app/output

CMD ["python3", "/app/server.py"]
