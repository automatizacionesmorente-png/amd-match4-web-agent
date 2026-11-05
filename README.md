# Plate & Sign OCR — AMD AI League, Match 2

Reads licence plates and road signs and returns only the text the grader expects: the registration number without the state banner or slogan (`CALIFORNIA 7ABC123` → `7ABC123`), Chinese plates with their province prefix kept (`京A·12345`), and multi-line signs joined top to bottom (`SPEED LIMIT 65`, `ROAD WORK AHEAD`).

Inference runs locally on an AMD GPU with ROCm using **Qwen2.5-VL-7B-Instruct**. No API keys, no network needed at run time.

## How it works

1. **Resident model.** `server.py` is the container's `CMD`. It loads the model onto the GPU once and listens on a unix socket. If a harness overrides `CMD`, the first `app.py` call starts the server itself.
2. **Thin client.** Each image is a fresh `python3 /app/app.py --input-image …` process. It writes a valid JSON file first, then overwrites it with the result, all within the 30-second budget.
3. **Preprocessing.** PNG, JPEG and TIFF (including multi-page TIFF) are opened with PIL and converted to RGB. Small crops are upscaled. Very large photos are capped at the vision encoder's pixel budget so they never blow the time limit.
4. **One prompt, structured output.** The model says whether it sees a plate or a sign and returns the text as JSON.
5. **Post-processing.** On plates, a list of US state names and slogans is stripped, along with any line that carries no digits. Sign text is never trimmed. Labels like `Text:` and code fences are removed.

The prompt is built by hand rather than through `AutoProcessor`, because the processor also builds a video processor that needs torchvision, which the mandated base image does not ship.

## Usage

```bash
python3 /app/app.py --input-image /app/input/image_01.png
# -> /app/output/image_01_output.json  {"text": "7ABC123", "confidence": 0.9}
```

## Build

Built on `rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0`. `transformers` is installed with `--no-deps` so pip cannot replace the ROCm torch, and the build fails if it ever does. The GitHub Actions workflow builds the image, checks torch is the ROCm build and that the image is under 60 GiB, runs a no-network contract test on PNG/JPEG/TIFF, and pushes. It needs the `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` secrets.
