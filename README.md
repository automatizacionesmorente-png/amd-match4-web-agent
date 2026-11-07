# Web Answer Agent — AMD AI League, Match 4

An agent that takes a natural-language question and a website, navigates the site, and returns **the one exact value asked for** (a phone number, a version, a price, a date…) together with the page it came from. Not a summary, not a list of links. If the site does not contain the answer, it returns an empty answer instead of guessing.

Inference runs locally on an AMD GPU with ROCm. No API keys.

## How it works

1. **Resident model.** `server.py` is the container's `CMD`. It listens on a unix socket at once and loads the model in the background; until the model is ready, questions are answered by the rules. On a GPU with room for it, it loads **Qwen3-14B** (non-thinking mode); otherwise, or if that fails, the small **Qwen3-1.7B** (also non-thinking), kept small so the image stays under 60 GiB. At start-up it measures the model's speed, so each question sizes its prompt to the time it has left, and the model runs under a hard deadline with the rules answering if it would be late.
2. **Thin client.** Each question is a fresh `python3 /app/app.py` process. It writes a valid placeholder JSON right away, then asks the server and overwrites it with the result. There is a hard internal deadline below the 30-second per-question limit.
3. **Focused crawl.** It fetches the start page and follows same-site links in parallel, ranked by how well the link text and URL match the question. It also tries the usual paths for the kind of question asked (`/contact`, `/pricing`, `/downloads`…). A site that hangs never holds the answer past the deadline.
4. **Passage ranking.** Pages are cleaned to text (table rows kept), split into overlapping passages and ranked with BM25.
5. **Extraction.** The model reads the top passages and returns `{answer, sources}`. The answer is accepted only if it actually appears in the cited passage, so hallucinated values are dropped.
6. **Fallback.** If the model runs out of time, a rule-based extractor for phones, emails, versions, prices, dates and counts answers from the best passage. It only accepts a value when the question's words appear right next to it.

If no URL is given, it uses any URL in the question, and otherwise a web search, as the starting point. It also accepts a folder of saved pages (`--corpus`); citations are then paths relative to that folder.

## Usage

```bash
python3 /app/app.py --query-id query_01 --query "What is the support phone number?" --url https://example.com
```

Output is written to `/app/output/query_01_output.json`:

```json
{"answer": "+1 555 0100", "citations": ["https://example.com/contact"], "confidence": 0.8,
 "sources": ["https://example.com/contact"], "source_url": "https://example.com/contact"}
```

Accepted argument spellings: `--query`/`--question`, `--query-id`/`--id`, `--url`/`--website`, `--corpus`, `--input` (a JSON file with these fields), `--index`.

## Build

Built on the base image the League mandates, `rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0`. `transformers` is installed with `--no-deps` so pip cannot replace the ROCm torch with a CUDA build. The build fails if that happens.

The included GitHub Actions workflow builds the image, checks that torch is the ROCm build and that the image is under 60 GiB, and runs a smoke test with the real model before pushing. It needs `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` repository secrets.
