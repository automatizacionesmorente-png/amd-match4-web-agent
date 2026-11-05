# Cited Document RAG — AMD AI League, Match 3

Answers questions over a folder of mixed company documents (PDF, DOCX, XLSX, CSV, logs, source code, PNG/JPG) with **the value only** and **exactly the files that answer needs**. If the corpus does not contain the answer, it returns an empty answer and no citations instead of guessing.

Everything runs locally on an AMD GPU with ROCm using **Qwen2.5-VL-7B-Instruct**: one vision-language model reads both the text and the images. No network at run time.

## How it works

**Index (`--index`, once).**
- Every file is parsed in its own `try`, so one bad file never costs the rest of the corpus. Unreadable, encrypted and unknown-type files are recorded and skipped; an encrypted file is never a source.
- PDF via pypdf, with embedded images pulled from pages that have no text layer. DOCX and PPTX are read straight from their XML, keeping tables. XLSX via openpyxl and CSV via the csv module: every row is written with its column headers ("part_number: … | compatible_with: TQ-40") so a row is meaningful on its own.
- Images are transcribed by the model into label/value pairs ("B14: THERM_ALERT#"), which makes them searchable.
- Superseded documents are labelled: files withdrawn by their own title or filename, and older revisions in an r1/r2 family.
- The index is persisted to `/app/index`, and the model stays resident behind a unix socket, so each question costs only retrieval plus one generation.

**Query (`--query`, per question).**
- BM25 over chunks, with identifiers like `ORR-1847` or `THERM_ALERT#` kept whole.
- A second hop follows identifiers found next to the question's terms to the files that hold their values. This is how "the log shows an incident → which release fixed it" reaches the bug database.
- The model gets the top files (marked SUPERSEDED where they are), plus up to two relevant images attached directly, and must answer in JSON with the value, the files it needed, and the evidence line.
- **Citations by necessity.** Citations are kept only for files that actually contain the answer, preferring current documents over superseded ones. A second file is added only when it supplies the identifier that leads to the answer (true multi-hop). An answer that appears in no retrieved file is refused.

## Usage

```bash
python3 /app/app.py --index /app/corpus
python3 /app/app.py --corpus /app/corpus --query-id query_01 --query "What is the maximum junction temperature of the TQ-40?"
# -> /app/output/query_01_output.json  {"answer": "94", "citations": ["specs/tq40_datasheet_r2.pdf"], "confidence": 0.8}
```

## Build

Built on `rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0`, with `transformers` installed `--no-deps` so the ROCm torch is never replaced. The workflow builds the image, checks the ROCm torch and the 60 GiB limit, then runs a contract test with `--network none` and `--cap-drop DAC_OVERRIDE` over a corpus containing an unreadable file, an unknown binary and an empty directory, and pushes. It needs the `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` secrets.
