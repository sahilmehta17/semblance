# tools/prep — one-time benchmark extraction

Converts a vCache HuggingFace benchmark parquet into compact JSONL that the Go
benchmark (`bench/`) replays. One record per row:

```json
{"prompt": "...", "id_set": 12, "embedding": [0.01, -0.02, ...]}
```

The `embedding` is the dataset's precomputed **`emb_gte`** vector (1024-dim), so
the benchmark never calls a live embedder — results are deterministic and
reproducible from committed vectors.

This is offline tooling (pandas/pyarrow allowed) and runs once. Large outputs
are gitignored; only the script, the label-check report, and a tiny JSONL sample
are committed.

## Setup (no sudo; venv bootstrapped via get-pip)

```bash
python3 -m venv --without-pip .venv
curl -sS https://bootstrap.pypa.io/get-pip.py | .venv/bin/python
.venv/bin/python -m pip install -r requirements.txt
```

## Usage

Label-integrity check + verdict:

```bash
.venv/bin/python extract.py --repo vCache/SemBenchmarkCombo --check --check-out label_check_combo.txt
```

Extract to JSONL:

```bash
.venv/bin/python extract.py --repo vCache/SemBenchmarkSearchQueries --out out/searchqueries.jsonl
```

Tiny committed sample:

```bash
.venv/bin/python extract.py --repo vCache/SemBenchmarkCombo --out samples/sample_combo.jsonl --limit 3
```

## Datasets

- **SemBenchmarkCombo** (436 MB) — dev-loop validation. Its `ID_Set` labels are
  checked for integrity before any use (see `label_check_combo.txt`).
- **SemBenchmarkSearchQueries** (2.4 GB) — primary benchmark, clean union-find
  `id_set` labels.

The label check groups by `(dataset_name, ID_Set)` and counts distinct
`response_llama_3_8b`. If any group spans more than one answer, `ID_Set` is not a
clean correctness label and that dataset is dev-loop-only.
