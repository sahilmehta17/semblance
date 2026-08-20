#!/usr/bin/env python3
"""One-time extraction of a vCache benchmark parquet into compact JSONL.

Emits one record per row:

    {"prompt": <str>, "id_set": <int>, "embedding": [<float>, ...]}

using the precomputed `emb_gte` vector (1024-dim). The Go benchmark (bench/)
replays these committed vectors directly into the cache, so extraction is the
only place the embeddings are read — the gateway itself never sees them.

This script is offline and run once; its large outputs are gitignored. See the
sibling README.md.

Two modes, either or both per invocation:
  --check       run the label-integrity check (distinct responses per
                (dataset_name, ID_Set) group) and print/save a verdict.
  --out PATH    extract JSONL to PATH.

pandas/pyarrow/huggingface_hub are required (see requirements.txt). The earlier
"no compiled Python libs" idea does not apply to this offline tool.
"""

import argparse
import json
import sys
from collections import defaultdict


def resolve_path(repo: str, file: str, local: str | None) -> str:
    """Return a local parquet path, downloading from HuggingFace if needed."""
    if local:
        return local
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, file, repo_type="dataset")


def parse_embedding(raw) -> list[float]:
    """emb_gte is stored as a JSON-ish string '[...]'; some datasets may store a
    native list. Handle both."""
    if isinstance(raw, (list, tuple)):
        return [float(x) for x in raw]
    return [float(x) for x in json.loads(raw)]


def run_check(
    path: str,
    report_out: str | None,
    dataset_col: str,
    idset_col: str,
    response_col: str,
) -> tuple[bool, str]:
    """Group by (dataset_name, ID_Set) and count distinct response_llama_3_8b.

    Returns (labels_reliable, report_text). Labels are reliable only if EVERY
    group maps to a single distinct response — i.e. ID_Set really is an
    equivalence class of answers. If any group spans multiple answers, treating
    ID_Set as a correctness label would inflate accuracy.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    available = set(pf.schema_arrow.names)
    # dataset_name is optional (some sets are single-source); fall back to a
    # constant so grouping is by ID_Set alone.
    has_dataset = dataset_col in available
    cols = [idset_col, response_col] + ([dataset_col] if has_dataset else [])
    distinct: dict[tuple[str, int], set[str]] = defaultdict(set)
    total_rows = 0
    for batch in pf.iter_batches(columns=cols, batch_size=8192):
        d = batch.to_pydict()
        datasets = d[dataset_col] if has_dataset else ["_"] * len(d[idset_col])
        for ds, idset, resp in zip(datasets, d[idset_col], d[response_col]):
            total_rows += 1
            distinct[(ds, int(idset))].add(resp)

    groups = len(distinct)
    impure = {k: len(v) for k, v in distinct.items() if len(v) > 1}
    neg_impure = {k: n for k, n in impure.items() if k[1] < 0}

    lines = []
    lines.append(f"rows scanned          : {total_rows}")
    lines.append(f"(dataset, ID_Set) groups: {groups}")
    lines.append(f"impure groups (>1 distinct response): {len(impure)}")
    lines.append(f"  of which negative ID_Set          : {len(neg_impure)}")

    # Show the worst offenders for transparency.
    worst = sorted(impure.items(), key=lambda kv: kv[1], reverse=True)[:10]
    if worst:
        lines.append("worst impure groups (dataset, ID_Set) -> distinct responses:")
        for (ds, idset), n in worst:
            lines.append(f"  ({ds!r}, {idset}) -> {n}")

    reliable = len(impure) == 0
    if reliable:
        lines.append("")
        lines.append("VERDICT: labels RELIABLE — every (dataset, ID_Set) maps to one answer.")
    else:
        lines.append("")
        lines.append(
            "VERDICT: labels UNRELIABLE — some ID_Set groups span multiple distinct "
            "answers, so ID_Set is NOT a clean correctness label. Use DEV-LOOP ONLY; "
            "do not use for any published number."
        )

    report = "\n".join(lines)
    print(report)
    if report_out:
        with open(report_out, "w") as f:
            f.write(report + "\n")
        print(f"\n(report written to {report_out})")
    return reliable, report


def run_structure(
    path: str,
    idset_col: str,
    prompt_col: str,
    response_col: str,
    report_out: str | None,
) -> str:
    """Validate id_set CLUSTER STRUCTURE, not response purity.

    For datasets whose responses are a constant placeholder (SearchQueries), the
    Combo-style "distinct responses per group" check is meaningless — every group
    is trivially pure. What matters instead is whether id_set is a usable
    clustering: are responses actually constant (so scoring must rely on id_set),
    and does id_set contain real multi-member clusters (paraphrase pairs to hit)
    rather than all singletons?
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    resp_counts: dict[str, int] = defaultdict(int)
    members: dict[int, list[str]] = defaultdict(list)
    total = 0
    for batch in pf.iter_batches(columns=[idset_col, prompt_col, response_col], batch_size=8192):
        d = batch.to_pydict()
        for idset, prompt, resp in zip(d[idset_col], d[prompt_col], d[response_col]):
            total += 1
            resp_counts[resp] += 1
            members[int(idset)].append(prompt)

    sizes = [len(v) for v in members.values()]
    groups = len(sizes)
    singletons = sum(1 for s in sizes if s == 1)
    multi = sum(1 for s in sizes if s >= 2)
    max_size = max(sizes) if sizes else 0
    pairs_or_more_rows = sum(s for s in sizes if s >= 2)

    lines = []
    lines.append(f"rows scanned            : {total}")
    lines.append("")
    lines.append("-- response constancy (are responses placeholders?) --")
    lines.append(f"distinct response values: {len(resp_counts)}")
    for r, n in sorted(resp_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]:
        lines.append(f"  {n:>7} x {r[:70]!r}")
    responses_constant = len(resp_counts) == 1

    lines.append("")
    lines.append("-- id_set cluster structure --")
    lines.append(f"id_set groups           : {groups}")
    lines.append(f"  singletons (size 1)   : {singletons}")
    lines.append(f"  multi-member (size>=2): {multi}")
    lines.append(f"  rows in multi clusters: {pairs_or_more_rows}")
    lines.append(f"  largest cluster size  : {max_size}")
    # coarse histogram
    buckets = {"1": 0, "2": 0, "3-5": 0, "6-10": 0, "11+": 0}
    for s in sizes:
        if s == 1:
            buckets["1"] += 1
        elif s == 2:
            buckets["2"] += 1
        elif s <= 5:
            buckets["3-5"] += 1
        elif s <= 10:
            buckets["6-10"] += 1
        else:
            buckets["11+"] += 1
    lines.append("  size histogram        : " + ", ".join(f"{k}:{v}" for k, v in buckets.items()))

    lines.append("")
    lines.append("-- spot check: same id_set should be paraphrases --")
    shown = 0
    for idset, prompts in members.items():
        if len(prompts) >= 2:
            lines.append(f"  id_set {idset} ({len(prompts)} members):")
            for p in prompts[:3]:
                lines.append(f"      - {p[:90]}")
            shown += 1
            if shown >= 3:
                break

    lines.append("")
    lines.append("-- spot check: different id_set should differ --")
    distinct_ids = list(members.keys())[:3]
    for idset in distinct_ids:
        lines.append(f"  id_set {idset}: {members[idset][0][:90]}")

    lines.append("")
    verdict = []
    verdict.append(
        "responses ARE a constant placeholder — scoring MUST rely on id_set, not response text."
        if responses_constant
        else f"responses are NOT constant ({len(resp_counts)} distinct) — revisit assumptions."
    )
    verdict.append(
        f"id_set is a usable clustering: {multi} multi-member clusters give real paraphrase pairs to hit."
        if multi > 0
        else "id_set has NO multi-member clusters — no paraphrase pairs; unusable."
    )
    lines.append("VERDICT: " + " ".join(verdict))

    report = "\n".join(lines)
    print(report)
    if report_out:
        with open(report_out, "w") as f:
            f.write(report + "\n")
        print(f"\n(report written to {report_out})")
    return report


def run_extract(
    path: str,
    out: str,
    prompt_col: str,
    idset_col: str,
    emb_col: str,
    emb_dim: int,
    limit: int | None,
) -> int:
    """Stream the parquet and write compact JSONL. Returns rows written."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    written = 0
    with open(out, "w") as fo:
        for batch in pf.iter_batches(columns=[prompt_col, idset_col, emb_col], batch_size=2048):
            d = batch.to_pydict()
            for prompt, idset, emb in zip(d[prompt_col], d[idset_col], d[emb_col]):
                vec = parse_embedding(emb)
                if emb_dim and len(vec) != emb_dim:
                    sys.exit(f"FATAL: embedding dim {len(vec)} != expected {emb_dim} (row {written})")
                rec = {"prompt": prompt, "id_set": int(idset), "embedding": vec}
                fo.write(json.dumps(rec, separators=(",", ":")))
                fo.write("\n")
                written += 1
                if limit and written >= limit:
                    return written
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="vCache/SemBenchmarkCombo", help="HuggingFace dataset repo id")
    ap.add_argument("--file", default="train.parquet", help="parquet file within the repo")
    ap.add_argument("--local", default=None, help="use a local parquet path instead of downloading")
    ap.add_argument("--check", action="store_true", help="run the label-integrity check")
    ap.add_argument("--check-out", default=None, help="write the check report to this file")
    ap.add_argument("--out", default=None, help="extract JSONL to this path")
    ap.add_argument("--limit", type=int, default=None, help="stop after N rows (for samples)")
    ap.add_argument("--emb-col", default="emb_gte", help="embedding column name")
    ap.add_argument("--emb-dim", type=int, default=1024, help="expected embedding dimension (0 = skip check)")
    ap.add_argument("--prompt-col", default="prompt", help="prompt column name")
    ap.add_argument("--idset-col", default="ID_Set", help="id-set label column name")
    ap.add_argument("--dataset-col", default="dataset_name", help="dataset-name column (optional)")
    ap.add_argument("--response-col", default="response_llama_3_8b", help="response column for the label check")
    ap.add_argument("--schema", action="store_true", help="print the parquet schema and exit")
    ap.add_argument("--structure", action="store_true", help="analyze id_set cluster structure (for placeholder-response sets)")
    args = ap.parse_args()

    if not args.check and not args.out and not args.schema and not args.structure:
        ap.error("nothing to do: pass --schema, --check, --structure, and/or --out")

    path = resolve_path(args.repo, args.file, args.local)
    print(f"parquet: {path}\n")

    if args.schema:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        print("rows:", pf.metadata.num_rows)
        for c in pf.schema_arrow.names:
            print("   ", c, "->", pf.schema_arrow.field(c).type)
        return

    if args.structure:
        run_structure(path, args.idset_col, args.prompt_col, args.response_col, args.check_out)
        print()

    if args.check:
        run_check(path, args.check_out, args.dataset_col, args.idset_col, args.response_col)
        print()

    if args.out:
        n = run_extract(path, args.out, args.prompt_col, args.idset_col, args.emb_col, args.emb_dim, args.limit)
        print(f"extracted {n} rows -> {args.out}")


if __name__ == "__main__":
    main()
