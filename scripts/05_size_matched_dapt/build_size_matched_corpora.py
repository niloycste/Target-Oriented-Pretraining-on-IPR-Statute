"""Build the three size-matched DAPT corpora for the scale-vs-content ablation.

The authentic pool contains only 901 distinct training paragraphs, so 901 is the
largest size at which authentic and synthetic corpora can be matched without
upsampling. All three corpora therefore hold exactly 901 records:

  auth901   901 distinct authentic training paragraphs
  synth901  901 passages sampled from the 13,211 accepted synthetic pool
  mixed901  450 authentic + 451 synthetic

Sampling uses seed 42. Output schema matches the existing DAPT corpora ({"context": ...}).
"""
import json
import random
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
MERGE = BASE / "data" / "dapt"
SEED = 42
N = 901

norm = lambda t: " ".join(t.strip().lower().split())


def load(path, key="context"):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write(path, texts):
    with open(path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"context": t}, ensure_ascii=False) + "\n")
    print(f"  wrote {len(texts):5d} -> {path.name}")


def main():
    rng = random.Random(SEED)

    # authentic: filter the merged pilot corpus for source == original
    merged = load(MERGE / "pretraining_contexts.jsonl")
    auth_records = [r for r in merged if r.get("source", "") == "original"]
    print(f"authentic records in pretraining_contexts.jsonl : {len(auth_records)}")

    seen, auth_unique = set(), []
    for r in auth_records:
        t = r["context"]
        k = norm(t)
        if k not in seen:
            seen.add(k)
            auth_unique.append(t)
    print(f"distinct authentic paragraphs                   : {len(auth_unique)}")
    assert len(auth_unique) == N, f"expected {N} distinct authentic, got {len(auth_unique)}"

    synth_all = [r["context"] for r in load(MERGE / "pretraining_contexts_synth_only.jsonl")]
    print(f"synthetic pool                                  : {len(synth_all)}")

    # exclude any synthetic passage that duplicates an authentic one
    auth_keys = {norm(t) for t in auth_unique}
    synth_pool = [t for t in synth_all if norm(t) not in auth_keys]
    print(f"synthetic pool after removing authentic dupes    : {len(synth_pool)}")

    print("\nbuilding corpora:")
    write(MERGE / "pretraining_contexts_auth901.jsonl", auth_unique)

    synth901 = rng.sample(synth_pool, N)
    write(MERGE / "pretraining_contexts_synth901.jsonl", synth901)

    n_auth = N // 2                      # 450
    n_syn = N - n_auth                   # 451
    mixed = rng.sample(auth_unique, n_auth) + rng.sample(synth_pool, n_syn)
    rng.shuffle(mixed)
    write(MERGE / "pretraining_contexts_mixed901.jsonl", mixed)
    print(f"  (mixed901 = {n_auth} authentic + {n_syn} synthetic)")

    for name in ("auth901", "synth901", "mixed901"):
        p = MERGE / f"pretraining_contexts_{name}.jsonl"
        rows = load(p)
        uniq = len({norm(r["context"]) for r in rows})
        print(f"check {name:9s} records {len(rows):4d}  distinct {uniq:4d}")


if __name__ == "__main__":
    main()
