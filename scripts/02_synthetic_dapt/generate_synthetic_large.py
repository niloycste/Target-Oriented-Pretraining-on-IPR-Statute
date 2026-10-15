"""
Large-Scale Synthetic Context Generation — Target: 8,000–10,000 contexts.

Model: Gemma 3:12b via Ollama (primary). Change MODEL_NAME to "qwen2.5:7b" if RAM issues.

Math:
  ~700 seeds × (8 solo styles + 6 broad-supported) = ~9,800 planned
  At ~80% acceptance → ~7,800–8,500 accepted

Key improvements over previous synthetic_context.py:
  - Relaxed validation: allows section numbers, Act year refs, subsection (1) — all valid legal text
  - 8 style variants (was 5)
  - All seeds used (was MAX_PRIMARY_SEEDS = 20 or None but skipped)
  - More broad supports per seed (6 vs 1-2)
  - Resume-safe: skips already-generated IDs
  - Progress prints accepted count running total

Steps:
  1. Run extract_seeds_from_train.py first → synthetic context/seed_pool_large.json
  2. Run this script                       → synthetic context/synthetic_contexts_large.jsonl
  3. Run merge_and_prepare_pretraining_v2.py to combine with original contexts
  4. Re-run pretrain_mlm.py (update PRETRAINING_FILE to v2)

Run: python generate_synthetic_large.py
"""

import json
import random
import re
import time
from collections import Counter
from pathlib import Path

import ollama

BASE_DIR      = Path(__file__).resolve().parents[2]
SYNTHETIC_DIR = BASE_DIR / "data" / "synthetic"
SEED_FILE = SYNTHETIC_DIR / "seed_pool_large.json"
OUTPUT_ACCEPTED = SYNTHETIC_DIR / "synthetic_contexts_large.jsonl"
OUTPUT_REJECTED = SYNTHETIC_DIR / "synthetic_contexts_large_rejected.jsonl"

# ── Model ──────────────────────────────────────────────────────────────────────
# Gemma 3:12b gives better quality legal text; Qwen 2.5:7b is faster / lower RAM
# MODEL_NAME = "gemma3:27b"
MODEL_NAME = "gemma3:12b"
# MODEL_NAME = "qwen2.5:7b"

# ── Volume settings ────────────────────────────────────────────────────────────
# N_STYLES_SOLO  = 8 → one generation per each of 8 style variants (solo, no support seed)
# N_BROAD        = 6 → 6 different broad-support seeds per primary seed
# Total per seed = 14 → 700 seeds × 14 = 9,800 planned → ~7,800 accepted at 80%
N_STYLES_SOLO   = 8
N_BROAD         = 6
MAX_RETRIES     = 2     # retries per slot if validation fails
MAX_SEEDS       = 1500  # use top-N seeds by frequency (highest quality)
                        # 1000 × 14 = 14,000 planned → ~11,200 accepted at 80%
                        # Time on gemma3:27b CPU (~4 tok/s): ~8-9 days
                        # Time on gemma3:12b CPU (~9 tok/s): ~3-4 days
MIN_SEED_COUNT  = 3     # skip seeds seen fewer than 3 times — filters noise/KeyBERT junk

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ── Ollama generation parameters ───────────────────────────────────────────────
TEMPERATURE    = 0.40    # slightly higher than 0.35 for variety across styles
TOP_P          = 0.85
REPEAT_PENALTY = 1.12
NUM_CTX        = 4096
NUM_PREDICT    = 250
SLEEP_BETWEEN  = 0.1     # seconds between calls (Ollama is local, keep low)

# ── 8 style variants with length hints for variety ─────────────────────────────
# Each entry: (style_name, length_hint)
# Length hint nudges the model toward shorter or longer passages per style —
# better alignment with original data where some provisions are brief, others long.
STYLE_VARIANTS = [
    ("definition-oriented",               "100 to 140 words"),
    ("registration-oriented",             "120 to 170 words"),
    ("rights-and-remedies-oriented",      "130 to 180 words"),
    ("authority-and-procedure-oriented",  "140 to 190 words"),
    ("conditions-and-compliance-oriented","120 to 170 words"),
    ("penalty-and-enforcement-oriented",  "110 to 160 words"),
    ("licensing-and-assignment-oriented", "130 to 180 words"),
    ("dispute-and-appeal-oriented",       "120 to 170 words"),
]

ACT_NAME_MAP = {
    "copyright_act":                    "The Copyright Act, 1957",
    "patent_act":                       "The Patents Act, 1970",
    "trademark_act":                    "The Trade Marks Act, 1999",
    "design_act":                       "The Designs Act, 2000",
    "gi_act":                           "The Geographical Indications of Goods (Registration and Protection) Act, 1999",
    "semiconductor_layout_design_act":  "The Semiconductor Integrated Circuits Layout-Design Act, 2000",
    "plant_varieties_farmers_rights_act": "The Protection of Plant Variety and Farmers' Rights Act, 2001",
}


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_text(text):
    return " ".join(str(text).split()).strip()


def act_display(act_code):
    return ACT_NAME_MAP.get(act_code, act_code.replace("_", " ").title())


def safe_id(text):
    text = re.sub(r"[^\w\-]", "_", str(text))
    return text[:160]


def existing_ids(path):
    ids = set()
    p = Path(path)
    if not p.exists():
        return ids
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                ids.add(row["synthetic_id"])
            except Exception:
                pass
    return ids


def is_clean_support_entity(entity: str) -> bool:
    """
    Filter out noisy/fragmented seeds from being used as broad support entities.
    Rejects:
      - More than 4 words (likely a sentence fragment, not a legal concept)
      - Contains digits mixed with words (e.g. "section 31 application")
      - Looks like a phrase fragment (starts/ends with stop words)
    """
    e = entity.strip().lower()
    words = e.split()
    if len(words) > 4:
        return False
    # Starts or ends with function words that indicate a fragment
    bad_edges = {"the", "a", "an", "of", "in", "to", "and", "or", "for",
                 "by", "on", "at", "is", "are", "means", "act", "that"}
    if words[0] in bad_edges or words[-1] in bad_edges:
        return False
    return True


def choose_broad_seeds(primary_seed, all_seeds, k):
    """Pick k diverse broad-support seeds. Prefer same act; fall back to others."""
    primary_entity = primary_seed["entity"].lower()
    primary_act    = primary_seed["primary_act"]

    same_act = [
        s for s in all_seeds
        if s["entity"].lower() != primary_entity
        and s["primary_act"] == primary_act
        and s["count"] >= 5                  # stricter than solo (was 2)
        and is_clean_support_entity(s["entity"])
    ]
    other_act = [
        s for s in all_seeds
        if s["entity"].lower() != primary_entity
        and s["primary_act"] != primary_act
        and s["count"] >= 5                  # stricter than solo (was 2)
        and is_clean_support_entity(s["entity"])
    ]

    pool = same_act + other_act
    if not pool:
        return []
    k = min(k, len(pool))
    return random.sample(pool, k)


# ── Prompt ─────────────────────────────────────────────────────────────────────
def build_prompt(primary_seed, support_seed, style_variant, length_hint, example_context):
    primary_entity = primary_seed["entity"]
    primary_act    = act_display(primary_seed["primary_act"])
    example_ctx    = clean_text(example_context)[:400]

    support_line = ""
    if support_seed:
        support_line = f"\nSecondary supporting concept: {support_seed['entity']}\n"

    prompt = f"""You are writing one synthetic legal context for continued pretraining of an extractive question answering model.

Write exactly one self-contained legal passage in the style of an Indian intellectual property statute.

Primary Act:
{primary_act}

Primary legal concept:
{primary_entity}
{support_line}
Stylistic goal:
Make the passage {style_variant}. It should resemble Indian IPR legislative drafting and be suitable as context for extractive question answering.

Strict requirements:
- Write only the BODY TEXT of one legal passage. No heading. No title.
- Do NOT use bullet points or numbered lists.
- Do NOT use markdown formatting such as **bold** or *italic*.
- Do NOT write any question or answer.
- Do NOT open with a short title clause or commencement clause sentence.
- Do NOT write repeal or saving clauses.
- Do NOT use placeholders such as [insert date] or [Act name].
- Do NOT explain the law like a textbook.
- Do NOT cite specific section numbers (e.g. "under section 31", "section 63") — use general phrases like "under this Act" or "as provided herein" instead.
- Do NOT reference the Copyright Board — it was abolished in 2017.
- Start with a capital letter and a complete legal sentence. Do NOT start with "…" or a mid-sentence fragment.
- Use statutory phrasing: "shall", "may", "subject to", "provided that", where natural.
- Keep it self-contained: {length_hint}.
- The passage must explicitly contain the primary legal concept.
- If a secondary supporting concept is provided, naturally incorporate it.
- Use the reference example only as a style guide; do NOT copy it verbatim.

Reference style example from the corpus:
{example_ctx}""".strip()

    return prompt


def pick_example_context(seed, idx):
    """Rotate through up to 3 stored example contexts for variety."""
    examples = seed.get("example_contexts", [seed.get("example_context", "")])
    if not examples:
        return ""
    return examples[idx % len(examples)]


# ── Ollama call ────────────────────────────────────────────────────────────────
def call_ollama(prompt):
    response = ollama.generate(
        model=MODEL_NAME,
        prompt=prompt,
        options={
            "temperature":    TEMPERATURE,
            "top_p":          TOP_P,
            "repeat_penalty": REPEAT_PENALTY,
            "num_ctx":        NUM_CTX,
            "num_predict":    NUM_PREDICT,
        },
    )
    return response["response"].strip()


# ── Validation — relaxed for legal text ───────────────────────────────────────
# NOTE: section numbers, Act year references (1957, 1999, etc.), subsection (1),
#       and the word "commencement" in body text are ALL VALID in Indian IPR law.
#       We only block actual hallucination artifacts and formatting errors.
def validate(text, primary_entity, support_entity):
    reasons = []
    if not text:
        reasons.append("empty")
        return False, reasons

    raw  = text
    low  = raw.lower()
    words = re.findall(r"\w+", raw)

    # Length check
    if len(words) < 70:
        reasons.append("too_short")
    if len(words) > 280:
        reasons.append("too_long")

    # Must contain primary entity
    if primary_entity.lower() not in low:
        reasons.append("missing_primary_entity")

    # For broad generations, support entity should also appear
    if support_entity and support_entity.lower() not in low:
        reasons.append("missing_support_entity")

    # Ellipsis / truncation start — model started mid-sentence
    if re.match(r"^[\u2026\.]{1,3}\s*[a-z]", raw.strip()):
        reasons.append("ellipsis_start")
    # Lowercase start — continuation fragment (sentence was cut)
    if re.match(r"^[a-z]", raw.strip()):
        reasons.append("lowercase_start")

    # Hard rejections — only actual artifacts, not legal text patterns
    hard_blocked = [
        (r"\[insert\s+\w+\]",           "placeholder_bracket"),
        (r"\bshall come into force\b",   "commencement_clause_verb"),
        (r"\bthis act may be cited\b",   "citation_clause"),
        (r"\*\*.*?\*\*",                 "markdown_bold"),
        (r"^\s*[-*]\s+\w",              "bullet_style"),
        (r"Primary Act:",               "prompt_leak"),
        (r"Primary legal concept:",     "prompt_leak"),
        (r"Stylistic goal:",            "prompt_leak"),
    ]
    for pat, label in hard_blocked:
        if re.search(pat, raw, flags=re.I | re.M):
            reasons.append(label)

    # Contains question mark → model wrote a question
    if "?" in raw:
        reasons.append("contains_question_mark")

    # At least one legal-style signal
    legal_signals = ["shall", "may", "provided that", "subject to",
                     "thereof", "therein", "notwithstanding", "herein"]
    if not any(kw in low for kw in legal_signals):
        reasons.append("weak_legal_style")

    # At least 2 sentences
    sentences = [s for s in re.split(r"[.!]+", raw.strip()) if s.strip()]
    if len(sentences) < 2:
        reasons.append("too_few_sentences")

    # Textbook / summary style openings — model sometimes writes these
    textbook_openers = [
        r"^(in conclusion|this passage|this section describes|this provision|"
        r"the following|as discussed|in summary|overview of|introduction to)",
    ]
    for pat in textbook_openers:
        if re.search(pat, low[:80], flags=re.I):
            reasons.append("textbook_style_opener")

    # Severe repetition — if any 6-word phrase repeats 3+ times, it's a loop
    ngram_counts = Counter()
    word_list = re.findall(r"\w+", low)
    for i in range(len(word_list) - 5):
        ng = " ".join(word_list[i:i+6])
        ngram_counts[ng] += 1
    if ngram_counts and ngram_counts.most_common(1)[0][1] >= 3:
        reasons.append("repetitive_loop")

    ok = (len(reasons) == 0)
    return ok, reasons


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    if not SEED_FILE.exists():
        print(f"ERROR: Seed file not found: {SEED_FILE}")
        print("Run extract_seeds_from_train.py first.")
        return

    seeds = load_json(SEED_FILE)
    print(f"Loaded {len(seeds):,} seeds from: {SEED_FILE}")

    # Quality filter: remove low-frequency noise seeds
    before = len(seeds)
    seeds = [s for s in seeds if s.get("count", 0) >= MIN_SEED_COUNT]
    print(f"After MIN_SEED_COUNT={MIN_SEED_COUNT} filter: {len(seeds):,} (removed {before - len(seeds):,} noise seeds)")

    # Cap to top-N by frequency (already sorted descending by count)
    seeds = seeds[:MAX_SEEDS]
    print(f"Using top {len(seeds):,} seeds (MAX_SEEDS={MAX_SEEDS})")
    print(f"Seed count range: {seeds[0]['count']} (highest) → {seeds[-1]['count']} (lowest)")

    Path(OUTPUT_ACCEPTED).parent.mkdir(parents=True, exist_ok=True)
    seen_ok  = existing_ids(OUTPUT_ACCEPTED)
    seen_rej = existing_ids(OUTPUT_REJECTED)
    seen     = seen_ok | seen_rej
    print(f"Already done: {len(seen_ok):,} accepted, {len(seen_rej):,} rejected → resuming\n")

    total_planned   = len(seeds) * (len(STYLE_VARIANTS) + N_BROAD)
    accepted_count  = len(seen_ok)
    rejected_count  = len(seen_rej)
    target_estimate = int(total_planned * 0.80)

    print(f"Seeds        : {len(seeds):,}")
    print(f"Per seed     : {N_STYLES_SOLO} solo styles + {N_BROAD} broad = {N_STYLES_SOLO + N_BROAD}")
    print(f"Total planned: {total_planned:,}")
    print(f"Target (~80%): ~{target_estimate:,}\n")
    print(f"Model        : {MODEL_NAME}")
    print("─" * 60)

    with open(OUTPUT_ACCEPTED, "a", encoding="utf-8") as fout_ok, \
         open(OUTPUT_REJECTED, "a", encoding="utf-8") as fout_bad:

        for seed_idx, primary_seed in enumerate(seeds, start=1):
            primary_entity = primary_seed["entity"]
            primary_act    = primary_seed["primary_act"]

            # ── Solo generations (one per style variant) ───────────────────────
            for style_idx, (style_variant, length_hint) in enumerate(STYLE_VARIANTS):
                synthetic_id = safe_id(
                    f"{MODEL_NAME}__{primary_entity}__solo__s{style_idx + 1}"
                )
                if synthetic_id in seen:
                    continue

                accepted    = False
                last_text   = ""
                last_reasons = []

                for attempt in range(1, MAX_RETRIES + 1):
                    try:
                        example_ctx = pick_example_context(primary_seed, style_idx + attempt)
                        prompt = build_prompt(primary_seed, None, style_variant, length_hint, example_ctx)
                        text   = call_ollama(prompt)
                        last_text = text
                        ok, reasons = validate(text, primary_entity, None)
                        last_reasons = reasons

                        if ok:
                            row = {
                                "synthetic_id":   synthetic_id,
                                "model":          MODEL_NAME,
                                "primary_entity": primary_entity,
                                "support_entity": None,
                                "primary_act":    primary_act,
                                "style_variant":  style_variant,
                                "attempt":        attempt,
                                "synthetic_context": text,
                            }
                            fout_ok.write(json.dumps(row, ensure_ascii=False) + "\n")
                            fout_ok.flush()
                            accepted_count += 1
                            seen.add(synthetic_id)
                            accepted = True
                            print(
                                f"[OK  {accepted_count:>6}] "
                                f"seed {seed_idx:>4}/{len(seeds)} | "
                                f"{primary_entity[:32]:<32} | {style_variant}"
                            )
                            break

                    except Exception as e:
                        last_reasons = [f"error:{e}"]

                    time.sleep(SLEEP_BETWEEN)

                if not accepted:
                    row = {
                        "synthetic_id":     synthetic_id,
                        "model":            MODEL_NAME,
                        "primary_entity":   primary_entity,
                        "support_entity":   None,
                        "primary_act":      primary_act,
                        "style_variant":    style_variant,
                        "synthetic_context": last_text,
                        "rejection_reasons": last_reasons,
                    }
                    fout_bad.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fout_bad.flush()
                    rejected_count += 1
                    seen.add(synthetic_id)
                    print(
                        f"[REJ {rejected_count:>6}] "
                        f"seed {seed_idx:>4}/{len(seeds)} | "
                        f"{primary_entity[:32]:<32} | {last_reasons}"
                    )

            # ── Broad-supported generations ────────────────────────────────────
            broad_seeds = choose_broad_seeds(primary_seed, seeds, N_BROAD)
            for b_idx, support_seed in enumerate(broad_seeds, start=1):
                support_entity = support_seed["entity"]
                style_variant, length_hint = STYLE_VARIANTS[(seed_idx + b_idx) % len(STYLE_VARIANTS)]
                synthetic_id   = safe_id(
                    f"{MODEL_NAME}__{primary_entity}__{support_entity}__b{b_idx}"
                )
                if synthetic_id in seen:
                    continue

                accepted     = False
                last_text    = ""
                last_reasons = []

                for attempt in range(1, MAX_RETRIES + 1):
                    try:
                        example_ctx = pick_example_context(primary_seed, b_idx + attempt)
                        prompt = build_prompt(primary_seed, support_seed, style_variant, length_hint, example_ctx)
                        text   = call_ollama(prompt)
                        last_text = text
                        ok, reasons = validate(text, primary_entity, support_entity)
                        last_reasons = reasons

                        if ok:
                            row = {
                                "synthetic_id":   synthetic_id,
                                "model":          MODEL_NAME,
                                "primary_entity": primary_entity,
                                "support_entity": support_entity,
                                "primary_act":    primary_act,
                                "style_variant":  style_variant,
                                "attempt":        attempt,
                                "synthetic_context": text,
                            }
                            fout_ok.write(json.dumps(row, ensure_ascii=False) + "\n")
                            fout_ok.flush()
                            accepted_count += 1
                            seen.add(synthetic_id)
                            accepted = True
                            print(
                                f"[OK  {accepted_count:>6}] "
                                f"seed {seed_idx:>4}/{len(seeds)} | "
                                f"{primary_entity[:22]:<22}+{support_entity[:15]:<15} | {style_variant}"
                            )
                            break

                    except Exception as e:
                        last_reasons = [f"error:{e}"]

                    time.sleep(SLEEP_BETWEEN)

                if not accepted:
                    row = {
                        "synthetic_id":     synthetic_id,
                        "model":            MODEL_NAME,
                        "primary_entity":   primary_entity,
                        "support_entity":   support_entity,
                        "primary_act":      primary_act,
                        "style_variant":    style_variant,
                        "synthetic_context": last_text,
                        "rejection_reasons": last_reasons,
                    }
                    fout_bad.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fout_bad.flush()
                    rejected_count += 1
                    seen.add(synthetic_id)
                    print(
                        f"[REJ {rejected_count:>6}] "
                        f"seed {seed_idx:>4}/{len(seeds)} | "
                        f"{primary_entity[:22]:<22}+{support_entity[:15]:<15} | {last_reasons}"
                    )

    print("\n" + "=" * 60)
    print(f"Done.")
    print(f"Accepted  : {accepted_count:,}")
    print(f"Rejected  : {rejected_count:,}")
    total_done = accepted_count + rejected_count
    if total_done > 0:
        print(f"Accept rate: {accepted_count / total_done * 100:.1f}%")
    print(f"\nAccepted file : {OUTPUT_ACCEPTED}")
    print(f"Rejected file : {OUTPUT_REJECTED}")
    print("\nNext step: run merge_and_prepare_pretraining_v2.py")


if __name__ == "__main__":
    main()
