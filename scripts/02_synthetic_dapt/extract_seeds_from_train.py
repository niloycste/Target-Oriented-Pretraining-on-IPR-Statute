"""
Extract Entity Seeds from train.jsonl — for large-scale synthetic context generation.

Reads data/splits/train.jsonl (3,710 records).
Uses: context + question fields.
Methods: rule-based legal patterns + KeyBERT keyphrases (if installed).

Output: synthetic context/seed_pool_large.json
  Each seed:  {entity, count, primary_act, acts, example_context}

Run: python extract_seeds_from_train.py
"""

import json
import re
from collections import defaultdict, Counter
from pathlib import Path

try:
    from keybert import KeyBERT
    HAS_KEYBERT = True
except ImportError:
    HAS_KEYBERT = False
    print("[INFO] KeyBERT not available — using rule-based extraction only.")
    print("[INFO] Install with: pip install keybert sentence-transformers")

BASE_DIR   = Path(__file__).resolve().parents[2]
TRAIN_FILE = BASE_DIR / "data" / "splits" / "train.jsonl"
OUTPUT_DIR = BASE_DIR / "data" / "synthetic"
OUTPUT_FILE = OUTPUT_DIR / "seed_pool_large.json"

# ── Settings ──────────────────────────────────────────────────────────────────
MIN_COUNT      = 1    # keep seeds seen at least this many times
TOP_N_KEYWORDS = 8    # KeyBERT keyphrases per record
MAX_TEXT_LEN   = 1500 # cap text length for KeyBERT

# ── Legal Patterns (entity, weight) ───────────────────────────────────────────
LEGAL_PATTERNS = [
    # IPR institutions
    (r"\bCentral Government\b",           4),
    (r"\bOfficial Gazette\b",             4),
    (r"\bRegistrar\b",                    3),
    (r"\bAppellate Board\b",              4),
    (r"\bHigh Court\b",                   3),
    (r"\bCopyright Office\b",             4),
    (r"\bTrade Marks Registry\b",         4),
    (r"\bController\b",                   3),
    (r"\bPatent Office\b",                3),
    (r"\bPatent Agent\b",                 4),
    # Copyright
    (r"\bcopyright\b",                    2),
    (r"\bauthor\b",                       2),
    (r"\bperformer\b",                    3),
    (r"\bbroadcasting organisation\b",    4),
    (r"\bcopyright society\b",            4),
    (r"\bliterary work\b",                3),
    (r"\bartistic work\b",                3),
    (r"\bdramatic work\b",                3),
    (r"\bmusical work\b",                 3),
    (r"\bcinematograph film\b",           4),
    (r"\bcomputer programme\b",           4),
    (r"\bsound recording\b",              3),
    (r"\bvisual recording\b",             3),
    (r"\bRight Management Information\b", 4),
    (r"\bmoral right\b",                  4),
    (r"\bperformer['']s right\b",         4),
    (r"\bcompulsory licence\b",           4),
    (r"\binfringing copy\b",              4),
    (r"\bexclusive licence\b",            4),
    (r"\blicence\b",                      3),
    (r"\bassignment\b",                   3),
    (r"\broyalty\b",                      3),
    # Patents
    (r"\bpatent\b",                       2),
    (r"\bpatentee\b",                     4),
    (r"\bnovelty\b",                      3),
    (r"\binventive step\b",               4),
    (r"\bindustrial application\b",       4),
    (r"\bprior art\b",                    3),
    (r"\bterm of patent\b",               4),
    (r"\bworking of patent\b",            4),
    (r"\blicence of right\b",             4),
    (r"\bstatement of working\b",         4),
    (r"\bIndian patent\b",                3),
    (r"\bspecification\b",                3),
    (r"\bclaims\b",                       2),
    (r"\bopposition\b",                   3),
    (r"\brevocation\b",                   3),
    (r"\bsurrender\b",                    3),
    (r"\bcompulsory licensing\b",         4),
    # Trade Marks
    (r"\btrade mark\b",                   3),
    (r"\bregistered trade mark\b",        4),
    (r"\bwell-known trade mark\b",        4),
    (r"\bservice mark\b",                 3),
    (r"\bcertification trade mark\b",     4),
    (r"\bcollective mark\b",              4),
    (r"\bregistered proprietor\b",        4),
    (r"\bregistered user\b",              4),
    (r"\bpassing off\b",                  3),
    (r"\binfringement\b",                 3),
    # Designs
    (r"\bdesign\b",                       2),
    (r"\bnovelty of design\b",            4),
    (r"\bregistered design\b",            4),
    (r"\bpiracy of design\b",             4),
    # GI
    (r"\bgeographical indication\b",      4),
    (r"\bauthorised user\b",              4),
    (r"\bproducers\b",                    2),
    (r"\bGI tag\b",                       3),
    # Semiconductor
    (r"\blayout-design\b",                4),
    (r"\bsemiconductor\b",                3),
    (r"\bintegrated circuit\b",           3),
    # Plant Varieties
    (r"\bbreeder\b",                      3),
    (r"\bfarmer\b",                       3),
    (r"\bplant variety\b",                3),
    (r"\bnew variety\b",                  3),
    (r"\bessentially derived variety\b",  4),
    (r"\bprotection of plant variety\b",  4),
    (r"\bfarmers[''] right\b",            4),
    (r"\bbreeder['']s right\b",           4),
    (r"\bnational register\b",            3),
    # General IPR
    (r"\bregistration\b",                 2),
    (r"\bapplication\b",                  2),
]

QUOTED_TERM_RE = re.compile(r'["\u201c]([^"\u201d]{3,60})["\u201d]')
DEFINED_TERM_RE = re.compile(
    r'(?i)(?:\([a-z0-9]+\)\s*)?["\u201c]?([^"\n;:]{3,60}?)["\u201d]?\s+(means|includes)\b'
)

STOP_EXACT = {
    "context", "question", "questions", "title", "section", "subsection",
    "sub-section", "act", "authority", "government", "the", "and", "or",
    "not", "any", "it", "its", "this", "that", "those", "these", "than",
    "then", "where", "such", "other", "unless", "until", "under", "without",
    "with", "within", "shall", "may", "from", "been", "have", "has", "had",
}

DROP_RE = re.compile(
    r"^(\d+|[ivxlcdm]+|one|two|three|first|second|third|\W+)$",
    flags=re.I
)


def clean_text(text: str) -> str:
    return " ".join(str(text).split()).strip()


def normalize_entity(ent: str) -> str:
    ent = clean_text(ent).strip(" ,;:.[](){}'\"")
    ent = re.sub(r"^the\s+", "", ent, flags=re.I)
    return ent


def should_drop(ent: str) -> bool:
    if not ent or len(ent) < 3:
        return True
    if ent.lower() in STOP_EXACT:
        return True
    if ent.lower().endswith(" means") or ent.lower().endswith(" includes"):
        return True
    if ent.lower().startswith(("what ", "under section ", "which ", "who ")):
        return True
    if DROP_RE.fullmatch(ent):
        return True
    return False


def main():
    kw_model = None
    if HAS_KEYBERT:
        print("Loading KeyBERT (all-MiniLM-L6-v2)...")
        kw_model = KeyBERT(model="all-MiniLM-L6-v2")
        print("KeyBERT loaded.\n")

    entity_counts  = Counter()
    entity_acts    = defaultdict(set)
    entity_examples = defaultdict(list)   # entity → [(act, context), ...]

    print(f"Reading: {TRAIN_FILE}")
    records = []
    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Total train records : {len(records):,}\n")

    for idx, row in enumerate(records, start=1):
        act     = row.get("act", "unknown")
        context = clean_text(row.get("context", ""))
        question = clean_text(row.get("question", ""))
        text    = context + " " + question

        def add(ent, weight):
            ent = normalize_entity(ent)
            if should_drop(ent):
                return
            entity_counts[ent] += weight
            entity_acts[ent].add(act)
            if len(entity_examples[ent]) < 3:
                entity_examples[ent].append((act, context))

        # 1. Rule-based
        for pat, w in LEGAL_PATTERNS:
            for m in re.findall(pat, text, flags=re.I):
                add(m, w)

        # 2. Quoted terms
        for m in QUOTED_TERM_RE.findall(text):
            add(m, 3)

        # 3. Defined terms (context only)
        for m in DEFINED_TERM_RE.findall(context):
            add(m[0], 4)

        # 4. KeyBERT on context
        if kw_model and context:
            try:
                kws = kw_model.extract_keywords(
                    context[:MAX_TEXT_LEN],
                    keyphrase_ngram_range=(1, 3),
                    stop_words="english",
                    use_maxsum=False,
                    top_n=TOP_N_KEYWORDS,
                )
                for phrase, score in kws:
                    add(phrase, max(1, int(round(score * 8))))
            except Exception:
                pass

        if idx % 300 == 0:
            print(f"  Processed {idx}/{len(records)} records ...")

    # ── Build final seed pool ─────────────────────────────────────────────────
    print("\nBuilding seed pool ...")
    seed_pool = []
    for ent, count in entity_counts.most_common():
        if count < MIN_COUNT:
            continue
        if should_drop(ent):
            continue

        acts_list = sorted(entity_acts[ent])
        examples  = entity_examples[ent]

        # Primary act = act where entity appears most
        act_freq = Counter(a for a, _ in examples)
        if act_freq:
            primary_act = act_freq.most_common(1)[0][0]
        elif acts_list:
            primary_act = acts_list[0]
        else:
            primary_act = "unknown"

        # Best example context from primary act
        example_context = ""
        for a, ctx in examples:
            if a == primary_act:
                example_context = ctx
                break
        if not example_context and examples:
            example_context = examples[0][1]

        # Collect up to 3 example contexts (rotating for prompt variety)
        all_examples = [ctx for _, ctx in examples]
        if not all_examples:
            all_examples = [""]

        seed_pool.append({
            "entity":           ent,
            "count":            count,
            "primary_act":      primary_act,
            "acts":             acts_list,
            "example_context":  example_context[:500],       # primary (backward compat)
            "example_contexts": [c[:500] for c in all_examples],  # list for rotation
        })

    # ── Stats ─────────────────────────────────────────────────────────────────
    print(f"\nTotal unique seeds : {len(seed_pool):,}")

    act_entity_counts = Counter(s["primary_act"] for s in seed_pool)
    print("\nSeeds per act:")
    for act, cnt in act_entity_counts.most_common():
        print(f"  {act:<48} : {cnt:>4}")

    print("\nTop 40 seeds:")
    for s in seed_pool[:40]:
        print(f"  [{s['count']:5d}] {s['entity']:<45} | {s['primary_act']}")

    # ── Save ──────────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(seed_pool, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(seed_pool):,} seeds to: {OUTPUT_FILE}")

    # ── Generation estimate ───────────────────────────────────────────────────
    n_seeds = len(seed_pool)
    n_solo  = 8    # 8 style variants (tuples in generate_synthetic_large.py)
    n_broad = 6    # broad-supported
    planned = n_seeds * (n_solo + n_broad)
    print(f"\nGeneration estimate with generate_synthetic_large.py:")
    print(f"  {n_seeds} seeds × ({n_solo} solo + {n_broad} broad) = {planned:,} planned")
    print(f"  At ~80% acceptance → ~{int(planned * 0.80):,} accepted")


if __name__ == "__main__":
    main()
