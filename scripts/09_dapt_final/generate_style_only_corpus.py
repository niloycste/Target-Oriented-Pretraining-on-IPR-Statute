"""STEP 4 (dapt.md sec. 16-18): generate the S_STYLE target-alignment control.

S_STYLE is the control that separates "legal-looking text" from "text aligned to
the downstream target". It holds everything constant against S -- generator,
decoding parameters, the eight style families, the quality filter, length
regime -- and varies exactly one thing: the prompt is NOT conditioned on
IndicIPR-QA training concepts and does not concern the seven target IPR Acts.

The filter and Ollama call are IMPORTED from the original generator rather than
reimplemented, so "same filtering pipeline" is enforced by construction and
cannot drift.

Per sec. 17 nothing is scraped: the non-target domains, concepts and style
exemplars below are a fixed, hand-authored, documented substitution for the
train-derived seed pool.

Resumable (sec. 14): re-running appends only what is missing.

Run: python scripts/09_dapt_final/generate_style_only_corpus.py
"""

import importlib.util
import io
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "scripts" / "02_synthetic_dapt" / "generate_synthetic_large.py"
OUT_DIR = ROOT / "experiments" / "dapt" / "final_cpu"
OUT = OUT_DIR / "style_only_generated.jsonl"
REJECTED = OUT_DIR / "style_only_rejected.jsonl"

# Generate a surplus; build_style_only_corpus.py trims to S's exact token budget.
TARGET_ACCEPTED = 1000
MAX_RETRIES = 2
GENERATION_SEED = 20260829

# Import the original generator so filter/decoding are literally the same code.
spec = importlib.util.spec_from_file_location("gen_large", SRC)
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)

STYLE_VARIANTS = gen.STYLE_VARIANTS  # the same eight families

# --- Non-target statutory domains (deliberately NOT the seven IPR Acts) -------
NON_TARGET_ACTS = [
    "The Indian Contract Act, 1872",
    "The Indian Evidence Act, 1872",
    "The Motor Vehicles Act, 1988",
    "The Consumer Protection Act, 2019",
    "The Arbitration and Conciliation Act, 1996",
    "The Food Safety and Standards Act, 2006",
    "The Electricity Act, 2003",
    "The Environment (Protection) Act, 1986",
    "The Right to Information Act, 2005",
    "The Payment of Gratuity Act, 1972",
]

# Bare noun phrases, mirroring the form of the train-derived seed entities
# ("commission offence", "application registration"). The quality filter requires
# the concept to appear verbatim, so a leading article would reject on wording
# rather than on quality.
NON_TARGET_CONCEPTS = [
    "appointed authority", "prescribed particulars", "competent officer",
    "appellate tribunal", "inspection report", "statutory declaration",
    "compounding of offences", "register of undertakings",
    "period of limitation", "notice of demand", "recovery of dues",
    "conditions of licence", "duty of disclosure", "adjudicating officer",
    "manner of service", "maintenance of records", "power of entry",
    "schedule of fees", "terms of settlement", "security deposit",
    "certificate of registration", "grounds of revocation",
    "delegation of powers", "summary procedure", "burden of proof",
]

# Hand-authored generic statutory exemplars, one per style family. These carry
# drafting register only and contain no IPR subject matter and no corpus text.
STYLE_EXEMPLARS = [
    "In this Chapter, unless the context otherwise requires, the expressions used "
    "shall have the meanings respectively assigned to them, and any reference to a "
    "prescribed particular shall be construed as a reference to such particular as "
    "may be prescribed by rules made under this Act.",
    "Every application under this Chapter shall be made in such form and accompanied "
    "by such fee as may be prescribed, and the authority may, before disposing of the "
    "application, require the applicant to furnish such further particulars as it "
    "considers necessary.",
    "Where any person is aggrieved by an act done in contravention of the provisions "
    "of this Act, such person shall be entitled to such relief as may be granted, and "
    "the remedies provided herein shall be in addition to and not in derogation of any "
    "other remedy available under any other law for the time being in force.",
    "The competent authority may, by order in writing and for reasons to be recorded "
    "therein, authorise any officer subordinate to it to exercise such of its powers "
    "under this Act as may be specified in the order, subject to such conditions and "
    "limitations as may be laid down therein.",
    "No person shall carry on any activity to which this Chapter applies except in "
    "accordance with the conditions specified herein, and every person so carrying on "
    "such activity shall maintain such records and furnish such returns as may be "
    "prescribed.",
    "Whoever contravenes any provision of this Chapter, or any rule or order made "
    "thereunder, shall be punishable with such penalty as may be provided, and where "
    "the contravention is a continuing one, with a further penalty for every day during "
    "which such contravention continues.",
    "Any right conferred under this Chapter may be transferred, either wholly or in "
    "part, by an instrument in writing signed by the party conferring the right, and no "
    "such transfer shall be effective unless it is registered in the manner prescribed.",
    "Any person aggrieved by an order made under this Chapter may, within the period of "
    "limitation specified herein, prefer an appeal to the appellate authority, and the "
    "appellate authority shall, after giving the parties a reasonable opportunity of "
    "being heard, pass such order as it thinks fit.",
]


def build_style_prompt(concept, act_display, style_variant, length_hint, exemplar):
    """Identical drafting contract to build_prompt(), minus target conditioning."""
    return f"""You are writing one synthetic legal context for continued pretraining of an extractive question answering model.

Write exactly one self-contained legal passage in the style of an Indian statute.

Primary Act:
{act_display}

Primary legal concept:
{concept}

Stylistic goal:
Make the passage {style_variant}. It should resemble Indian legislative drafting and be suitable as context for extractive question answering.

Strict requirements:
- Write only the BODY TEXT of one legal passage. No heading. No title.
- Do NOT use bullet points or numbered lists.
- Do NOT use markdown formatting such as **bold** or *italic*.
- Do NOT write any question or answer.
- Do NOT open with a short title clause or commencement clause sentence.
- Do NOT write repeal or saving clauses.
- Do NOT use placeholders such as [insert date] or [Act name].
- Do NOT explain the law like a textbook.
- Do NOT cite specific section numbers — use general phrases like "under this Act" or "as provided herein" instead.
- Do NOT write about copyright, patents, trade marks, designs, geographical indications, semiconductor layout-designs, or plant varieties.
- Start with a capital letter and a complete legal sentence. Do NOT start with "…" or a mid-sentence fragment.
- Use statutory phrasing: "shall", "may", "subject to", "provided that", where natural.
- Keep it self-contained: {length_hint}.
- The passage must explicitly contain the primary legal concept.
- Use the reference example only as a style guide; do NOT copy it verbatim.

Reference style example:
{exemplar}""".strip()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        for line in io.open(OUT, encoding="utf8"):
            if line.strip():
                done.add(json.loads(line)["style_id"])
    print(f"  resuming: {len(done)} already accepted, target {TARGET_ACCEPTED}")

    rng = random.Random(GENERATION_SEED)
    slots = []
    for act in NON_TARGET_ACTS:
        for concept in NON_TARGET_CONCEPTS:
            for i, (style, hint) in enumerate(STYLE_VARIANTS):
                slots.append((act, concept, style, hint, STYLE_EXEMPLARS[i]))
    rng.shuffle(slots)

    accepted, rejected, t0 = len(done), 0, time.time()
    fout = io.open(OUT, "a", encoding="utf8")
    frej = io.open(REJECTED, "a", encoding="utf8")

    for act, concept, style, hint, exemplar in slots:
        if accepted >= TARGET_ACCEPTED:
            break
        sid = gen.safe_id(f"{act}|{concept}|{style}")
        if sid in done:
            continue
        prompt = build_style_prompt(concept, act, style, hint, exemplar)
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                text = gen.clean_text(gen.call_ollama(prompt))
            except Exception as exc:  # generation is long-running; never abort
                print(f"    ollama error: {exc}")
                time.sleep(2)
                continue
            ok, reasons = gen.validate(text, concept, None)
            if ok:
                fout.write(json.dumps({
                    "style_id": sid, "model": gen.MODEL_NAME, "condition": "S_STYLE",
                    "non_target_act": act, "primary_entity": concept,
                    "style_variant": style, "attempt": attempt,
                    "synthetic_context": text,
                }, ensure_ascii=False) + "\n")
                fout.flush()
                accepted += 1
                break
            rejected += 1
            frej.write(json.dumps({"style_id": sid, "attempt": attempt,
                                   "reasons": reasons, "text": text},
                                  ensure_ascii=False) + "\n")
            frej.flush()
        if accepted % 25 == 0 and accepted:
            el = time.time() - t0
            rate = (accepted - len(done)) / max(el, 1e-9)
            left = (TARGET_ACCEPTED - accepted) / rate / 3600 if rate > 0 else 0
            print(f"    accepted {accepted}/{TARGET_ACCEPTED}  rejected {rejected}  "
                  f"~{left:.1f}h left")

    fout.close()
    frej.close()
    print(f"\n  accepted {accepted}  rejected {rejected}")
    print(f"  wrote {OUT.relative_to(ROOT)}")
    if accepted < TARGET_ACCEPTED:
        print(
            f"  ERROR: generation exhausted its fixed slots at {accepted}/"
            f"{TARGET_ACCEPTED}; no finished S_STYLE corpus can be built."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
