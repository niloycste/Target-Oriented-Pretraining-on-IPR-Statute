"""
Advanced inference script for extractive QA models.

Features:
  - model shortcuts (A, D, E, G direct; B', D2', E2', F', C', H', I' transfer/DAPT)
  - the same overlapping-window constrained decoder used for paper evaluation
  - literal source-context answer text and joint start--end logit reporting

Usage examples:
  python predict_qa.py --model A  --question "What does Section 1 say?" --context "..."
  python predict_qa.py --model C' --question "What does section 2 say?" --context "..."
  python predict_qa.py --model H' --question "..." --context-file sample_context.txt
"""

import argparse
import sys
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForQuestionAnswering


BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.qa_evaluation import predict_answer, resolve_device  # noqa: E402

MAX_SEQ_LENGTH = 384
DOC_STRIDE = 128

DIRECT = BASE_DIR / "artifacts" / "models" / "direct"
SQ1 = BASE_DIR / "artifacts" / "models" / "squad_v1"

MODEL_PATHS = {
    # Direct IPR fine-tuning models
    "A": DIRECT / "baseline_a" / "final",
    "D": DIRECT / "baseline_d" / "final",
    "E": DIRECT / "baseline_e" / "final",
    "G": DIRECT / "baseline_g" / "final",
    # SQuAD v1 ablation models
    "B'":  SQ1 / "baseline_b_squadv1"  / "final",
    "D2'": SQ1 / "baseline_d2_squadv1" / "final",
    "E2'": SQ1 / "baseline_e2_squadv1" / "final",
    "F'":  SQ1 / "baseline_f_squadv1"  / "final",
    "C'": SQ1 / "squad_v1_modelc_v2" / "ipr_final" / "final",
    "H'":  SQ1 / "original_only_squadv1" / "ipr_final" / "final",
    "I'": SQ1 / "squad_v1_synth_only" / "ipr_final" / "final",
}


def load_context(args):
    if args.context:
        return args.context.strip()
    if args.context_file:
        return Path(args.context_file).read_text(encoding="utf-8").strip()
    raise ValueError("Provide either --context or --context-file")


def main():
    parser = argparse.ArgumentParser(description="Run extractive QA inference with a trained model.")
    parser.add_argument("--model", required=True,
                        help="Model shortcut: A D E G (Direct IPR) or "
                             "B' D2' E2' F' C' H' I' (transfer/DAPT)")
    parser.add_argument("--question", required=True, help="Question text")
    parser.add_argument("--context", help="Context text")
    parser.add_argument("--context-file", help="Path to a text file containing context")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    model_key = args.model.strip()
    if model_key not in MODEL_PATHS:
        raise ValueError(f"Unknown model '{args.model}'. Choose from: {', '.join(MODEL_PATHS.keys())}")

    model_path = MODEL_PATHS[model_key]
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    question = args.question.strip()
    context = load_context(args)

    print(f"Loading model {model_key} from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForQuestionAnswering.from_pretrained(str(model_path), local_files_only=True)
    device = resolve_device(args.device)
    model.to(device)
    model.eval()

    answer, score = predict_answer(
        tokenizer,
        model,
        device,
        question,
        context,
        max_seq_length=MAX_SEQ_LENGTH,
        doc_stride=DOC_STRIDE,
        max_answer_tokens=None,
    )

    print("\nQuestion:")
    print(question)

    if not answer:
        print("\nNo non-empty answer span could be extracted.")
        return

    print("\nBest predicted answer:")
    print(answer)
    print(f"\nJoint logit score: {score:.6f}")


if __name__ == "__main__":
    main()
