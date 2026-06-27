from __future__ import annotations

import argparse
import sys

from bilm.model import BILM


def main() -> None:
    parser = argparse.ArgumentParser(description="BILM text generation")
    parser.add_argument("--prompt", required=True, help="Seed prompt for generation")
    parser.add_argument("--checkpoint", default=None, help="Path to .npz checkpoint")
    parser.add_argument("--max-chars", type=int, default=200, help="Characters to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (0=argmax)")
    args = parser.parse_args()

    model = BILM()

    if args.checkpoint:
        import os
        if not os.path.exists(args.checkpoint):
            print(f"Error: checkpoint not found: {args.checkpoint}", file=sys.stderr)
            sys.exit(1)
        print(f"Loading checkpoint: {args.checkpoint}")
        model.load(args.checkpoint)
        print(f"Loaded. Tokens trained: {model.tokens_seen:,}\n")
    else:
        print("No checkpoint provided. Generating from untrained model (random output expected).\n")

    print(f"Prompt: {args.prompt!r}")
    print("Generated:")
    print("-" * 50)
    output = model.generate(
        prompt=args.prompt,
        max_bytes=args.max_chars,
        temperature=args.temperature,
    )
    print(args.prompt + output)
    print("-" * 50)


if __name__ == "__main__":
    main()
