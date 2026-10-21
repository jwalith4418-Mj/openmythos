#!/usr/bin/env python3
"""Export OpenMythos checkpoint to HuggingFace / SafeTensors format for vLLM."""

import argparse
import sys

sys.path.insert(0, ".")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to OpenMythos checkpoint dir")
    p.add_argument("--output", required=True, help="Output directory for HF format")
    p.add_argument("--tokenizer", default=None, help="Path to tokenizer to copy into output")
    args = p.parse_args()

    from open_mythos.main import OpenMythos

    print(f"Loading checkpoint from {args.checkpoint}")
    model = OpenMythos.from_pretrained(args.checkpoint, device="cpu")
    print(f"Saving HF-format checkpoint to {args.output}")
    model.save_pretrained(args.output)

    if args.tokenizer:
        import shutil

        shutil.copytree(args.tokenizer, args.output, dirs_exist_ok=True)
        print(f"Tokenizer copied from {args.tokenizer}")

    print(f"Export complete. Load with: OpenMythos.from_pretrained('{args.output}')")
    print(f"vLLM: vllm serve {args.output} --trust-remote-code")


if __name__ == "__main__":
    main()
