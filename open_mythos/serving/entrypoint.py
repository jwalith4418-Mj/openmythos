"""
Production entrypoint for OpenMythos inference server.
Usage: python -m open_mythos.serving.entrypoint [args]
"""

import argparse
import os


def main():
    p = argparse.ArgumentParser(description="OpenMythos Inference Server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    p.add_argument("--model-path", default=os.getenv("MODEL_PATH", ""))
    p.add_argument("--vllm-model-path", default=os.getenv("VLLM_MODEL_PATH", ""))
    p.add_argument("--sglang-model-path", default=os.getenv("SGLANG_MODEL_PATH", ""))
    p.add_argument("--tp-size", type=int, default=int(os.getenv("TENSOR_PARALLEL_SIZE", "1")))
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()

    model, tokenizer = None, None

    if args.model_path and not args.vllm_model_path and not args.sglang_model_path:
        from open_mythos.main import OpenMythos

        print(f"[server] Loading native model from {args.model_path}")
        model = OpenMythos.from_pretrained(args.model_path, device=args.device)
        model.eval()
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        except Exception:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")

            class _Tok:
                eos_token_id = enc.eot_token

                def encode(self, t):
                    return enc.encode(t)

                def decode(self, ids):
                    return enc.decode(ids)

            tokenizer = _Tok()

    from open_mythos.serving.server import serve

    serve(
        model=model,
        tokenizer=tokenizer,
        host=args.host,
        port=args.port,
        device=args.device,
        workers=args.workers,
        vllm_model_path=args.vllm_model_path or None,
        sglang_model_path=args.sglang_model_path or None,
        tensor_parallel_size=args.tp_size,
    )


if __name__ == "__main__":
    main()
