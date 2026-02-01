import os
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    seeclick_dir = root / "SeeClick"
    assets_img = seeclick_dir / "assets" / "test_img.png"

    print(f"Project root: {root}")
    print(f"SeeClick folder exists: {seeclick_dir.exists()}")
    print(f"Test image exists: {assets_img.exists()}")

    # Basic dependency check
    try:
        from transformers import AutoTokenizer  # type: ignore

        print("Transformers imported successfully.")
    except Exception as e:  # pragma: no cover - simple environment check
        print("Error importing transformers:", repr(e))
        return

    # Light-weight check: tokenizer only (no large model weights)
    try:
        print("Loading Qwen-VL-Chat tokenizer (this may download once and then be cached)...")
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen-VL-Chat",
            trust_remote_code=True,
        )
        # Simple sanity check on tokenizer
        test_ids = tokenizer("hello", return_tensors="pt")
        print("Qwen-VL tokenizer loaded. Example token IDs shape:", {k: v.shape for k, v in test_ids.items()})
    except Exception as e:  # pragma: no cover - simple environment check
        print("Error loading Qwen-VL tokenizer:", repr(e))
        return

    print("SeeClick basic setup test finished.")


if __name__ == "__main__":
    main()


