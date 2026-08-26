from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(r"D:\data\lsy\models\scTranslator")
PATTERNS = [
    "normalization",
    "normalize",
    "min-max",
    "log1p",
    "expm1",
    "inverse",
    "scale",
    "scaler",
    "y_pred",
    "y_truth",
    "protein abundance",
    "leave the values as 0",
    "h5ad",
]


def main() -> None:
    rows = []
    for path in ROOT.rglob("*"):
        if path.suffix.lower() not in {".py", ".md", ".txt", ".csv"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            low = line.lower()
            for pat in PATTERNS:
                if pat.lower() in low:
                    rows.append(
                        {
                            "path": str(path),
                            "line": i,
                            "pattern": pat,
                            "text": line.strip(),
                        }
                    )
                    break
    out = ROOT / "_official_preprocess_keyword_hits.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in rows[:300]:
        print(f"{row['path']}:{row['line']} [{row['pattern']}] {row['text']}")
    print(f"\nN_HITS={len(rows)}")
    print(f"OUT={out}")


if __name__ == "__main__":
    main()
