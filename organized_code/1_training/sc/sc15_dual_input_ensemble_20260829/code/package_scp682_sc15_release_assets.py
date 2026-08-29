from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.release_dir / "release_assets"
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.release_dir / "ensemble_config.json").read_text())
    rows = []
    for fold in config["folds"]:
        split_seed = int(fold["split_seed"])
        archive = output / f"SCP682-SC15_models_split_{split_seed}.zip"
        files = [Path(value) for value in fold["members"]]
        files.append(Path(fold["preprocessing"]))
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
            for relative in files:
                bundle.write(args.release_dir / relative, arcname=relative.as_posix())
        rows.append(
            {
                "split_seed": split_seed,
                "asset": archive.name,
                "bytes": archive.stat().st_size,
                "sha256": sha256(archive),
            }
        )
    import pandas as pd

    pd.DataFrame(rows).to_csv(
        args.release_dir / "release_asset_hashes.tsv", sep="\t", index=False
    )


if __name__ == "__main__":
    main()
