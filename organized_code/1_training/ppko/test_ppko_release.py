"""Check all packaged PPKO weights against their architecture and deterministic output."""

import importlib.util
import json
from pathlib import Path

import torch


def load_module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    root = Path(__file__).resolve().parent
    registry = json.loads((root / "PPKO_MODELS.json").read_text())
    torch.set_num_threads(4)
    torch.manual_seed(682)
    for entry in registry["models"]:
        for name in ["checkpoint", "training", "inference", "evaluation"]:
            if name in entry:
                assert (root / entry[name]).is_file(), entry[name]
        source, classname = entry["architecture"].split(":")
        cls = getattr(load_module(root / source), classname)
        checkpoint = torch.load(root / entry["checkpoint"], map_location="cpu", weights_only=False)
        if entry["id"] == "target_v10b":
            ns, np_ = len(checkpoint["sites"]), len(checkpoint["proteins"])
            model = cls(ns, np_, hidden=checkpoint["args"]["hidden"], latent=checkpoint["args"]["latent"])
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            inputs = (torch.randn(2, ns), torch.ones(2, ns, dtype=torch.bool),
                      torch.randn(2, np_), torch.randn(2, ns))
            def predict():
                outputs = model(*inputs)
                torch.testing.assert_close(outputs[0], outputs[1] + outputs[2] + outputs[3])
                return outputs[0]
        else:
            cfg = checkpoint["model_config"]
            model = cls(**{k: cfg[k] for k in ["n_sites", "n_cells", "fingerprint_dim"]})
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            inputs = dict(baseline=torch.randn(2, cfg["n_sites"]),
                          input_mask=torch.ones(2, cfg["n_sites"], dtype=torch.bool),
                          fingerprint=torch.zeros(2, cfg["fingerprint_dim"]),
                          cell_ids=torch.zeros(2, dtype=torch.long), condition=torch.zeros(2, 2))
            def predict():
                return model(inputs)
        model.eval()
        with torch.inference_mode():
            first, second = predict(), predict()
        assert first.shape == (2, entry["n_outputs"])
        assert torch.isfinite(first).all()
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        print("PASS", entry["id"], tuple(first.shape))


if __name__ == "__main__":
    main()
