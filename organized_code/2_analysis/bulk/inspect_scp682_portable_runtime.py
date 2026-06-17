from pathlib import Path

import torch


path = Path(r"D:\data\lsy\vm_lsy_parent\lsy\SCP682_PORTABLE\models\scp682_graph_runtime_state.pt")
obj = torch.load(path, map_location="cpu", weights_only=False)
print(type(obj))
print(sorted(obj.keys()))
for key, value in obj.items():
    if hasattr(value, "shape"):
        print(key, tuple(value.shape), getattr(value, "dtype", None))
    elif isinstance(value, (list, tuple)):
        print(key, "list", len(value), "first", list(value[:3]))
    elif isinstance(value, dict):
        print(key, "dict", sorted(value.keys())[:20])
    else:
        shown = value if isinstance(value, (str, int, float, bool)) else ""
        print(key, type(value), shown)
