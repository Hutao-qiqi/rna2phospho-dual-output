import importlib
import sys

print(sys.executable)
for name in ["torch", "torch_geometric", "pandas", "pyarrow", "scipy"]:
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "ok")
        if name == "torch":
            print(name, version, "cuda", mod.cuda.is_available())
            if mod.cuda.is_available():
                print("device_count", mod.cuda.device_count())
                print("device0", mod.cuda.get_device_name(0))
        else:
            print(name, version)
    except Exception as exc:
        print(name, "ERR", repr(exc))
