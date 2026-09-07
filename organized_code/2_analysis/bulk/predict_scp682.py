"""调用SCP682 M2.2基础推理入口。"""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parents[3]/"SCP682/predict.py"),run_name="__main__")
