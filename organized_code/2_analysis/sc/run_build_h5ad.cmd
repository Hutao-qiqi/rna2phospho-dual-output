@echo off
D:
cd /d D:\data\lsy\vm_lsy_parent\lsy\03_code\single_cell\modeling
D:\Tools\anaconda3\python.exe D:\data\lsy\vm_lsy_parent\lsy\03_code\single_cell\modeling\build_foundation_model_h5ad_inputs.py --root D:\data\lsy\vm_lsy_parent\lsy --model-input-dir 01_data\single_cell\intermediate\phospho_model_inputs\scfoundation_cap12000_gse300551_signal_seq_multidomain_v1 --output-dir 01_data\single_cell\intermediate\foundation_model_h5ad_inputs_v1 > "D:\data\lsy\vm_lsy_parent\lsy\02_results\single_cell\20260531_foundation_model_h5ad_inputs_v1\logs\build_h5ad_stdout.log" 2> "D:\data\lsy\vm_lsy_parent\lsy\02_results\single_cell\20260531_foundation_model_h5ad_inputs_v1\logs\build_h5ad_stderr.log"
echo %ERRORLEVEL% > "D:\data\lsy\vm_lsy_parent\lsy\02_results\single_cell\20260531_foundation_model_h5ad_inputs_v1\logs\build_h5ad_exit_code.txt"
