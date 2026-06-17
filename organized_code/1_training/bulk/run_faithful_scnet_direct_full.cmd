@echo off
set CUDA_VISIBLE_DEVICES=0
D:\data\lsy\envs\scgpt\python.exe D:\data\lsy\vm_lsy_parent\lsy\SCP682_MAIN\scripts\train_faithful_scnet_direct_bulk.py ^
  --package-dir D:\data\lsy\vm_lsy_parent\lsy\SCP682_MAIN ^
  --rna-path D:\data\lsy\vm_lsy_parent\lsy\SCP682_MAIN\training_set\rna_log2_tpm_paired.parquet ^
  --prior-dir D:\data\lsy\vm_lsy_parent\lsy\SCP682_MAIN\priors ^
  --output-dir D:\data\lsy\vm_lsy_parent\lsy\02_results\model_validation\20260602_faithful_scnet_direct_bulk_c1 ^
  --device cuda:0 ^
  --epochs 160 ^
  --batch-size 8 ^
  --inter-dim 96 ^
  --embd-dim 32 ^
  --num-layers 1
