$Root = "D:\data\lsy\vm_lsy_parent\lsy"
$Gpu1Bat = Join-Path $Root "03_code\single_cell\modeling\run_sctranslator_sc12_all_protein_cache_iccite_gpu1.bat"
$Gpu0Bat = Join-Path $Root "03_code\single_cell\modeling\run_sctranslator_sc12_all_protein_cache_remaining_gpu0.bat"

Start-Process -FilePath "cmd.exe" -ArgumentList "/c `"$Gpu1Bat`"" -WindowStyle Hidden
Start-Process -FilePath "cmd.exe" -ArgumentList "/c `"$Gpu0Bat`"" -WindowStyle Hidden

Write-Host "started SC12 all-protein cache jobs"
