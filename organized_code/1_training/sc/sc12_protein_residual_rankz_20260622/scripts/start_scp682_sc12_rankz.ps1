$bat = "D:\data\lsy\vm_lsy_parent\lsy\03_code\single_cell\modeling\run_scp682_sc12_protein_residual_rankz_logged.bat"
$p = Start-Process -FilePath "cmd.exe" -ArgumentList @("/c", $bat) -WindowStyle Hidden -PassThru
$pidPath = "D:\data\lsy\vm_lsy_parent\lsy\02_results\single_cell\20260622_scp682_sc12_protein_residual_rankz_v1\logs\formal_launcher_pid.txt"
$p.Id | Out-File -Encoding ascii $pidPath
Write-Output $p.Id
