$taskName = "SCP682_SC12_SCTRANSLATOR_CACHE_20260621"
$bat = "D:\data\lsy\vm_lsy_parent\lsy\03_code\single_cell\modeling\run_sctranslator_sc12_cache_all.bat"
$action = "cmd.exe /c `"$bat`""
schtasks /Create /TN $taskName /SC ONCE /ST 23:59 /TR $action /F | Out-Host
schtasks /Run /TN $taskName | Out-Host
schtasks /Query /TN $taskName /V /FO LIST | Out-Host
