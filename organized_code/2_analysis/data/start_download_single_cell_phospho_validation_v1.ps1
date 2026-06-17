$ErrorActionPreference = "Stop"
$taskName = "SCP682_SC_PHOSPHO_VALIDATION_DOWNLOAD_20260518"
$cmdPath = "D:\data\lsy\vm_lsy_parent\lsy\03_code\single_cell\data_download\run_download_single_cell_phospho_validation_v1.cmd"
$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$cmdPath`""
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1)
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Force | Out-Null
Start-ScheduledTask -TaskName $taskName
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName, State
