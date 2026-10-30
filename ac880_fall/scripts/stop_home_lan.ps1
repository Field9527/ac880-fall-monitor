$ErrorActionPreference='Stop'
$root=Split-Path $PSScriptRoot -Parent
$pidPath=Join-Path $root 'out/web_20261018/lan_proxy.pid'
if(Test-Path -LiteralPath $pidPath){
 $relayPid=[int](Get-Content -LiteralPath $pidPath)
 $relayProcess=Get-CimInstance Win32_Process -Filter "ProcessId=$relayPid" -ErrorAction SilentlyContinue
 if($relayProcess){
  if($relayProcess.CommandLine -notlike '*scripts*lan_*.py*'){throw '记录的PID已不属于本项目转发程序。'}
  Stop-Process -Id $relayPid
 }
 Write-Host '本项目局域网转发已停止。开发板识别服务不受影响。'
}
