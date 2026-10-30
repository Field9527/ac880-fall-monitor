﻿$ErrorActionPreference='Stop'
$root=Split-Path $PSScriptRoot -Parent
$pidPath=Join-Path $root 'out/web_gateway/gateway.pid'
if(Test-Path -LiteralPath $pidPath){
 $relayPid=[int](Get-Content -LiteralPath $pidPath)
 $relayProcess=Get-CimInstance Win32_Process -Filter "ProcessId=$relayPid" -ErrorAction SilentlyContinue
 if($relayProcess){
  if($relayProcess.CommandLine -notlike '*scripts*lan_*.py*'){throw '记录的PID已不属于本项目转发程序。'}
  Stop-Process -Id $relayPid
 }
 Write-Host '共享网页网关已停止，局域网和远程转发均已断开。开发板识别继续运行。'
}
