$ErrorActionPreference='Stop'
$root=(Split-Path $PSScriptRoot -Parent)
$out=Join-Path $root 'out/web_20261018'
$env:TEMP='E:\Zynq_cache\temp'
$env:TMP='E:\Zynq_cache\temp'
New-Item -ItemType Directory -Force -Path $out | Out-Null
$listener=Get-NetTCPConnection -LocalPort 8088 -State Listen -ErrorAction SilentlyContinue
if($listener){
 $process=Get-CimInstance Win32_Process -Filter "ProcessId=$($listener[0].OwningProcess)"
 if($process.CommandLine -notlike '*scripts*lan_gateway.py*'){throw '8088端口被其他程序占用。'}
 Write-Host '局域网转发已在运行。'
} else {
 $arguments=@((Join-Path $PSScriptRoot 'lan_gateway.py'),'--host','0.0.0.0','--port','8088')
 $process=Start-Process -FilePath 'E:\Anaconda\envs\DL\python.exe' -ArgumentList $arguments -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $out 'lan_proxy.log') -RedirectStandardError (Join-Path $out 'lan_proxy_error.log')
 Set-Content -LiteralPath (Join-Path $out 'lan_proxy.pid') -Value $process.Id
 Write-Host '局域网转发已启动。'
}
Get-NetIPAddress -AddressFamily IPv4 | Where-Object {$_.InterfaceAlias -match 'WLAN|Wi-Fi|无线' -and $_.IPAddress -notlike '169.254*'} | ForEach-Object {Write-Host ('手机浏览器访问: http://'+$_.IPAddress+':8088')}
try {
 $reply=Invoke-RestMethod -Uri 'http://127.0.0.1:8088/api/state' -TimeoutSec 5
 if($reply.detector.running -and $reply.detector.ready){Write-Host '开发板识别已连接，网页已可用。'}
 else {Write-Host '网页服务已启动；开发板识别尚未就绪。SSH中执行：cd /home/xilinx/fall_gesture_trial，然后 ./run.sh'}
} catch {Write-Host '网页服务正在启动；稍后刷新。若仍打不开，请检查8088端口及电脑Wi-Fi。'}
Write-Host '手机与电脑连接同一Wi-Fi/热点。开发板继续网线直连电脑。'
