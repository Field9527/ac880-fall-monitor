param()
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$out = Join-Path $root 'out/web_gateway'
$cloudOut = 'E:\Zynq_project\ac880_fall\out\cloud_20261029\hongkong'
$env:TEMP = 'E:\Zynq_cache\temp'
$env:TMP = 'E:\Zynq_cache\temp'
New-Item -ItemType Directory -Force -Path $out,$cloudOut | Out-Null
$listeners = @(Get-NetTCPConnection -LocalPort 8088,8089 -State Listen -ErrorAction SilentlyContinue)
if ($listeners.Count) {
    $owners = @($listeners.OwningProcess | Sort-Object -Unique)
    if ($owners.Count -ne 1 -or @($listeners.LocalPort | Sort-Object -Unique).Count -ne 2) {
        throw '网页端口被不同程序占用，不能重复启动网关。'
    }
    $gatewayProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$($owners[0])"
    if ($gatewayProcess.CommandLine -notlike '*ac880-fall-monitor*scripts*lan_gateway.py*') {
        throw '网页端口不属于当前工程，不能覆盖其他程序。'
    }
    $gatewayId = $gatewayProcess.ProcessId
    Write-Host '共享网页网关已在运行。'
} else {
    $arguments = @((Join-Path $PSScriptRoot 'lan_gateway.py'), '--host', '127.0.0.1', '--port', '8089', '--lan-port', '8088', '--fps', '15', '--jpeg-quality', '75', '--max-width', '640', '--data', 'E:\Zynq_project\ac880_fall\data\home_events')
    $gatewayProcess = Start-Process -FilePath 'E:\Anaconda\envs\DL\python.exe' -ArgumentList $arguments -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $out 'gateway.log') -RedirectStandardError (Join-Path $out 'gateway_error.log')
    $gatewayId = $gatewayProcess.Id
    Write-Host '共享网页网关已启动。'
}
Set-Content -LiteralPath (Join-Path $out 'gateway.pid') -Value $gatewayId
Set-Content -LiteralPath (Join-Path $cloudOut 'gateway.pid') -Value $gatewayId
$ready = $false
for ($attempt=0; $attempt -lt 15; $attempt++) {
    $client = [System.Net.WebClient]::new()
    $client.Proxy = $null
    try {
        $health = $client.DownloadString('http://127.0.0.1:8088/healthz') | ConvertFrom-Json
        if ($health.ok) {$ready=$true;break}
    } catch {Start-Sleep -Milliseconds 400}
    finally {$client.Dispose()}
}
if (!$ready) {throw ('网关未就绪，请查看 '+$out+'\gateway_error.log')}
Write-Host '本机访问：http://127.0.0.1:8088'
Get-NetIPAddress -AddressFamily IPv4 | Where-Object {$_.InterfaceAlias -match 'WLAN|Wi-Fi|无线' -and $_.IPAddress -notlike '169.254*'} | ForEach-Object {Write-Host ('手机访问：http://'+$_.IPAddress+':8088')}
Write-Host '局域网与远程共用一次视频接收和编码；开发板仍需手动运行 run.sh。'
