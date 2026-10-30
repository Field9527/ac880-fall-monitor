param()
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$cloudOutput = 'E:\Zynq_project\ac880_fall\out\cloud_20261029\hongkong'
$env:TEMP = 'E:\Zynq_cache\temp'
$env:TMP = 'E:\Zynq_cache\temp'
New-Item -ItemType Directory -Force -Path $cloudOutput | Out-Null

function Get-RecordedProcess([string]$RecordPath, [string]$CommandPart) {
    if (!(Test-Path -LiteralPath $RecordPath)) { return $null }
    $candidateText = (Get-Content -LiteralPath $RecordPath -Raw).Trim()
    if ($candidateText -notmatch '^\d+$') { return $null }
    $candidateProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$candidateText" -ErrorAction SilentlyContinue
    if ($candidateProcess -and $candidateProcess.CommandLine -like ('*' + $CommandPart + '*')) {
        return $candidateProcess
    }
    return $null
}

& (Join-Path $PSScriptRoot 'start_home_lan.ps1')

$listener = Get-NetTCPConnection -LocalPort 8089 -State Listen -ErrorAction SilentlyContinue
if ($listener) {
    $gatewayOwner = Get-CimInstance Win32_Process -Filter "ProcessId=$($listener[0].OwningProcess)"
    if ($gatewayOwner.CommandLine -notlike '*lan_gateway.py*') { throw 'Port 8089 belongs to another application.' }
    Write-Host 'Web gateway is already running.'
} else {
    $gatewayArguments = @((Join-Path $PSScriptRoot 'lan_gateway.py'), '--host', '127.0.0.1', '--port', '8089', '--lan-port', '8088', '--fps', '15', '--jpeg-quality', '75', '--max-width', '640', '--data', 'E:\Zynq_project\ac880_fall\data\home_events')
    $gatewayProcess = Start-Process -FilePath 'E:\Anaconda\envs\DL\python.exe' -ArgumentList $gatewayArguments -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $cloudOutput 'gateway.log') -RedirectStandardError (Join-Path $cloudOutput 'gateway_error.log')
    Set-Content -LiteralPath (Join-Path $cloudOutput 'gateway.pid') -Value $gatewayProcess.Id
}

$gatewayReady = $false
for ($attempt = 0; $attempt -lt 12; $attempt++) {
    $probeClient = [System.Net.WebClient]::new()
    $probeClient.Proxy = $null
    try {
        $health = $probeClient.DownloadString('http://127.0.0.1:8089/healthz') | ConvertFrom-Json
        if ($health.ok) { $gatewayReady = $true; break }
    } catch { Start-Sleep -Milliseconds 500 }
    finally { $probeClient.Dispose() }
}
if (!$gatewayReady) { throw ('Gateway did not become ready. Check ' + $cloudOutput + '\gateway_error.log') }

$supervisor = Get-RecordedProcess (Join-Path $cloudOutput 'tunnel_supervisor.pid') 'start_cloud_tunnel.py'
if ($supervisor) {
    Write-Host 'Cloud tunnel supervisor is already running.'
} else {
    $tunnelArguments = @((Join-Path $PSScriptRoot 'start_cloud_tunnel.py'), '--jump-host', '182.92.239.147', '--output', $cloudOutput)
    $tunnelProcess = Start-Process -FilePath 'E:\Anaconda\python.exe' -ArgumentList $tunnelArguments -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $cloudOutput 'tunnel.log') -RedirectStandardError (Join-Path $cloudOutput 'tunnel_error.log')
    Write-Host ('Cloud tunnel supervisor started: ' + $tunnelProcess.Id)
}

Write-Host 'Website: https://shuaidao.online'
Write-Host 'Login details: E:\Zynq_keys\aliyun\shuaidao_login.txt'
Write-Host 'Keep this computer and the board powered on and connected.'
Write-Host 'The board detector uses its existing manual run.sh launcher.'
