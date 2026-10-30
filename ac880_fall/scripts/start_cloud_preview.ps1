param(
    [string]$CloudHost = '182.92.239.147',
    [string]$CloudUser = 'ecs-user',
    [string]$KeyPath = 'E:\Zynq_keys\aliyun\ac880_home.pem',
    [string]$KnownHostsPath = 'E:\Zynq_keys\aliyun\known_hosts',
    [ValidateRange(1024,65535)][int]$LocalGatewayPort = 8089,
    [ValidateRange(1024,65535)][int]$LocalPreviewPort = 18089
)
$ErrorActionPreference = 'Stop'
if (!(Test-Path -LiteralPath $KeyPath -PathType Leaf)) { throw 'Private key not found.' }
if (!(Test-Path -LiteralPath $KnownHostsPath -PathType Leaf)) { throw 'Verify the cloud SSH host first.' }
if ($LocalGatewayPort -eq $LocalPreviewPort) { throw 'Gateway and preview ports must be different.' }
$gatewayUrl = 'http://127.0.0.1:' + $LocalGatewayPort + '/healthz'
$probe = [System.Net.WebClient]::new()
$probe.Proxy = $null
try { $null = $probe.DownloadString($gatewayUrl) }
finally { $probe.Dispose() }

Write-Host ('Private preview: http://127.0.0.1:' + $LocalPreviewPort)
Write-Host 'Login details: E:\Zynq_keys\aliyun\home_web_preview.json'
Write-Host 'Keep this window open. Ctrl+C disconnects; this does not stop the board.'
$sshArguments = @(
    '-N', '-T', '-i', $KeyPath,
    '-o', ('UserKnownHostsFile=' + $KnownHostsPath.Replace('\','/')),
    '-o', 'StrictHostKeyChecking=yes', '-o', 'BatchMode=yes',
    '-o', 'IdentitiesOnly=yes', '-o', 'ExitOnForwardFailure=yes',
    '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3',
    '-o', 'ConnectTimeout=10',
    '-R', ('127.0.0.1:18088:127.0.0.1:' + $LocalGatewayPort),
    '-L', ('127.0.0.1:' + $LocalPreviewPort + ':127.0.0.1:18089'),
    ($CloudUser + '@' + $CloudHost)
)
& ssh.exe @sshArguments
if ($LASTEXITCODE -ne 0) { throw ('SSH tunnel exited with code ' + $LASTEXITCODE) }
