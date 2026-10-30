param([Parameter(Position=0,ValueFromRemainingArguments=$true)][string[]]$CommandArgs)
$cacheRoot = 'E:\Zynq_cache'
$cachePaths = @{
    TORCH_HOME='torch'; TORCH_EXTENSIONS_DIR='torch_extensions'
    HF_HOME='huggingface'; HF_HUB_CACHE='huggingface\hub'; HUGGINGFACE_HUB_CACHE='huggingface\hub'
    PIP_CACHE_DIR='pip'; CUDA_CACHE_PATH='cuda'; XDG_CACHE_HOME='xdg'
    MPLCONFIGDIR='matplotlib'; NUMBA_CACHE_DIR='numba'; TRITON_CACHE_DIR='triton'
    JOBLIB_TEMP_FOLDER='temp'; TEMP='temp'; TMP='temp'
}
foreach ($key in $cachePaths.Keys) {
    $value = Join-Path $cacheRoot $cachePaths[$key]
    New-Item -ItemType Directory -Force -Path $value | Out-Null
    [Environment]::SetEnvironmentVariable($key,$value,'Process')
}
if (-not $CommandArgs.Count) { Write-Output "Cache and temp directories configured: $cacheRoot"; exit 0 }
$executable = $CommandArgs[0]
[string[]]$remaining = @()
if ($CommandArgs.Count -gt 1) { $remaining = @($CommandArgs | Select-Object -Skip 1) }
& $executable @remaining
exit $LASTEXITCODE
