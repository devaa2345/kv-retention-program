param(
    [Parameter(Mandatory=$true)][ValidateSet('M2','M3')][string]$Model,
    [Parameter(Mandatory=$true)][ValidatePattern('^[A-Za-z0-9_.-]+\.jsonl$')][string]$DataName,
    [Parameter(Mandatory=$true)][ValidatePattern('^[A-Za-z0-9_.-]+\.jsonl$')][string]$OutName
)

$ErrorActionPreference = 'Stop'
$outPath = Join-Path $PSScriptRoot $OutName
$sidecar = [System.IO.Path]::ChangeExtension($outPath, '.outer.json')
if ((Test-Path -LiteralPath $outPath) -or (Test-Path -LiteralPath $sidecar)) {
    throw "Output exists: $outPath or $sidecar"
}
$startUtc = [DateTimeOffset]::UtcNow
$timer = [System.Diagnostics.Stopwatch]::StartNew()
$command = "cd /mnt/d/INNOCREW/Blockage/paper4 && HF_HOME=/mnt/d/hf_cache PYTHONPATH=/mnt/d/INNOCREW/Blockage/paper4:/mnt/d/INNOCREW/Blockage/paper3:/mnt/d/INNOCREW/Blockage/paper2 TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_DISABLE=1 /opt/p2venv/bin/python out/_realtext_tight_latency_pilot.py --model $Model --data out/$DataName --out out/$OutName"
& wsl -d Ubuntu-24.04 -e bash -lc $command
$exitCode = $LASTEXITCODE
$timer.Stop()
$endUtc = [DateTimeOffset]::UtcNow
if ($exitCode -ne 0) { throw "WSL benchmark exited with $exitCode" }
$firstLine = Get-Content -LiteralPath $outPath -TotalCount 1 | ConvertFrom-Json
$pythonStart = [double]$firstLine.meta.python_process_start_unix
$startUnix = $startUtc.ToUnixTimeMilliseconds() / 1000.0
$record = [ordered]@{
    model = $Model
    data = $DataName
    output = $OutName
    outer_start_utc = $startUtc.ToString('o')
    outer_end_utc = $endUtc.ToString('o')
    outer_wall_sec = $timer.Elapsed.TotalSeconds
    wsl_to_python_process_start_sec = $pythonStart - $startUnix
    python_job_wall_sec = [double]$firstLine.meta.job_wall
    outer_minus_python_job_sec = $timer.Elapsed.TotalSeconds - [double]$firstLine.meta.job_wall
}
$record | ConvertTo-Json | Set-Content -LiteralPath $sidecar -Encoding utf8
$record | ConvertTo-Json
