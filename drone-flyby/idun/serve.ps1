param(
    [Parameter(Mandatory)][ValidatePattern('^[a-zA-Z0-9][a-zA-Z0-9_-]+$')]
    [string]$Experiment,
    [Parameter(Mandatory)][ValidatePattern('^/[a-zA-Z0-9_./-]+$')]
    [string]$WeightsPath,
    [Parameter(Mandatory)][ValidatePattern('^[a-fA-F0-9]{64}$')]
    [string]$WeightsSha256,
    [ValidateRange(1024, 65535)][int]$Port = 9052,
    [string]$Remote = 'idun',
    [string]$RemoteRoot = 'nordic-cup/drone-score-loop',
    [string]$Environment = 'env-v2',
    [string]$Account = 'share-ie-idi'
)

$ErrorActionPreference = 'Stop'
foreach ($value in @($Remote, $RemoteRoot, $Environment, $Account)) {
    if ($value -notmatch '^[a-zA-Z0-9_./@-]+$' -or $value.Contains('..') -or $value.StartsWith('/')) {
        throw "Unsafe remote identifier: $value"
    }
}
if ($WeightsPath.Contains('..')) { throw 'WeightsPath must not contain parent traversal' }

$root = Split-Path $PSScriptRoot -Parent
& (Join-Path $PSScriptRoot 'submit.ps1') -Action sync -Experiment $Experiment `
    -Remote $Remote -RemoteRoot $RemoteRoot -Environment $Environment -Account $Account

$localRun = Join-Path $root "runs\idun\$Experiment"
$remoteDir = "$RemoteRoot/experiments/$Experiment"
$sshOptions = @(
    '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
    '-o', 'StrictHostKeyChecking=yes', '-o', 'ServerAliveInterval=15',
    '-o', 'ServerAliveCountMax=3', '-tt'
)
$command = "set -euo pipefail; cd '$remoteDir'; mkdir .serving; " +
    "export DRONE_ENV_PATH=`"`$HOME/$RemoteRoot/$Environment`"; " +
    "export SLURM_SUBMIT_DIR=`"`$PWD`"; " +
    "exec srun --unbuffered --account='$Account' --job-name=drone-serve$Port " +
    "--partition=GPUQ --gres=gpu:1 --constraint='gpu80g&(a100|h100)' " +
    "--nodes=1 --ntasks=1 --cpus-per-task=8 --mem=48G --time=12:00:00 " +
    "--kill-on-bad-exit=1 --chdir=`"`$PWD`" bash idun/serve.sh " +
    "--weights '$WeightsPath' --sha256 '$($WeightsSha256.ToLowerInvariant())' --port $Port"

@{
    experiment = $Experiment
    remote = $Remote
    remote_directory = $remoteDir
    weights = $WeightsPath
    checkpoint_sha256 = $WeightsSha256.ToLowerInvariant()
    port = $Port
    lifetime = 'Attached SSH/srun session; maximum 12 hours'
    started_at = [DateTimeOffset]::UtcNow.ToString('o')
} | ConvertTo-Json | Set-Content (Join-Path $localRun 'launch.json') -Encoding utf8NoBOM

Write-Output 'Serving is attached to this command. Disconnecting stops the owned allocation.'
& ssh @sshOptions $Remote $command 2>&1 | Tee-Object -FilePath (Join-Path $localRun 'session.log')
if ($LASTEXITCODE -ne 0) {
    throw "Serving SSH session ended with exit code $LASTEXITCODE. Inspect $localRun\session.log"
}
