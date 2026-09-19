param(
    [Parameter(Mandatory)][ValidateSet('sync', 'setup', 'test', 'run', 'status', 'fetch')]
    [string]$Action,
    [Parameter(Mandatory)][ValidatePattern('^[a-zA-Z0-9][a-zA-Z0-9_-]+$')]
    [string]$Experiment,
    [string]$RunCommand = '',
    [string]$Remote = 'idun',
    [string]$RemoteRoot = 'nordic-cup/drone-score-loop',
    [string]$Environment = 'env-v1',
    [string]$Account = 'share-ie-idi'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
foreach ($value in @($Remote, $RemoteRoot, $Environment, $Account)) {
    if ($value -notmatch '^[a-zA-Z0-9_./@-]+$' -or $value.Contains('..') -or $value.StartsWith('/')) {
        throw "Unsafe remote identifier: $value"
    }
}
$remoteDir = "$RemoteRoot/experiments/$Experiment"
$envDir = "`$HOME/$RemoteRoot/$Environment"
$localRun = Join-Path $root "runs\idun\$Experiment"
$sshOptions = @('-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'StrictHostKeyChecking=yes')

function Invoke-Remote([string]$Command) {
    $result = & ssh @sshOptions $Remote "set -euo pipefail; $Command"
    if ($LASTEXITCODE -ne 0) { throw "Remote command failed ($LASTEXITCODE): $Command" }
    return $result
}

if ($Action -eq 'sync') {
    New-Item -ItemType Directory -Force $localRun | Out-Null
    $manifestPath = Join-Path $localRun 'snapshot.json'
    if (Test-Path $manifestPath) { throw "Snapshot already exists: $Experiment. Choose a new experiment ID." }
    $files = @()
    foreach ($directory in @('src', 'idun', 'data\helsinki', 'weights')) {
        $path = Join-Path $root $directory
        if (Test-Path $path) {
            $files += Get-ChildItem -LiteralPath $path -Recurse -File |
                Where-Object { $_.FullName -notmatch '[\\/](?:__pycache__|\.pytest_cache)[\\/]' }
        }
    }
    foreach ($name in @('requirements.txt', 'pytest.ini', 'profiles\champion.json',
                        'research\foreground_review.json', 'research\background_sources.json')) {
        $path = Join-Path $root $name
        if (Test-Path $path) { $files += Get-Item $path }
    }
    $entries = @($files | Sort-Object FullName | ForEach-Object {
        $hash = if ($_.Extension -in '.sh', '.slurm') {
            $bytes = [Text.Encoding]::UTF8.GetBytes([IO.File]::ReadAllText($_.FullName).Replace("`r`n", "`n"))
            [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($bytes)).ToLowerInvariant()
        } else {
            (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        }
        @{
            path = [IO.Path]::GetRelativePath($root, $_.FullName).Replace('\', '/')
            sha256 = $hash
        }
    })
    $manifest = @{
        experiment = $Experiment
        commit = (& git -C $root rev-parse HEAD)
        files = $entries
        data_role = 'training-development'
    }
    $manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifestPath -Encoding utf8NoBOM
    $listPath = Join-Path $localRun 'upload-files.txt'
    $entries.path | Set-Content -LiteralPath $listPath -Encoding utf8NoBOM
    $archive = Join-Path $localRun 'snapshot.tar.gz'
    & tar -czf $archive -C $root -T $listPath
    if ($LASTEXITCODE -ne 0) { throw 'Snapshot archiving failed' }
    Invoke-Remote "mkdir -p $RemoteRoot/experiments; mkdir $remoteDir" | Out-Null
    & scp @sshOptions $archive $manifestPath "${Remote}:$remoteDir/"
    if ($LASTEXITCODE -ne 0) { throw 'Snapshot upload failed' }
    Invoke-Remote "cd $remoteDir && tar -xzf snapshot.tar.gz && sed -i 's/\r`$//' idun/*.sh idun/*.slurm && for script in idun/*.sh idun/*.slurm; do bash -n `"`$script`"; done && mkdir -p logs runs && chmod -R a-w src idun requirements.txt pytest.ini && rm snapshot.tar.gz" | Out-Null
    Remove-Item -LiteralPath $archive, $listPath
    Write-Output "Snapshot ready: ${Remote}:$remoteDir"
} elseif ($Action -eq 'setup') {
    Invoke-Remote "cd $remoteDir && export DRONE_ENV_PATH=$envDir && bash idun/setup_env.sh"
} elseif ($Action -in @('run', 'test')) {
    if ($Action -eq 'run' -and -not $RunCommand.Trim()) { throw 'run requires -RunCommand' }
    $job = if ($Action -eq 'test') { 'job_test.slurm' } else { 'job.slurm' }
    $encodedCommand = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($RunCommand))
    # The remote mkdir is a durable idempotency claim, not a process-local check.
    $jobId = Invoke-Remote "cd $remoteDir && mkdir .submitted && export DRONE_ENV_PATH=$envDir && export RUN_CMD=`"`$(printf '%s' '$encodedCommand' | base64 -d)`" && sbatch --parsable --export=ALL --account=$Account idun/$job | tee job.id"
    New-Item -ItemType Directory -Force $localRun | Out-Null
    $jobId | Set-Content (Join-Path $localRun 'job.id')
    Write-Output "Submitted $Experiment as job $jobId"
} elseif ($Action -eq 'status') {
    Invoke-Remote "cd $remoteDir && if [ -f job.id ]; then job=`$(cut -d';' -f1 job.id); squeue -j `"`$job`" -o '%.12i %.8T %.12M %.30R'; sacct -j `"`$job`" --format=JobID,State,ExitCode,Elapsed,MaxRSS --noheader; else echo 'Not submitted'; fi"
} elseif ($Action -eq 'fetch') {
    New-Item -ItemType Directory -Force $localRun | Out-Null
    & scp @sshOptions -r "${Remote}:$remoteDir/runs" "${Remote}:$remoteDir/logs" $localRun
    if ($LASTEXITCODE -ne 0) { throw 'Artifact fetch failed' }
}
