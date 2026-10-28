<#
.SYNOPSIS
  Record one Kinect v2 depth take, and say GO only once frames are really landing.

.DESCRIPTION
  - Every take gets its own NEW folder; an existing folder is never reused.
  - Refuses to start while another program owns the sensor (KinectStream by default).
  - Starts KinectBurst in the take folder and waits for burst-ready.txt AND the
    first burst frame (f001.bin) before announcing GO: two beeps, a GO line, and
    capture-status.txt = "GO ..." (poll that file when driving this remotely).
  - Waits for the burst to finish, then reports frames recorded, the achieved
    frame rate and any sensor stalls from burst-log.csv; one long beep = done.

  capture-status.txt goes ARMING -> GO -> DONE (or FAIL <reason>).
  Exit codes: 0 done, 2 sensor busy, 3 folder exists, 4 no recorder,
  5 low disk, 6 sensor never became ready, 7 burst did not finish cleanly.

.EXAMPLE
  .\capture_take.ps1 -TakeDir D:\takes\luna-01
.EXAMPLE
  .\capture_take.ps1 -TakeDir D:\takes\michael-02 -Frames 400 -IntervalMs 28 -Quiet
#>
param(
    [Parameter(Mandatory = $true)][string]$TakeDir,
    [int]$Frames = 400,          # ~26 s at the default cadence
    [int]$IntervalMs = 28,       # measured: ~63 ms real interval (~15 fps), see KinectBurst.cs
    [string]$Exe = (Join-Path $PSScriptRoot 'KinectBurst.exe'),
    [string[]]$SensorOwners = @('KinectStream'),
    [int]$ReadyTimeoutSec = 45,
    [int]$MaxGapMs = 150,
    [switch]$Quiet
)
$ErrorActionPreference = 'Stop'
$status = $null

function Set-Status([string]$text) {
    if ($script:status) { Set-Content -LiteralPath $script:status -Value $text -Encoding ASCII }
    Write-Output $text
}
function Fail([string]$why, [int]$code) {
    if ($script:status) { Set-Status "FAIL $why" } else { Write-Output "FAIL $why" }
    exit $code
}
function Beep([int]$hz, [int]$ms) { if (-not $Quiet) { try { [console]::Beep($hz, $ms) } catch { } } }

if (Test-Path -LiteralPath $TakeDir) { Fail "$TakeDir already exists - every take gets a new folder" 3 }
foreach ($o in $SensorOwners) {
    $p = Get-Process -Name $o -ErrorAction SilentlyContinue
    if ($p) { Fail "$o (pid $(($p.Id) -join ',')) owns the sensor - close it first" 2 }
}
if (-not (Test-Path -LiteralPath $Exe)) { Fail "recorder not found at $Exe (run build_kinectburst.ps1)" 4 }

$full = [IO.Path]::GetFullPath($TakeDir)
$parent = Split-Path -Parent $full
if (-not (Test-Path -LiteralPath $parent)) { Fail "parent folder $parent does not exist" 3 }
$need = [int64]$Frames * 560000            # .bin 434 KB + .png ~110 KB, plus slack
$free = (Get-PSDrive -Name $full.Substring(0, 1)).Free
if ($free -lt $need) { Fail "only $([int]($free / 1MB)) MB free, need $([int]($need / 1MB)) MB" 5 }

New-Item -ItemType Directory -Path $full | Out-Null
$status = Join-Path $full 'capture-status.txt'
Set-Status "ARMING $Frames frames @ ${IntervalMs} ms"

$proc = Start-Process -FilePath $Exe -ArgumentList $Frames, $IntervalMs -WorkingDirectory $full -PassThru `
    -NoNewWindow -RedirectStandardOutput (Join-Path $full 'burst-stdout.txt') `
    -RedirectStandardError (Join-Path $full 'burst-stderr.txt')

$deadline = (Get-Date).AddSeconds($ReadyTimeoutSec)
$ready = Join-Path $full 'burst-ready.txt'
$first = Join-Path $full 'f001.bin'
while (-not ((Test-Path -LiteralPath $ready) -and (Test-Path -LiteralPath $first))) {
    if ($proc.HasExited) {
        $out = (Get-Content -LiteralPath (Join-Path $full 'burst-stdout.txt') -ErrorAction SilentlyContinue) -join ' '
        Fail "recorder exited before frames landed (exit $($proc.ExitCode)): $out" 6
    }
    if ((Get-Date) -gt $deadline) {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue   # only the recorder we started
        Fail "no depth frames after $ReadyTimeoutSec s - is the sensor plugged in and free?" 6
    }
    Start-Sleep -Milliseconds 50
}
Beep 880 150; Beep 880 150
Set-Status "GO $(Get-Date -Format o) - recording $Frames frames (~$([int]($Frames * 0.063)) s), cue the performer"

$proc.WaitForExit()
$log = Join-Path $full 'burst-log.csv'
$rows = @(Import-Csv -LiteralPath $log)
$ts = @($rows | Select-Object -Skip 1 | ForEach-Object { [int64]$_.unix_ms })
$summary = "recorded $($rows.Count - 1)/$Frames"
if ($ts.Count -ge 2) {
    $span = $ts[-1] - $ts[0]
    $gaps = @(for ($i = 1; $i -lt $ts.Count; $i++) { $d = $ts[$i] - $ts[$i - 1]; if ($d -gt $MaxGapMs) { $d } })
    $summary += " in $([math]::Round($span / 1000, 1)) s = $([math]::Round(1000 * ($ts.Count - 1) / [math]::Max($span, 1), 1)) fps"
    $summary += "; stalls>${MaxGapMs}ms: " + $(if ($gaps.Count) { ($gaps -join ',') + ' ms' } else { 'none' })
}
Beep 660 500
if (-not (Test-Path -LiteralPath (Join-Path $full 'burst-done.txt'))) { Fail "burst did not finish cleanly: $summary" 7 }
Set-Status "DONE $summary"
exit 0
