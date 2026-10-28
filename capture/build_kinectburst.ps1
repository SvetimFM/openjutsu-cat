# Build KinectBurst.exe next to this script. Needs the Kinect for Windows SDK 2.0
# (Microsoft.Kinect.dll) and the .NET Framework 4 C# compiler that ships with Windows.
param(
    [string]$KinectDll = 'C:\Program Files\Microsoft SDKs\Kinect\v2.0_1409\Assemblies\Microsoft.Kinect.dll',
    [string]$Csc = 'C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe'
)
$ErrorActionPreference = 'Stop'
foreach ($f in $KinectDll, $Csc) { if (-not (Test-Path -LiteralPath $f)) { throw "not found: $f" } }
$out = Join-Path $PSScriptRoot 'KinectBurst.exe'
& $Csc /nologo /target:exe "/out:$out" "/reference:$KinectDll" /reference:System.Drawing.dll (Join-Path $PSScriptRoot 'KinectBurst.cs')
if ($LASTEXITCODE -ne 0) { throw "csc failed ($LASTEXITCODE)" }
"built $out ($((Get-Item $out).Length) bytes)"
