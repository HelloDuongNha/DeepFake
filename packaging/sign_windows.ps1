param(
  [Parameter(Mandatory=$true)][string]$CertificateThumbprint,
  [Parameter(Mandatory=$true)][string]$TimestampUrl
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..'))
$exe = 'dist\desktop-windows\DeepLiveCam\DeepLiveCam.exe'
$setup = 'dist\desktop-windows\DeepLiveCam-Setup-Windows.exe'
foreach ($file in @($exe, $setup)) {
  if (-not (Test-Path $file)) { throw "Missing $file" }
  & signtool.exe sign /sha1 $CertificateThumbprint /fd SHA256 /tr $TimestampUrl /td SHA256 $file
  if ($LASTEXITCODE -ne 0) { throw "Signing failed: $file" }
  & signtool.exe verify /pa /v $file
  if ($LASTEXITCODE -ne 0) { throw "Signature verification failed: $file" }
}
