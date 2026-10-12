Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..'))
$env:PYINSTALLER_CONFIG_DIR = (Join-Path (Get-Location) '.cache\pyinstaller')
New-Item -ItemType Directory -Force -Path $env:PYINSTALLER_CONFIG_DIR | Out-Null
if (-not (Get-Command py -ErrorAction SilentlyContinue)) { throw 'Install Python 3.13 and the py launcher.' }
if (-not (Test-Path '.build-venv\Scripts\python.exe')) { py -3.13 -m venv .build-venv }
$python = '.build-venv\Scripts\python.exe'
& $python -m pip install --upgrade pip
& $python -m pip install -r requirements.txt -r packaging\requirements-build.txt
if (-not $env:FFMPEG_EXE -or -not $env:FFPROBE_EXE) { throw 'Set FFMPEG_EXE and FFPROBE_EXE to Windows x64 binaries.' }
& $python packaging\make_icons.py
& $python -m PyInstaller --noconfirm --clean --distpath dist\desktop-windows --workpath build\desktop-windows packaging\DeepLiveCam.spec
& '.\dist\desktop-windows\DeepLiveCam\DeepLiveCam.exe' --package-smoke-test
if ($LASTEXITCODE -ne 0) { throw 'Package smoke test failed.' }
if ($env:DLC_SIGN_THUMBPRINT -and $env:DLC_TIMESTAMP_URL) {
  & signtool.exe sign /sha1 $env:DLC_SIGN_THUMBPRINT /fd SHA256 /tr $env:DLC_TIMESTAMP_URL /td SHA256 'dist\desktop-windows\DeepLiveCam\DeepLiveCam.exe'
  if ($LASTEXITCODE -ne 0) { throw 'EXE signing failed.' }
}
$iscc = Get-Command ISCC.exe -ErrorAction SilentlyContinue
if (-not $iscc) { throw 'Install Inno Setup 6 and add ISCC.exe to PATH.' }
& $iscc.Source packaging\DeepLiveCam.iss
if ($LASTEXITCODE -ne 0) { throw 'Inno Setup build failed.' }
if ($env:DLC_SIGN_THUMBPRINT -and $env:DLC_TIMESTAMP_URL) {
  & signtool.exe sign /sha1 $env:DLC_SIGN_THUMBPRINT /fd SHA256 /tr $env:DLC_TIMESTAMP_URL /td SHA256 'dist\desktop-windows\DeepLiveCam-Setup-Windows.exe'
  if ($LASTEXITCODE -ne 0) { throw 'Installer signing failed.' }
}
$portableZip = 'dist\desktop-windows\DeepLiveCam-Windows.zip'
if (Test-Path $portableZip) { Remove-Item $portableZip -Force }
Compress-Archive -Path 'dist\desktop-windows\DeepLiveCam', 'dist\desktop-windows\DeepLiveCam-Setup-Windows.exe' -DestinationPath $portableZip -CompressionLevel Optimal
Write-Host "Built $portableZip (portable app + installer)"
