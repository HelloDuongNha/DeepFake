#define AppName "Deep Live Cam"
#define AppVersion "1.0.0"
[Setup]
AppId={{D21CEB68-6A31-469C-9DEE-9C7CA79AF167}
AppName={#AppName}
AppVersion={#AppVersion}
DefaultDirName={autopf}\DeepLiveCam
DefaultGroupName={#AppName}
OutputDir=..\dist\desktop-windows
OutputBaseFilename=DeepLiveCam-Setup-Windows
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
WizardStyle=modern
SetupIconFile=DeepLiveCam.ico
UninstallDisplayIcon={app}\DeepLiveCam.exe
PrivilegesRequired=admin
[Files]
Source: "..\dist\desktop-windows\DeepLiveCam\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion
[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\DeepLiveCam.exe"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\DeepLiveCam.exe"; Tasks: desktopicon
[Tasks]
Name: desktopicon; Description: "Create a desktop icon"; GroupDescription: "Additional icons:"
[Run]
Filename: "{app}\DeepLiveCam.exe"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent
