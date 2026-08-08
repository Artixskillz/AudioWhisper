; AudioWhisper Inno Setup Script
; Requires Inno Setup 6.3+ (https://jrsoftware.org/isinfo.php)
; Built by build.py, which passes /DMyAppVersion=x.y.z

#define MyAppName "AudioWhisper"
#ifndef MyAppVersion
  #define MyAppVersion "1.1.0"
#endif
#define MyAppPublisher "Artixskillz"
#define MyAppURL "https://github.com/Artixskillz/AudioWhisper"
#define MyAppExeName "AudioWhisper.exe"

[Setup]
AppId={{A8D1F2E3-4B5C-6D7E-8F9A-0B1C2D3E4F5A}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=installer_output
OutputBaseFilename=AudioWhisper_Setup_{#MyAppVersion}
SetupIconFile=AudioWhisper.ico
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[InstallDelete]
; Upgrades must not merge stale files from the previous version into the
; new trees. Removing runtime also drops the optional GPU add-on - the app
; detects that and offers to re-enable it (models/settings are elsewhere).
Type: filesandordirs; Name: "{app}\_internal"
Type: filesandordirs; Name: "{app}\runtime"

[Files]
; The complete app: GUI, its libraries, and the bundled Python runtime
; with the transcription engine preinstalled. Nothing downloads at install
; time - only Whisper models download on first use, inside the app.
Source: "dist\AudioWhisper\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[UninstallDelete]
; The uninstaller only removes files it installed - these trees also gain
; files at runtime (GPU add-on wheels, .pyc caches) that must go too.
; No user data lives here; models and settings are under {localappdata}.
Type: filesandordirs; Name: "{app}\_internal"
Type: filesandordirs; Name: "{app}\runtime"

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent

[Code]
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
  begin
    // Models and settings live in {localappdata}\AudioWhisper (v1.1+)
    // and possibly {userappdata}\AudioWhisper (v1.0). Ask before removing.
    // Silent uninstalls never prompt and never delete user data.
    if (not UninstallSilent) and
       (DirExists(ExpandConstant('{localappdata}\{#MyAppName}')) or
        DirExists(ExpandConstant('{userappdata}\{#MyAppName}'))) then
    begin
      if MsgBox('Do you also want to remove downloaded models and settings?'#13#10 +
                'This frees up disk space but they''ll need to be re-downloaded if you reinstall.',
                mbConfirmation, MB_YESNO) = IDYES then
      begin
        DelTree(ExpandConstant('{localappdata}\{#MyAppName}'), True, True, True);
        DelTree(ExpandConstant('{userappdata}\{#MyAppName}'), True, True, True);
      end;
    end;
  end;
end;
