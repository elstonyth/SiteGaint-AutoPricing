' SiteGiant Pricing Automation Launcher
' Double-click to start the application

Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

' Get the script directory
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)

' Change to script directory
WshShell.CurrentDirectory = scriptDir

' Check if venv exists
If Not fso.FileExists(scriptDir & "\.venv\Scripts\python.exe") Then
    MsgBox "Virtual environment not found!" & vbCrLf & vbCrLf & _
           "Please run these commands first:" & vbCrLf & _
           "  python -m venv .venv" & vbCrLf & _
           "  .venv\Scripts\pip install -r requirements.txt", _
           vbCritical, "SiteGiant Pricing - Error"
    WScript.Quit 1
End If

' Open browser after delay
WshShell.Run "cmd /c timeout /t 3 /nobreak >nul && start http://127.0.0.1:8000", 0, False

' Start the server (visible window)
WshShell.Run "cmd /k cd /d """ & scriptDir & """ && echo Starting SiteGiant Pricing Automation... && echo. && .venv\Scripts\python.exe -m uvicorn src.webapp.main:app --host 127.0.0.1 --port 8000", 1, False
