# Create Desktop Shortcut for SiteGiant Pricing Automation
# Run this script once to create the shortcut

$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$shortcutPath = [Environment]::GetFolderPath("Desktop") + "\SiteGiant Pricing.lnk"
$targetPath = "$scriptPath\SiteGiant Pricing.vbs"

# Create WScript.Shell COM object
$WshShell = New-Object -ComObject WScript.Shell

# Create shortcut
$shortcut = $WshShell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $targetPath
$shortcut.WorkingDirectory = $scriptPath
$shortcut.Description = "SiteGiant Pricing Automation Tool"
$shortcut.IconLocation = "shell32.dll,21"  # Dollar/money icon
$shortcut.Save()

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Desktop shortcut created successfully!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Location: $shortcutPath" -ForegroundColor Yellow
Write-Host ""
Write-Host "  Double-click 'SiteGiant Pricing' on your desktop to launch!"
Write-Host ""

# Also create a Start Menu shortcut
$startMenuPath = [Environment]::GetFolderPath("StartMenu") + "\Programs\SiteGiant Pricing.lnk"
$shortcut2 = $WshShell.CreateShortcut($startMenuPath)
$shortcut2.TargetPath = $targetPath
$shortcut2.WorkingDirectory = $scriptPath
$shortcut2.Description = "SiteGiant Pricing Automation Tool"
$shortcut2.IconLocation = "shell32.dll,21"
$shortcut2.Save()

Write-Host "  Start Menu shortcut also created!" -ForegroundColor Green
Write-Host ""

Read-Host "Press Enter to close"
