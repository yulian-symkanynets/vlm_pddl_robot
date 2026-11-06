# Forward a Windows port to the planning server running inside WSL2, and open the firewall.
# Run in PowerShell AS ADMINISTRATOR (WSL/Ubuntu must be running):
#   powershell -ExecutionPolicy Bypass -File scripts\wsl_portproxy.ps1             # port 8010
#   powershell -ExecutionPolicy Bypass -File scripts\wsl_portproxy.ps1 -Port 8020
#   powershell -ExecutionPolicy Bypass -File scripts\wsl_portproxy.ps1 -Remove     # undo
# The WSL IP changes after every Windows restart -> run it again after a restart.
param([int]$Port = 8010, [switch]$Remove)

$ErrorActionPreference = 'Stop'
$rule = "WSL planner $Port"

if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Run PowerShell as Administrator (right click -> Run as administrator).'
}

netsh interface portproxy delete v4tov4 listenaddress=0.0.0.0 listenport=$Port 2>$null | Out-Null
if ($Remove) {
    Remove-NetFirewallRule -DisplayName $rule -ErrorAction SilentlyContinue
    Write-Host "Removed forwarding and firewall rule for port $Port."
    exit 0
}

$wslIp = ((wsl hostname -I) -split '\s+' | Where-Object { $_ -match '^\d+\.\d+\.\d+\.\d+$' } | Select-Object -First 1)
if (-not $wslIp) { throw 'Could not get the WSL IP. Open Ubuntu first, then run this again.' }

# portproxy needs the "IP Helper" service
Set-Service iphlpsvc -StartupType Automatic
Start-Service iphlpsvc

netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=$Port connectaddress=$wslIp connectport=$Port | Out-Null
if (-not (Get-NetFirewallRule -DisplayName $rule -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule -DisplayName $rule -Direction Inbound -Protocol TCP -LocalPort $Port -Action Allow | Out-Null
}

Write-Host ""
Write-Host "Forwarding  Windows 0.0.0.0:${Port}  ->  WSL ${wslIp}:${Port}"
netsh interface portproxy show v4tov4
Write-Host "Open on this PC:          http://127.0.0.1:${Port}/"
Write-Host "Open from other machines:"
Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.IPAddress -notmatch '^(127\.|169\.254\.)' -and $_.InterfaceAlias -notmatch 'vEthernet|WSL|Loopback' } |
    ForEach-Object { Write-Host ("   http://{0}:{1}/      ({2})" -f $_.IPAddress, $Port, $_.InterfaceAlias) }

try {
    $h = Invoke-RestMethod -Uri "http://${wslIp}:${Port}/health" -TimeoutSec 3
    Write-Host "Server check: OK, model = $($h.model)"
} catch {
    Write-Host "Server check: no answer on ${wslIp}:${Port}. Start it in WSL:  PORT=$Port ./serve.sh start qwen3-vl-8b 4bit"
}
