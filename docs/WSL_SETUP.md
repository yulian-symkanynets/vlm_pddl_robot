# New Windows machine: WSL check, server start, port forwarding

Port **8010** is the default (8000 is often taken by other servers, e.g. VisionQ). Another port: `PORT=8020 ./serve.sh start ...`.
**PS>** = Windows PowerShell, **$** = Ubuntu (WSL) terminal.

---

## 1. Check WSL (Windows side)

```powershell
PS> wsl --version          # WSL version 2.x; an error means old WSL -> run: wsl --update
PS> wsl -l -v              # list of distros; VERSION must be 2
PS> winver                 # Windows 11 22H2+ can use "mirrored" networking (step 5A)
```

| You see | Do |
|---|---|
| `wsl` not found / no distro | `PS> wsl --install -d Ubuntu-24.04` (as Administrator), restart Windows, open Ubuntu, create user |
| `VERSION 1` | `PS> wsl --set-version Ubuntu-24.04 2` |
| `wsl --version` fails | `PS> wsl --update`, then `PS> wsl --shutdown` |

## 2. Check GPU and tools (inside WSL)

```bash
$ nvidia-smi               # must show the GPU. Driver is installed on WINDOWS only, never install one in Ubuntu
$ python3 --version        # 3.10+
$ git --version
$ free -h                  # RAM visible to WSL (models need ~16 GB+; raise it in .wslconfig: memory=24GB)
```
If `nvidia-smi` is missing: update the NVIDIA driver on Windows, then `PS> wsl --shutdown` and reopen Ubuntu.

## 3. Get the code and install

```bash
$ git clone https://github.com/yulian-symkanynets/vlm_pddl_robot.git ~/vlm_pddl_robot
$ cd ~/vlm_pddl_robot
$ ./setup.sh                       # venv + torch + requirements
$ source .venv/bin/activate
$ python check_env.py              # CUDA, bitsandbytes, planners
$ python -m pytest tests -q        # all tests should pass
```

## 4. Start the server and check it

```bash
$ ./serve.sh start qwen3-vl-8b 4bit     # first start downloads + loads the model (minutes)
$ ./serve.sh status
$ curl http://localhost:8010/health               # {"status":"up","model":"qwen3-vl-8b",...}
```
Quick test without the GPU model: `./serve.sh start mock`.

**On the same Windows PC** open `http://127.0.0.1:8010/` (not `localhost`, not the html file).
If it does not open: `$ hostname -I` and use `http://<that-ip>:8010/`.

Something else on the port? `PS> netstat -ano | findstr :8010` -> pick another port.

## 5. Make it reachable from other machines (laptop, robot PC)

Choose ONE.

### 5A. Mirrored networking (Windows 11 22H2+, simplest, survives restarts)

`C:\Users\<you>\.wslconfig`:
```ini
[wsl2]
networkingMode=mirrored
```
```powershell
PS> wsl --shutdown
# as Administrator:
PS> New-NetFirewallRule -DisplayName "WSL planner 8010" -Direction Inbound -Protocol TCP -LocalPort 8010 -Action Allow
PS> New-NetFirewallHyperVRule -Name "WSLPlanner8010" -DisplayName "WSL planner 8010" -Direction Inbound -VMCreatorId '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -Protocol TCP -LocalPorts 8010
```
Reopen Ubuntu, start the server (step 4). Address for others: Windows IP from `PS> ipconfig`.

### 5B. Port forwarding script (any Windows, redo after each restart)

Start the server in WSL first (step 4), then in **PowerShell as Administrator**:
```powershell
PS> cd \\wsl.localhost\Ubuntu-24.04\home\<user>\vlm_pddl_robot
PS> powershell -ExecutionPolicy Bypass -File scripts\wsl_portproxy.ps1
```
It finds the current WSL IP, adds the forwarding + firewall rule, prints the address to use from
other machines and checks that the server answers. Undo: add `-Remove`.

Same thing by hand:
```powershell
PS> $ip = (wsl hostname -I).Trim().Split(' ')[0]
PS> netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=8010 connectaddress=$ip connectport=8010
PS> New-NetFirewallRule -DisplayName "WSL planner 8010" -Direction Inbound -Protocol TCP -LocalPort 8010 -Action Allow
PS> netsh interface portproxy show v4tov4
```

### 5C. Tailscale (when the network blocks device-to-device traffic, or from home)

```bash
$ curl -fsSL https://tailscale.com/install.sh | sh
$ sudo tailscale up          # log in via the printed link (if "daemon not running": sudo tailscaled & first)
$ tailscale ip -4            # 100.x.y.z
```
Install Tailscale on the laptop, same account -> `http://100.x.y.z:8010/`. No forwarding, no firewall rules.

## 6. Test from the other machine

```bash
curl http://<windows-ip>:8010/health
SERVER_URL=http://<windows-ip>:8010 python client.py demo
```
Browser: `http://<windows-ip>:8010/`.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Browser shows another app (e.g. VisionQ) | port taken -> use another `PORT=` |
| `file://...` in the address bar, "Failed to fetch" | page opened as a file -> open `http://127.0.0.1:8010/` |
| Works on the PC, not from the laptop | firewall rule missing (5A/5B), or university Wi-Fi isolates devices -> 5C |
| Worked yesterday, not today | WSL IP changed after restart -> run the 5B script again (or use 5A) |
| `curl localhost:8010/health` fails inside WSL | server not running / still loading -> `./serve.sh logs` |
| Port forwarding active but no answer | `PS> Get-Service iphlpsvc` must be Running; VPN clients can also block it |
| `Could not resolve host: github.com` in WSL | DNS broken: `.wslconfig` -> `[wsl2]` `dnsTunneling=true`, `wsl --shutdown`. Else `/etc/wsl.conf` `[network]` `generateResolvConf = false`, restart WSL, write `nameserver <windows-dns>` + `nameserver 8.8.8.8` into `/etc/resolv.conf` |
| No network at all in WSL (`ping 8.8.8.8` fails) | VPN -> disconnect or use `networkingMode=mirrored`; proxy -> `netsh winhttp show proxy`, then `git config --global http.proxy ...` and `export https_proxy=...` |
| Out of memory while loading the model | `.wslconfig` -> `[wsl2]` `memory=24GB`, then `wsl --shutdown` |
