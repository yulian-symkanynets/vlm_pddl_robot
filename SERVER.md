# How to start the planning server

Short version. Full details and troubleshooting: [docs/WSL_SETUP.md](docs/WSL_SETUP.md).

**PS>** = Windows PowerShell, **$** = Ubuntu (WSL) terminal. Default port: **8010**.

---

## First time on a new machine

```bash
$ nvidia-smi                                   # GPU visible in WSL?
$ ping -c 2 github.com                         # internet + DNS ok? (if not: "DNS fix" below)
$ git clone https://github.com/yulian-symkanynets/vlm_pddl_robot.git ~/vlm_pddl_robot
$ cd ~/vlm_pddl_robot
$ ./setup.sh                                   # creates .venv, installs torch + everything
$ source .venv/bin/activate
$ python check_env.py                          # CUDA, bitsandbytes, planners
$ python -m pytest tests -q                    # all tests pass
```

## Every time: start the server

```bash
$ cd ~/vlm_pddl_robot && git pull
$ ./serve.sh start qwen3-vl-8b 4bit            # runs in the background, waits until the model is loaded
$ ./serve.sh status                            # running? which model?
```

| Command | What it does |
|---|---|
| `./serve.sh start <model> <quant>` | start in background (models: see `models.yaml`; `mock` = no GPU) |
| `./serve.sh status` | is it running + `/health` |
| `./serve.sh logs` | follow the log (Ctrl+C to leave, server keeps running) |
| `./serve.sh restart <model> <quant>` | after `git pull` or to switch model |
| `./serve.sh stop` | stop it |
| `PORT=8020 ./serve.sh start ...` | other port, if 8010 is taken |

## Open it

On the same PC, in the Windows browser: **http://127.0.0.1:8010/**

- Type the address in the address bar; do not double-click `web/index.html`.
- If it does not open: `$ hostname -I` and use `http://<that-ip>:8010/`.
- If you see a different app: the port is taken, use `PORT=8020`.

API docs (try every endpoint): http://127.0.0.1:8010/docs

Command line test:
```bash
$ python client.py health
$ python client.py demo                        # all example cases, PASS/DIFF table
$ python client.py plan -i "put the yellow block into the box" --scene examples/scenes/box.json
$ python client.py plan -i "swap the two stacks" --image images/scene5_reorder_close.jpg
```

## Make it reachable from other machines (port forwarding)

1. Start the server (above).
2. Open **PowerShell as Administrator** (Start menu, right-click PowerShell, *Run as administrator*):
   ```powershell
   PS> cd \\wsl.localhost\Ubuntu-24.04\home\<your-user>\vlm_pddl_robot
   PS> powershell -ExecutionPolicy Bypass -File scripts\wsl_portproxy.ps1
   ```
   It forwards Windows port 8010 to WSL, opens the firewall, prints the address for other machines
   (e.g. `http://192.168.1.50:8010/`) and checks that the server answers.
3. On the laptop: open that address in the browser, or
   ```bash
   SERVER_URL=http://192.168.1.50:8010 python client.py health
   ```

**After every Windows restart run step 2 again** (the WSL IP changes).
Undo: `scripts\wsl_portproxy.ps1 -Remove`.

No admin rights, or the university network blocks connections between devices? Use Tailscale instead
(no forwarding needed): see section 5C in [docs/WSL_SETUP.md](docs/WSL_SETUP.md).

## DNS fix (`Could not resolve host: github.com`)

```bash
$ sudo rm -f /etc/resolv.conf
$ printf 'nameserver 8.8.8.8\nnameserver 1.1.1.1\n' | sudo tee /etc/resolv.conf
$ printf '[network]\ngenerateResolvConf = false\n' | sudo tee /etc/wsl.conf     # keep it after restarts
```

## Quick problem table

| Problem | Fix |
|---|---|
| Page says "server not reachable" / `file://` in address bar | start the server, open `http://127.0.0.1:8010/` |
| Browser shows another app (VisionQ, ...) | port taken: `PORT=8020 ./serve.sh restart qwen3-vl-8b 4bit` |
| Works on the PC, not from the laptop | run the port forwarding script (as admin); after restart again |
| `./serve.sh start` says "server exited" | `./serve.sh logs`: missing package -> `pip install -r requirements.txt` |
| Very slow / out of memory | smaller model: `./serve.sh restart qwen3-vl-4b 4bit` |
