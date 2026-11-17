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

## Why the first start of a model takes long

* **First start of a model downloads it** (2B about 4-5 GB, 8B about 17 GB). The log says
  `NOT in the local cache. Downloading about N GB once` and `./serve.sh start` shows the progress line.
  Download in advance, while nothing else runs: `python download_models.py qwen3-vl-2b qwen3-vl-8b`.
* **Later starts load from the cache without any network check** (log: `found in the local cache`),
  so a slow or broken network no longer delays the start.
* **4-bit models are quantized while loading**: about 1-2 minutes for 8B at every start. That is
  normal. Small models in `bf16` skip this step.
* Which models are downloaded: `python check_env.py`. Where: `~/.cache/huggingface` (set `HF_HOME`
  to use another disk).

## Which model (faster ones)

All models are listed in `models.yaml`; start any of them with `./serve.sh start <key> <quant>`.

| Key | Size | Note |
|---|---|---|
| `qwen3-vl-8b` | 8B | the default so far |
| `qwen3-vl-4b` | 4B | |
| `qwen3-vl-2b` | 2B | same family as the default: the safest faster model to try first |
| `qwen3.5-4b` | 4B | newer (2026); thinks by default, switched off in `models.yaml` |
| `qwen3.5-2b` | 2B | newer (2026), non-thinking |
| `qwen3.5-0.8b` | 0.8B | smallest; meant for prototyping |
| `lfm2.5-vl-1.6b` | 1.6B | Liquid AI, built for speed; license lfm1.0 |

* Small models are usually fastest in `bf16` (no 4-bit decompression on every token):
  `./serve.sh start qwen3-vl-2b bf16`.
* Qwen3.5 needs the newest transformers, LFM2.5-VL transformers >= 5.1:
  `.venv/bin/pip install -U "transformers>=5.1"` (Qwen3.5: `pip install -U git+https://github.com/huggingface/transformers`
  if the release is not new enough). Check that the older models still load afterwards.
* None of these new models has been run yet. Compare them on the photographed scenes:
  `./serve.sh stop && scripts/compare_models.sh`, then open `/dashboard` (chart "Models: speed against
  correct plans"). A faster model is only useful if its plans still work.

## Is the plan the shortest one?

Every `ok` response carries a `quality` field (the robot side may ignore it):

```json
"quality": {"moves": 3, "lower_bound": 3, "optimal": true,
            "proof": "the plan moves only objects that must move, each once",
            "max_extra_moves": 0, "moved_more_than_once": []}
```

* `lower_bound`: objects that must be picked up at least once (goal not met, in the way of
  another object's goal, or standing on top of something that must move). No plan can be shorter.
* The server first asks a fast planner (about 0.1 s). If that plan has exactly `lower_bound`
  moves it is proven shortest and returned at once. Only otherwise does it run the optimal
  search (Fast Downward `astar(lmcut())`, at most `OPTIMAL_TIMEOUT_S` = 5 s).
* `optimal: false` only if the optimal search did not finish in time; then
  `max_extra_moves` says how many moves the plan could have in excess.
* `moved_more_than_once` lists objects picked up twice. In a plan proven shortest this is
  unavoidable (for example two objects that stand in each other's zone).
* Goals that are certainly impossible (a zone with fewer slots than objects that must go
  there, something on a prism, too tall for a shelf) are answered as `no_plan` before any
  planning, instead of after the planners' timeouts.
* After a scene change, a kept plan keeps its proof: the rest of a shortest plan, from a
  state on that plan, is itself shortest.

## Quick problem table

| Problem | Fix |
|---|---|
| Page says "server not reachable" / `file://` in address bar | start the server, open `http://127.0.0.1:8010/` |
| Browser shows another app (VisionQ, ...) | port taken: `PORT=8020 ./serve.sh restart qwen3-vl-8b 4bit` |
| Works on the PC, not from the laptop | run the port forwarding script (as admin); after restart again |
| `./serve.sh start` says "server exited" | `./serve.sh logs`: missing package -> `pip install -r requirements.txt` |
| `no kernel image is available` / `sm_120 is not compatible` (RTX 50xx) | torch too old for the GPU: `pip uninstall -y torch torchvision && pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128` |
| `ensurepip is not available` in `./setup.sh` | `sudo apt update && sudo apt install -y python3-venv python3-pip`, then `rm -rf .venv && ./setup.sh` |
| Very slow / out of memory | smaller model: `./serve.sh restart qwen3-vl-4b 4bit` |
