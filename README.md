# gpubox

A Fedora image for rented GPU nodes.

```
docker pull dockerstatic.tsuku.re/gpubox:latest
```

- **CUDA 13.0** toolkit (nvcc, libraries, cupti, compute-sanitizer), **Nsight Compute** (`ncu`) and **Nsight Systems** (`nsys`). 13.0 runs on any host whose driver reports CUDA 13.0 or newer.
- **uv** with a managed Python and a ready venv at `/opt/venv` (on `PATH`): PyTorch (cu130), CuTeDSL (`nvidia-cutlass-dsl[cu13]`), numpy, ninja.
- **zsh** with starship's default prompt, showing the node name instead of the container id.
- **sshd** for root, key-only.
- **nodestat**, a status page on port 8080: host and GPU stats, live graphs, and a warning when the NVIDIA driver stops answering.
- fastfetch on login, plus nano, tmux, btop, nvtop, git-lfs, rclone, rsync, zstd, ffmpeg and opus-tools.

## Running it

| Variable | Default | What it does |
|---|---|---|
| `SSH_PUBKEY` | | Public key(s) for root, one per line. `PUBLIC_KEY` and `SSH_PUBLIC_KEY` also work. sshd starts only if a key is set. |
| `NODE_NAME` | generated | Name shown in the prompt and on the status page. |
| `GPUBOX_NO_GEO` | `0` | Set to `1` to skip the country lookup when naming the node. |
| `SSH_PORT` | `22` | sshd port. |
| `NODESTAT_PORT` | `8080` | Status page port. |
| `NODESTAT_TOKEN` | | If set, the status page asks for `?token=…` once, then remembers it in a cookie. |
| `GPUBOX_SSH`, `GPUBOX_NODESTAT` | `1` | Set to `0` to skip starting either one. |

Expose ports 22 and 8080. Any arguments become the container's main command; with none, the container idles and serves ssh and the status page.

Every variable the container starts with is also visible in ssh sessions, so API keys set in a template reach your shell.

### Node names

The generated name is the GPU count and model, plus the country of the node's public IP: `4x4070-de`, `2xGB300-fr`, `8xH100+1x4090-us`. A node without GPUs is `cpu-xx`. The country comes from `ipinfo.io`, falling back to `ifconfig.co`. Set `NODE_NAME` to choose your own.

## Python

`/opt/venv` is active in every shell, so `python` and `uv pip install` use it. For a project that should reuse the preinstalled PyTorch without downloading it again:

```sh
uv venv --python /opt/venv/bin/python --system-site-packages .venv
```

## CUDA notes

- Compile real machine code for the target GPU, e.g. `nvcc -arch=native`. Binaries built with 13.0 run on any 13.x driver.
- CUDA 13 dropped Maxwell, Pascal and Volta, so the oldest GPUs this supports are Turing (sm_75).
- The CUDA static libraries are removed to save about 2 GB, except `libcudart_static.a`, which nvcc links by default.

## Building and publishing

Build locally, then publish to the static registry on dev-vm:

```sh
podman build -f Containerfile -t gpubox .
./publish.sh localhost/gpubox:latest latest cuda13.0-$(date -u +%Y%m%d)
```

`publish.sh` exports the image with gzip layers, uploads only the layers dev-vm doesn't have yet, then the manifest, then repoints the tags. Rebuilding after a small change therefore re-uploads only the changed layers. The registry is plain files in `/srv/dockerstatic` on dev-vm, served read-only by nginx. Its config lives in the tsukure infra repo.

Old layers stay on dev-vm until removed by hand.

`nodestat/` builds on its own with `make`. `mock_nvml.so` simulates multi-GPU nodes for working on the page without the hardware (see the top of `nodestat/mock_nvml.c`).

## Font

The status page embeds Obel Mono under the SIL Open Font License 1.1 (`nodestat/fonts/OFL.txt`).
