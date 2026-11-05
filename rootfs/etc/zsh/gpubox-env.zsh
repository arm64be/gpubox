# Sourced from /etc/zshenv: every zsh, including `ssh node some-command`.

# Variables the container was started with (saved by gpubox-init).
[[ -r /run/gpubox/env ]] && source /run/gpubox/env

typeset -U path
path=(/opt/venv/bin /usr/local/cuda/bin /usr/local/bin /usr/local/sbin /usr/bin /usr/sbin $path)
export VIRTUAL_ENV=/opt/venv
