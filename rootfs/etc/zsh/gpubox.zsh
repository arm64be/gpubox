# Sourced from /etc/zshrc: interactive shells.

HISTFILE=~/.zsh_history
HISTSIZE=100000
SAVEHIST=100000
setopt extended_history hist_ignore_dups hist_ignore_space share_history
setopt auto_cd interactive_comments no_beep

bindkey -e
bindkey '^[[1;5C' forward-word      # ctrl-right
bindkey '^[[1;5D' backward-word     # ctrl-left
bindkey '^[[H' beginning-of-line
bindkey '^[[F' end-of-line
bindkey '^[[3~' delete-char

autoload -Uz compinit && compinit -d ~/.zcompdump
zstyle ':completion:*' menu select
zstyle ':completion:*' matcher-list 'm:{a-z}={A-Z}'

alias ls='ls --color=auto' ll='ls -lh' la='ls -lAh' grep='grep --color=auto'
alias gpus="nodestat --once | jq -r '.gpu.devices[] | \"gpu \\(.index)  \\(.short)  \\(.util)%  \\((.mem_used/1073741824*10|round)/10)/\\((.mem_total/1073741824)|round) GiB  \\(.power|round) W  \\(.temp) °C\"'"

# Prompt: starship's defaults, with the node name in place of the container id.
export STARSHIP_CONFIG=/etc/starship.toml
eval "$(starship init zsh)"

# Once per ssh login.
if [[ -o login && -t 1 && -z ${GPUBOX_GREETED:-} ]]; then
    export GPUBOX_GREETED=1
    fastfetch 2>/dev/null
fi
