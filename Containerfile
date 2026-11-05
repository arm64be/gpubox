# gpubox: a Fedora GPU dev image for rented nodes.
#
# CUDA 13.0 toolkit with Nsight Compute and Nsight Systems, uv with a ready
# PyTorch + CuTeDSL venv, zsh + starship, sshd, and the nodestat status page.

ARG FEDORA=44

# ---------------------------------------------------------------- nodestat
FROM registry.fedoraproject.org/fedora-minimal:${FEDORA} AS nodestat
RUN dnf -y install --setopt=install_weak_deps=False --setopt=retries=10 gcc make && dnf clean all
COPY nodestat/ /src/
RUN make -C /src nodestat CFLAGS="-O2 -Wall"

# ------------------------------------------------------------------- image
FROM registry.fedoraproject.org/fedora-minimal:${FEDORA}

# CUDA 13.0 matches the r580 drivers on most hosts ("Max CUDA 13.0").
# NVIDIA's RHEL 10 repo is the one that still carries 13.0 alongside Nsight.
ARG CUDA=13-0
ARG CUDA_DIR=/usr/local/cuda-13.0
ARG TORCH_BACKEND=cu130
ARG PYTHON=3.13
ARG STARSHIP=v1.27.0

LABEL org.opencontainers.image.source="https://github.com/arm64be/gpubox" \
      org.opencontainers.image.description="Fedora GPU dev box: CUDA 13.0 + Nsight, uv + PyTorch + CuTeDSL, zsh, sshd, status page"

ENV LANG=C.UTF-8 \
    EDITOR=nano \
    SHELL=/bin/zsh \
    CUDA_HOME=/usr/local/cuda \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NUMPY_MADVISE_HUGEPAGE=0 \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin

# Base system. gcc/g++ are also what Triton and torch.compile need at runtime.
RUN dnf -y install --setopt=install_weak_deps=False --setopt=retries=10 \
        zsh nano less which findutils diffutils file procps-ng util-linux shadow-utils hostname \
        git git-lfs tmux rsync curl jq zstd xz tar gzip unzip \
        htop btop nvtop fastfetch lsof strace gdb pciutils iproute iputils bind-utils numactl \
        rclone ffmpeg-free opus-tools \
        gcc gcc-c++ make cmake ninja-build \
        openssh-server openssh-clients catatonit \
    && dnf clean all

# CUDA toolkit and profilers. Static libraries other than cudart (which nvcc
# links by default) are deleted: they are ~2 GB and rarely wanted.
RUN curl -fsSLo /etc/yum.repos.d/cuda-rhel10.repo \
        https://developer.download.nvidia.com/compute/cuda/repos/rhel10/x86_64/cuda-rhel10.repo \
    && dnf -y install --setopt=install_weak_deps=False --setopt=retries=10 \
        cuda-compiler-${CUDA} cuda-libraries-devel-${CUDA} cuda-command-line-tools-${CUDA} \
        cuda-nsight-compute-${CUDA} cuda-nsight-systems-${CUDA} \
    && dnf clean all \
    && find ${CUDA_DIR} -name '*_static*.a' ! -name 'libcudart_static.a' -delete \
    && ln -sfn ${CUDA_DIR} /usr/local/cuda

# uv, a managed Python, and the venv. Nothing is left in a uv cache.
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /usr/local/bin/
RUN uv python install ${PYTHON} \
    && uv venv /opt/venv --python ${PYTHON} --seed \
    && UV_NO_CACHE=1 uv pip install --python /opt/venv --torch-backend ${TORCH_BACKEND} \
        torch "nvidia-cutlass-dsl[cu13]" numpy ninja setuptools wheel packaging

RUN curl -fsSLo /tmp/starship.tgz \
        https://github.com/starship/starship/releases/download/${STARSHIP}/starship-x86_64-unknown-linux-musl.tar.gz \
    && echo "$(curl -fsSL https://github.com/starship/starship/releases/download/${STARSHIP}/starship-x86_64-unknown-linux-musl.tar.gz.sha256)  /tmp/starship.tgz" | sha256sum -c - \
    && tar -xzf /tmp/starship.tgz -C /usr/local/bin starship \
    && rm /tmp/starship.tgz

COPY rootfs/ /
COPY --from=nodestat /src/nodestat /usr/local/bin/nodestat

# Root logs in to zsh. Containers can't set a login uid, so pam_loginuid must
# not be allowed to refuse ssh logins.
RUN sed -i 's|^root:\(.*\):/bin/bash$|root:\1:/bin/zsh|' /etc/passwd \
    && echo 'source /etc/zsh/gpubox-env.zsh' >> /etc/zshenv \
    && echo 'source /etc/zsh/gpubox.zsh' >> /etc/zshrc \
    && sed -i 's/^session\s\+required\s\+pam_loginuid.so/session optional pam_loginuid.so/' /etc/pam.d/sshd \
    && rm -f /etc/ssh/ssh_host_* \
    && chmod 755 /usr/local/bin/gpubox-init /usr/local/bin/gpubox-name

WORKDIR /root
EXPOSE 22 8080
ENTRYPOINT ["/usr/libexec/catatonit/catatonit", "--", "/usr/local/bin/gpubox-init"]
