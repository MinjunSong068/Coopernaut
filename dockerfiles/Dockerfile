
# =============================================================================
# Dockerfile for UT-Austin-RPL/Coopernaut
# End-to-End Driving with Cooperative Perception for Networked Vehicles
# https://github.com/UT-Austin-RPL/Coopernaut
#
# This builds the TRAINING/CLIENT container only: conda env "autocast"
# (Python 3.7), PyTorch 1.7.1 + CUDA 11.0, torch-geometric, MinkowskiEngine,
# and the CARLA 0.9.11 Python client.
#
# The CARLA SERVER runs as a separate container from the official
# `carlasim/carla:0.9.15` image -- see docker-compose.yml. This mirrors how
# CARLA is normally run in Docker and avoids baking a ~15GB simulator binary
# into the training image, which doesn't need it (the server does the actual
# Unreal Engine rendering; the client here just talks to it over RPC).
#
# The CARLA Python client isn't on PyPI for versions before 0.9.12, so
# instead of pip, we extract the exact-matching client .egg straight out of
# the carlasim/carla:${CARLA_VERSION} server image via a multi-stage build.
# This guarantees a byte-for-byte version match with whatever server you run
# (0.9.15 ships a py3.7 egg too, same as 0.9.11), and since you need that
# image locally anyway to run the server, it costs no extra download once
# it's pulled.
#
# IMPORTANT: the CARLA client and server versions must match. AutoCastSim/
# Coopernaut was originally built against 0.9.11 -- some scenario_runner.py /
# agent / sensor code may behave differently on 0.9.15's API surface, so
# validate incrementally (connect + tick the world) before trusting results.
#
# One such break, worked around below: CARLA restructured its PythonAPI
# `agents` navigation helper package (Agent/AgentState, GlobalRoutePlanner,
# ...) starting around 0.9.13 -- files AutoCastSim's agent code imports
# (agents.navigation.agent, agents.navigation.global_route_planner_dao) no
# longer exist in 0.9.15's copy. We pull an 0.9.11-vintage copy of just that
# helper package from the 0.9.11 server image instead (CARLA_AGENTS_VERSION,
# independent of CARLA_VERSION) -- it only touches the stable core carla.*
# API, so it works fine against a newer server. See the README for details.
#
# Build:
#   docker pull carlasim/carla:0.9.15   # if not already pulled
#   docker build --build-arg CARLA_VERSION=0.9.15 -t coopernaut:latest .
#
# Run: use docker-compose.yml, which wires this container up to a
# carlasim/carla:0.9.15 server container on a shared network. See README.md.
# =============================================================================

# CARLA_VERSION must match the server image tag used in docker-compose.yml.
ARG CARLA_VERSION=0.9.15
# Pinned independently -- see the "agents" navigation helper note above.
ARG CARLA_AGENTS_VERSION=0.9.11

# -----------------------------------------------------------------------------
# Stage 1: extract the matching CARLA Python client from the official server
# image. Nothing from this stage runs -- we only copy one file out of it.
# -----------------------------------------------------------------------------
FROM carlasim/carla:${CARLA_VERSION} AS carla_pyapi

# -----------------------------------------------------------------------------
# Stage 2: extract an 0.9.11-vintage `agents` navigation helper package (see
# note above). Independent of CARLA_VERSION/carla_pyapi on purpose.
# -----------------------------------------------------------------------------
FROM carlasim/carla:${CARLA_AGENTS_VERSION} AS carla_agents_pyapi

FROM nvidia/cudagl:11.0-devel-ubuntu18.04

LABEL maintainer="coopernaut-docker" \
      description="Training environment for UT-Austin-RPL/Coopernaut"

ARG CARLA_VERSION=0.9.15
ARG CARLA_AGENTS_VERSION=0.9.11
# NOTE: Coopernaut's own docs/INSTALL.md pins MinkowskiEngine to commit
# f81ae66b33b883cd08ee4f64d08cf633608b118, but that commit is no longer
# reachable in the upstream repo's git history (rewritten/rebased at some
# point upstream), so `git checkout <hash>` fails during pip's VCS clone.
# NVIDIA's own current install docs, for this exact PyTorch 1.7.1 + CUDA 11.0
# combination, recommend installing directly from `master` with the same
# --blas=openblas flags -- see https://nvidia.github.io/MinkowskiEngine/quick_start.html
# so that's what we do below instead of pinning a dead commit.
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8
 
# -----------------------------------------------------------------------------
# System dependencies
#
# NOTE: the nvidia/cudagl base image ships apt sources for NVIDIA's
# machine-learning/CUDA repos whose signing key was rotated in 2022 and was
# never patched into this (deprecated) image, so `apt-get update` fails with
# "NO_PUBKEY" / "not signed". We don't need anything from those repos -- the
# CUDA 11.0 runtime/libs we need are already baked into the base image -- so
# we just drop those broken source lists rather than re-trusting a new key.
#
#   - build-essential/cmake: needed to compile MinkowskiEngine
#   - libopenblas-dev: BLAS backend for MinkowskiEngine
#   - mosquitto: MQTT broker used for the simulated V2V message bus (paho-mqtt)
#   - libsdl2/libjpeg/libpng/libtiff: pygame + CARLA client rendering deps
# -----------------------------------------------------------------------------
RUN rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list
 
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        curl \
        ca-certificates \
        pkg-config \
        libopenblas-dev \
        mosquitto \
        mosquitto-clients \
        libsdl2-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff5-dev \
    && rm -rf /var/lib/apt/lists/*
 
# -----------------------------------------------------------------------------
# Miniconda (Coopernaut targets Python 3.7 specifically)
# -----------------------------------------------------------------------------
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py37_23.1.0-1-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniconda.sh
ENV PATH=${CONDA_DIR}/bin:${PATH}
 
RUN conda create -y -n autocast python=3.7 && conda clean -afy
 
# Run all subsequent RUN commands inside the "autocast" conda env
SHELL ["conda", "run", "-n", "autocast", "/bin/bash", "-c"]
 
# -----------------------------------------------------------------------------
# PyTorch (pinned to match torch-scatter/torch-sparse/torch-geometric wheels)
# -----------------------------------------------------------------------------
RUN conda install -y pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 \
        cudatoolkit=11.0 -c pytorch && \
    conda clean -afy
 
# -----------------------------------------------------------------------------
# Core pip dependencies (per docs/INSTALL.md)
# -----------------------------------------------------------------------------
RUN pip install --no-cache-dir \
        paho-mqtt \
        scipy \
        pygame \
        py-trees==0.8.3 \
        networkx==2.2 \
        xmlschema \
        numpy \
        shapely \
        imageio \
        ray \
        tqdm \
        numba \
        pandas \
        scikit-image \
        scikit-learn \
        opencv-python \
        h5py

RUN pip install --no-cache-dir open3d==0.13.0 tensorboard

# -----------------------------------------------------------------------------
# CARLA Python client -- copied from the official carlasim/carla server image
# (see the carla_pyapi stage above), NOT pip/PyPI (no wheel exists there for
# versions before 0.9.12). This is the exact client that ships with whatever
# server image you run, guaranteeing a version match. It's just the RPC
# client bindings, NOT the simulator itself -- the simulator runs in a
# separate carlasim/carla container (see docker-compose.yml).
# -----------------------------------------------------------------------------
RUN mkdir -p /opt/carla-pyapi
COPY --from=carla_pyapi /home/carla/PythonAPI/carla/dist/carla-${CARLA_VERSION}-py3.7-linux-x86_64.egg /opt/carla-pyapi/
# A .egg is just a zip archive -- Python can import straight out of it once
# it's on PYTHONPATH, no install step needed (and easy_install, the
# traditional way to "install" one, no longer ships in modern setuptools).
ENV CARLA_EGG=/opt/carla-pyapi/carla-${CARLA_VERSION}-py3.7-linux-x86_64.egg

# `agents` navigation helper package, pinned to CARLA_AGENTS_VERSION (see the
# note at the top of this file) -- NOT from the carla_pyapi stage above.
COPY --from=carla_agents_pyapi /home/carla/PythonAPI/carla/agents /opt/carla-pyapi/agents

ENV PYTHONPATH="${CARLA_EGG}:/opt/carla-pyapi"

# Sanity check: fail the build now if `import carla` / `import agents`
# doesn't work, rather than surfacing a confusing error later at training
# time.
RUN python -c "import carla; print('carla client OK:', carla.__file__)" && \
    python -c "from agents.navigation.agent import Agent, AgentState; from agents.navigation.global_route_planner import GlobalRoutePlanner; print('carla agents helper OK')"
 
# -----------------------------------------------------------------------------
# PyTorch Geometric stack (needs to match torch 1.7.1 + cu110 wheel index)

# -----------------------------------------------------------------------------
RUN pip install --no-cache-dir torch-scatter==2.0.5 \
        -f https://data.pyg.org/whl/torch-1.7.1+cu110.html && \
    pip install --no-cache-dir torch-sparse==0.6.9 \
        -f https://data.pyg.org/whl/torch-1.7.1+cu110.html && \
    pip install --no-cache-dir torch-geometric==1.7.2
 
# -----------------------------------------------------------------------------
# MinkowskiEngine (sparse convolution backbone used by Coopernaut's point
# encoder). Built against system OpenBLAS, no-deps to avoid clobbering the
# pinned torch build above.
# -----------------------------------------------------------------------------
RUN conda install -y openblas-devel -c anaconda && conda clean -afy
 
RUN pip install -U git+https://github.com/NVIDIA/MinkowskiEngine.git \
        -v --no-deps \
        --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" \
        --install-option="--blas=openblas"
 
# -----------------------------------------------------------------------------
# Coopernaut source + AutoCastSim submodule
#
# NOTE: this image does NOT bake in a clone of the repo. Your local checkouts
# of Coopernaut (and, if you keep it separate, AutoCastSim) are meant to be
# bind-mounted at container runtime to /workspace/Coopernaut (and
# /workspace/Coopernaut/AutoCastSim) — see docker-compose.yml / the README.
# That way edits you make on the host (e.g. timestamp instrumentation in
# scenario_runner.py) are reflected immediately in the container with no
# rebuild. We just create the mount point and PYTHONPATH here.
# -----------------------------------------------------------------------------
WORKDIR /workspace
RUN mkdir -p /workspace/Coopernaut
 
ENV SCENARIO_RUNNER_ROOT=/workspace/Coopernaut/AutoCastSim/srunner
# /opt/carla-pyapi (carla egg + the pinned-vintage `agents` helper package)
# comes first on PYTHONPATH. NOTE: this does NOT protect you from a stray
# `agents/` directory sitting directly inside a mounted AutoCastSim checkout
# -- Python puts the running script's own directory (sys.path[0]) ahead of
# PYTHONPATH, so a local AutoCastSim/agents/ always wins over this one. If
# `from agents.navigation.agent import ...` fails with ModuleNotFoundError
# even though this image is fine, check for (and fix/remove) that directory
# on the host -- see the README.
ENV PYTHONPATH="${CARLA_EGG}:/opt/carla-pyapi:${SCENARIO_RUNNER_ROOT}:/workspace/Coopernaut/AutoCastSim:/workspace/Coopernaut"

# -----------------------------------------------------------------------------
# pygame (used by AVR/HUD.py) needs fontconfig + at least one system font to
# resolve its default monospace font -- without this, pygame.font.get_fonts()
# returns an empty list and HUD init crashes with "IndexError: list index out
# of range" before a single frame runs. ttf-ubuntu-font-family provides the
# "ubuntumono" family HUD.py looks for by name.
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        fontconfig \
        ttf-ubuntu-font-family \
    && rm -rf /var/lib/apt/lists/* \
    && fc-cache -f

# Make conda env auto-activate in interactive shells
RUN echo "conda activate autocast" >> /root/.bashrc
 
WORKDIR /workspace/Coopernaut
 
# Default: drop into a bash shell with the autocast env active.
# Use docker-entrypoint style activation so `docker run ... <cmd>` also works
# inside the right conda env.
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "autocast"]
CMD ["/bin/bash"]