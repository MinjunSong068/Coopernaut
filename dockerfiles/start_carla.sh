#!/bin/bash
# Launch the official CARLA server container headless, matching the version
# Coopernaut/AutoCastSim needs (0.9.15 by default).
#
# This is the plain `docker run` equivalent of the `carla-server` service in
# docker-compose.yml -- use this if you're not using compose. If you already
# run CARLA via Docker normally, this is the same pattern, just pinned to
# 0.9.15 and headless.
#
# Usage: ./start_carla.sh [VERSION] [PORT]
set -e

CARLA_VERSION=${1:-0.9.15}
PORT=${2:-2000}
 
echo "Starting carlasim/carla:${CARLA_VERSION} headless on port ${PORT}..."
echo "(if this image isn't pulled yet: docker pull carlasim/carla:${CARLA_VERSION})"
 
docker run -d --name carla-server \
    --gpus all \
    --shm-size=4g \
    -p ${PORT}-$((PORT+2)):${PORT}-$((PORT+2)) \
    carlasim/carla:${CARLA_VERSION} \
    /bin/bash CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=${PORT}
 
echo "carla-server container started. Give it ~15-20s to finish booting."
echo "Connect the coopernaut container to it via --network (see README)."