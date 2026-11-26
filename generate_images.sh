#!/usr/bin/env bash
# Generate Dockerfile and Singularity.def with Neurodocker
# (https://github.com/ReproNim/neurodocker). Neurodocker runs through uvx, so
# neither Docker nor a local Neurodocker installation is needed. Both recipes
# are built from the same specification below: the code, the notebooks and the
# Python environment locked in uv.lock, without the dataset.
#
# Usage: bash generate_images.sh

set -euo pipefail

cd "$(dirname "$0")"

NEURODOCKER="neurodocker==2.1.2"
UV_VERSION="0.10.9"
PROJECT="/opt/facescenes"
BUILD_TMP="/opt/build-tmp"

spec=(
  --pkg-manager apt
  --base-image python:3.12-slim-bookworm
  --label org.opencontainers.image.source=https://github.com/VCCN-lab/scene-context-face-detection
  --label org.opencontainers.image.licenses=MIT
  --env UV_PROJECT_ENVIRONMENT=/opt/venv UV_PYTHON_DOWNLOADS=never
        UV_NO_CACHE=1 UV_COMPILE_BYTECODE=1
  --env 'PATH=/opt/venv/bin:$PATH'
  --copy pyproject.toml uv.lock .python-version README.md LICENSE "$PROJECT/"
  --workdir "$PROJECT"
  --run "pip install --no-cache-dir uv==$UV_VERSION && uv sync --frozen --no-dev"
  # Only these folders: never data/ (it holds the scene photographs) or .git.
  --copy code "$PROJECT/code"
  --copy notebooks "$PROJECT/notebooks"
  --copy tests "$PROJECT/tests"
  # Create the user, give it the project folder, then run as that user.
  --user facescenes
  --user root
  --run "chown -R facescenes $PROJECT"
  --user facescenes
  --entrypoint "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser"
  --yes
)

uvx "$NEURODOCKER" generate docker "${spec[@]}" > Dockerfile

# apptainer build binds the host /tmp into %post, so every build step of the
# Apptainer image writes its temporaries to a folder inside the image instead.
uvx "$NEURODOCKER" generate singularity "${spec[@]}" | awk -v tmp="$BUILD_TMP" '
  /^%post$/      { print; print "mkdir -p " tmp; print "export TMPDIR=" tmp; next }
  /^%runscript$/ { print "rm -rf " tmp; print ""; print; next }
  { print }' > Singularity.def
echo "wrote Dockerfile and Singularity.def"
