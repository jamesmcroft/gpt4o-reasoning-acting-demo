#!/usr/bin/env bash

# Default: Exit on any failure.
set -e

# Setup STDERR.
err() {
    echo "(!) $*" >&2
}

if [ "$(id -u)" -ne 0 ]; then
    err 'Script must be run as root. Use sudo, su, or add "USER root" to your Dockerfile before running this script.'
    exit 1
fi

###################
# Helper Functions
###################

apt_get_update() {
    if [ "$(find /var/lib/apt/lists/* | wc -l)" = "0" ]; then
        echo "Running apt-get update..."
        apt-get update -y
    fi
}

pip_get_update() {
    echo "Running pip update..."
    pip install --upgrade pip
}

# Checks if apt packages are installed and installs them if not
install_apt_packages() {
    if ! dpkg -s "$@" >/dev/null 2>&1; then
        apt_get_update
        apt-get -y install --no-install-recommends "$@"
    fi
}

# Checks if pip packages are installed and installs them if not
install_pip_packages() {
    if ! pip show "$@" >/dev/null 2>&1; then
        pip_get_update
        pip install "$@"
    fi
}

###########################################
# Install Feature
###########################################

# Ensure apt is in non-interactive to avoid prompts
export DEBIAN_FRONTEND=noninteractive

# Install dependencies
install_pip_packages notebook ipykernel ipycanvas

echo 'dev-tools script has completed!'
