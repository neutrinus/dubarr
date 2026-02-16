#!/bin/bash

# Get PUID and PGID, default to 1000 if not set
USER_ID=${PUID:-1000}
GROUP_ID=${PGID:-1000}

echo "Starting with UID : $USER_ID, GID: $GROUP_ID"

# Create group if it doesn't exist
if ! getent group dubgroup >/dev/null; then
    groupadd -g "$GROUP_ID" dubgroup
fi

# Create user if it doesn't exist
if ! getent passwd dubuser >/dev/null; then
    useradd -u "$USER_ID" -g "$GROUP_ID" -m dubuser
fi

# Set custom cache locations to a writable path
export HOME=/home/dubuser
export HF_HOME=/data/cache/huggingface
export TORCH_HOME=/data/cache/torch
export TTS_HOME=/data/cache/tts
export XDG_CACHE_HOME=/data/cache

# Ensure directories exist
mkdir -p /app/output /app/videos /app/models /data/cache

# Fix permissions for the volumes and cache
# We only chown what's necessary to save time
chown dubuser:dubgroup /app/output /app/videos /app/models /data/cache

# Run the command as the user
exec gosu dubuser "$@"
