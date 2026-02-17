#!/bin/bash
set -ex # Added -x for debugging

# Get PUID and PGID, default to 1000 if not set
USER_ID=${PUID:-1000}
GROUP_ID=${PGID:-1000}

echo "Starting with UID : $USER_ID, GID: $GROUP_ID"

# 1. Handle Group
if getent group "$GROUP_ID" >/dev/null 2>&1; then
    GROUP_NAME=$(getent group "$GROUP_ID" | cut -d: -f1)
    echo "Using existing group: $GROUP_NAME (GID $GROUP_ID)"
else
    # Check if name conflict exists
    if getent group dubgroup >/dev/null 2>&1; then
        echo "Group dubgroup exists but GID mismatch. Renaming old dubgroup."
        groupmod -n dubgroup_old dubgroup
    fi
    groupadd -g "$GROUP_ID" dubgroup
    GROUP_NAME="dubgroup"
    echo "Created group: $GROUP_NAME (GID $GROUP_ID)"
fi

# 2. Handle User
if getent passwd "$USER_ID" >/dev/null 2>&1; then
    USER_NAME=$(getent passwd "$USER_ID" | cut -d: -f1)
    echo "Using existing user: $USER_NAME (UID $USER_ID)"
    # Ensure user is in the group
    usermod -aG "$GROUP_NAME" "$USER_NAME"
else
    # Check if name conflict exists
    if getent passwd dubuser >/dev/null 2>&1; then
        echo "User dubuser exists but UID mismatch. Renaming old dubuser."
        usermod -l dubuser_old dubuser
    fi
    useradd -u "$USER_ID" -g "$GROUP_ID" -m -s /bin/bash dubuser
    USER_NAME="dubuser"
    echo "Created user: $USER_NAME (UID $USER_ID)"
fi

# Set custom cache locations to a writable path
# If user exists, HOME might already be set, but we override for container consistency
export HOME=/home/"$USER_NAME"
export HF_HOME=/data/cache/huggingface
export TORCH_HOME=/data/cache/torch
export TTS_HOME=/data/cache/tts
export XDG_CACHE_HOME=/data/cache

# Ensure directories exist
echo "Creating directories..."
mkdir -p /app/output /app/videos /app/models /app/logs /config /data/cache

# Fix permissions for the volumes and cache
echo "Fixing permissions for $USER_NAME:$GROUP_NAME..."
chown -R "$USER_NAME":"$GROUP_NAME" /app/output /app/videos /app/models /app/logs /config /data/cache /app/src

# Run the command as the user
echo "Executing command as $USER_NAME: $@"
exec gosu "$USER_NAME" "$@"
