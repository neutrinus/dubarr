#!/bin/bash

# ==============================================================================
# DUBARR DEMO GENERATOR
# ==============================================================================
# 
# Description:
#   Creates a comparison video for YouTube/Social Media. It takes segments
#   from a dubbed file and shows them side-by-side (sequentially):
#   Original -> Polish -> French -> German -> etc.
#
# Usage:
#   1. Edit the CONFIG section below (Input file, Timestamps, Tracks).
#   2. Ensure FFmpeg is installed.
#   3. Run: ./scripts/generate_demo.sh
#
# Requirements:
#   - Input video must have multiple audio tracks.
#   - A TTF font file for labels (default Adwaita or DejaVu).
# ==============================================================================

# --- CONFIGURATION ---
INPUT="nasa.mkv"
OUTPUT="dubarr_demo_labeled.mp4"
FONT="/usr/share/fonts/adwaita-mono-fonts/AdwaitaMono-Bold.ttf"

# Segments to extract: "StartTime Duration"
# Example: "00:04:35 10" will take 10 seconds starting at 4m 35s.
SEGMENTS=(
    "00:04:35 10"
    "00:08:02 10"
)

# Audio Track Mapping (0-based indices for AUDIO streams)
# To check indices run: ffprobe -v error -show_entries stream=index:stream_tags=language -of csv=p=0 file.mkv
# Current mapping for nasa.mkv:
# 3: English (Original)
# 0: Polish
# 2: French
# 1: German
TRACKS=(
    "3 ORIGINAL (English)"
    "0 AI DUBBING (Polish)"
    "2 AI DUBBING (French)"
    "1 AI DUBBING (German)"
)

# --- ENGINE ---

TEMP_DIR="temp_demo_segments"
mkdir -p "$TEMP_DIR"
LIST_FILE="segments_list.txt"
> "$LIST_FILE"

echo "🎬 Starting Demo Generation for $INPUT..."

# Function to extract segment with specific audio track and label
extract_labeled_seg() {
    local start=$1
    local dur=$2
    local track_idx=$3
    local label=$4
    local out=$5
    
    echo "  -> Processing: $label ($start)"
    
    ffmpeg -hide_banner -loglevel error -y -ss "$start" -t "$dur" -i "$INPUT" 
        -map 0:v:0 -map 0:a:"$track_idx" 
        -vf "drawtext=fontfile='${FONT}':text='${label}':fontcolor=white:fontsize=72:box=1:boxcolor=black@0.5:boxborderw=10:x=(w-text_w)/2:y=h-th-100" 
        -c:v mpeg4 -q:v 2 -c:a aac -b:a 192k "$out"
}

SEG_COUNTER=0
for SEG_INFO in "${SEGMENTS[@]}"; do
    START=$(echo $SEG_INFO | cut -d' ' -f1)
    DUR=$(echo $SEG_INFO | cut -d' ' -f2)
    
    for TRACK_INFO in "${TRACKS[@]}"; do
        T_IDX=$(echo $TRACK_INFO | cut -d' ' -f1)
        T_LABEL=$(echo $TRACK_INFO | cut -d' ' -f2-)
        
        OUT_FILE="${TEMP_DIR}/seg_${SEG_COUNTER}.mp4"
        extract_labeled_seg "$START" "$DUR" "$T_IDX" "$T_LABEL" "$OUT_FILE"
        
        echo "file '$OUT_FILE'" >> "$LIST_FILE"
        ((SEG_COUNTER++))
    done
done

echo "🔗 Concatenating segments into final video..."
ffmpeg -hide_banner -loglevel error -y -f concat -safe 0 -i "$LIST_FILE" -c copy "$OUTPUT"

echo "🧹 Cleaning up..."
rm -rf "$TEMP_DIR"
rm "$LIST_FILE"

echo "✅ DONE! Created: $OUTPUT"
