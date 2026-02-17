#!/bin/bash
export PYTHONPATH="/app/F5-TTS/src"
/app/.venv/bin/python3 /app/F5-TTS/src/f5_tts/infer/infer_cli.py 
    --ref_audio "/app/output/debug_myth_test/refs/SPEAKER_00.wav" 
    --ref_text "Please, don't try anything that you're about to see us do at home." 
    --gen_text "This is a test of the emergency broadcast system. This is only a test." 
    --output_dir "/app/output" 
    --output_file "cli_test_out.wav" 
    --device "cuda"
