import os
import sys
import time
import subprocess
import argparse

def print_smi(label):
    print(f"
--- {label} ---")
    subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total", "--format=csv,noheader"])

def run_test(gpu_id, model_path):
    print_smi("PRZED ZAŁADOWANIEM")
    
    try:
        from llama_cpp import Llama
        print(f"
Ładowanie modelu na cuda:{gpu_id}...")
        
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=99, # Wszystko na GPU
            main_gpu=gpu_id,
            n_ctx=8192,
            verbose=False
        )
        
        print_smi(f"MODEL ZAŁADOWANY NA GPU {gpu_id}")
        print("
Oczekiwanie 10 sekund przed zwolnieniem...")
        time.sleep(10)
        
        del llm
        print("
Model usunięty z pamięci.")
        
    except Exception as e:
        print(f"
BŁĄD: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True, help="ID procesora GPU (0 lub 1)")
    parser.add_argument("--model", type=str, default="/app/data/models/google_gemma-3-12b-it-Q4_K_M.gguf")
    args = parser.parse_args()
    
    run_test(args.gpu, args.model)
