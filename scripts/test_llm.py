import time
import subprocess
import argparse


def print_smi(label):
    print(f"\n--- {label} ---")
    subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total", "--format=csv,noheader"])


def run_test(gpu_id, model_path):
    print_smi("PRZED ZAŁADOWANIEM")

    try:
        from llama_cpp import Llama

        print(f"\nŁadowanie modelu na cuda:{gpu_id}...")

        llm = Llama(
            model_path=model_path,
            n_gpu_layers=99,  # Wszystko na GPU
            main_gpu=gpu_id,
            n_ctx=8192,
            verbose=False,
        )

        print_smi(f"MODEL ZAŁADOWANY NA GPU {gpu_id}")
        print("\nOczekiwanie 10 sekund przed zwolnieniem...")
        time.sleep(10)

        del llm
        print("\nModel usunięty z pamięci.")

    except Exception as e:
        print(f"\nBŁĄD: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True, help="ID procesora GPU (0 lub 1)")
    parser.add_argument("--model", type=str, default="/app/data/models/google_gemma-3-12b-it-Q4_K_M.gguf")
    args = parser.parse_args()

    run_test(args.gpu, args.model)
