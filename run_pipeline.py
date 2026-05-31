"""
Pipeline Runner — Executes all phases sequentially:
  Phase 1: Generate synthetic dataset
  Phase 2: Preprocess data
  Phase 3: Train neural network
  (Phases 4–6 run inside the Streamlit app)
"""

import sys
import time

def run_phase(phase_num: int, name: str, func):
    sep = "═" * 65
    print(f"\n{sep}")
    print(f"  PHASE {phase_num}: {name}")
    print(sep)
    t0 = time.time()
    result = func()
    elapsed = time.time() - t0
    print(f"\n  ✅ Phase {phase_num} completed in {elapsed:.1f}s")
    return result


if __name__ == "__main__":
    print("=" * 65)
    print("  CA Intelligence Suite -- Full Pipeline")
    print("=" * 65)

    # Phase 1
    from generate_dataset import generate_dataset
    run_phase(1, "Synthetic Dataset Generation", generate_dataset)

    # Phase 2
    from preprocess import preprocess
    run_phase(2, "Data Preprocessing", preprocess)

    # Phase 3
    from train_model import train
    run_phase(3, "Neural Network Training", train)

    print("\n" + "=" * 65)
    print("  All training phases complete!")
    print("  Launch the Streamlit app with:")
    print("     streamlit run app.py")
    print("=" * 65)
