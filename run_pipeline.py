#!/usr/bin/env python3
"""
Main entry point for the Meme Virality Prediction System
========================================================

This script runs the complete pipeline:
1. Load and preprocess data
2. Train all 3 models (Hypergraph, Image, Text)
3. Generate predictions
4. Ensemble and evaluate
5. Save trained models

Usage:
    python run_pipeline.py
"""

import sys
import os

# Add main_approach folder to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run main
from main_approach.main import main

if __name__ == "__main__":
    # ensure we capture all printed output to a log file
    log_name = "pipeline_output.log"
    log_path = os.path.join(os.getcwd(), log_name)
    print(f"Logging pipeline output to {log_path}")
    # redirect both stdout and stderr
    from contextlib import redirect_stdout, redirect_stderr
    with open(log_path, "w", encoding="utf-8") as log_file:
        with redirect_stdout(log_file), redirect_stderr(log_file):
            main()
    # also write a copy to terminal when finished
    print(f"Pipeline finished; output saved in {log_path}")
