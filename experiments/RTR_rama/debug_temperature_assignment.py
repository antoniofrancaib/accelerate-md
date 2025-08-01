#!/usr/bin/env python3
"""
Debug script to verify temperature assignments for different pairs.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.accelmd.utils.config import load_config, create_run_config, setup_device


def debug_temperature_assignment(config_path: str, model_type: str):
    """Debug what temperatures are assigned to each pair."""
    
    print(f"\n=== Debugging {model_type} temperature assignments ===")
    print(f"Config: {config_path}")
    
    base_cfg = load_config(config_path)
    
    for pair in [(0,1), (1,2), (2,3), (3,4)]:
        print(f"\n--- Pair {pair} ---")
        device = setup_device(base_cfg)
        cfg = create_run_config(base_cfg, pair, device)
        
        temps = cfg["temperatures"]["values"]
        temp_low = temps[pair[0]]
        temp_high = temps[pair[1]]
        
        print(f"Temperature ladder: {temps}")
        print(f"Pair {pair}: T_low={temp_low:.1f}K, T_high={temp_high:.1f}K")
        print(f"Temperature ratio: {temp_high/temp_low:.3f}")


def main():
    """Debug both simple and transformer configs."""
    
    debug_temperature_assignment("configs/AA_simple_01.yaml", "simple")
    debug_temperature_assignment("configs/multi_transformer_01.yaml", "transformer")


if __name__ == "__main__":
    main()