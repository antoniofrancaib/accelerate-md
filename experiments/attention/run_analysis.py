#!/usr/bin/env python3
"""
Simple runner script for attention analysis.

This script makes it easy to run the attention vs graph connectivity analysis
with proper error handling and progress reporting.
"""

import sys
import os
from pathlib import Path

def main():
    """Run the attention analysis with error handling."""
    
    # Check if we're in the right directory
    project_root = Path(__file__).parent.parent.parent
    if not (project_root / "src" / "accelmd").exists():
        print("❌ Error: Could not find AccelMD source code.")
        print(f"Expected to find: {project_root / 'src' / 'accelmd'}")
        print("Please run this script from the project root directory:")
        print("  cd /path/to/accelerate-md")
        print("  python experiments/attention/run_analysis.py")
        sys.exit(1)
    
    # Check if checkpoints exist
    graph_checkpoint = project_root / "checkpoints/multi_graph/pair_0_1/models/best_model_epoch1475.pt"
    transformer_checkpoint = project_root / "checkpoints/multi_transformer/pair_0_1/models/best_model_epoch331.pt"
    
    if not graph_checkpoint.exists():
        print(f"❌ Error: Graph checkpoint not found at {graph_checkpoint}")
        print("Please ensure the graph model checkpoint exists.")
        sys.exit(1)
    
    if not transformer_checkpoint.exists():
        print(f"❌ Error: Transformer checkpoint not found at {transformer_checkpoint}")
        print("Please ensure the transformer model checkpoint exists.")
        sys.exit(1)
    
    # Check if data exists
    data_path = project_root / "datasets/pt_dipeptides/AA/pt_AA.pt"
    if not data_path.exists():
        print(f"❌ Error: AA dipeptide data not found at {data_path}")
        print("Please ensure the AA dipeptide dataset exists.")
        sys.exit(1)
    
    print("✅ All prerequisites found!")
    print("🚀 Starting attention analysis...")
    print()
    
    # Change to project root directory
    os.chdir(project_root)
    
    # Add src directory to Python path for accelmd imports
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    # Add attention directory to path for analysis module
    attention_path = project_root / "experiments" / "attention"
    if str(attention_path) not in sys.path:
        sys.path.insert(0, str(attention_path))
    
    # Import and run analysis
    try:
        from attention_analysis import AttentionAnalyzer
        
        analyzer = AttentionAnalyzer()
        analyzer.run_analysis()
        
        print()
        print("🎉 Analysis completed successfully!")
        print("📊 Check the results in experiments/attention/")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all required packages are installed:")
        print("  pip install torch numpy matplotlib seaborn scikit-learn scipy pandas")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        print("Please check the error above and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()