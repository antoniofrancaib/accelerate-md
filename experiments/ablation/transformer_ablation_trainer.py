#!/usr/bin/env python3
"""
Transformer Architecture Ablation Study - Training Phase (Transformer Layers)

This script trains transformer flow models with different transformer layer configurations
for systematic ablation analysis. Uses optimal 8 attention heads from Phase 1.

Usage:
    python figures/ablation/transformer_ablation_trainer.py
"""

import sys
import subprocess
from pathlib import Path
import time
from typing import Tuple
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class TrainingResult:
    """Container for training results."""
    config_name: str
    success: bool
    training_time: float
    convergence_epoch: int
    error_message: str = ""


class TransformerAblationTrainer:
    """Handles training of transformer ablation configurations."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.configs_dir = self.project_root / "configs" / "ablation"
        self.outputs_dir = self.project_root / "outputs"
        
    def run_training(self, config_path: Path) -> Tuple[bool, float, int, str]:
        """Run training for a single configuration."""
        print(f"Training with config: {config_path.name}")
        start_time = time.time()
        
        try:
            # Run main.py with the config
            cmd = ["python", "main.py", "--config", str(config_path)]
            result = subprocess.run(
                cmd, 
                cwd=self.project_root,
                capture_output=True, 
                text=True, 
                timeout=7200  # 2 hours timeout
            )
            
            training_time = time.time() - start_time
            
            if result.returncode != 0:
                error_msg = f"Training failed: {result.stderr}"
                print(f"❌ Training failed for {config_path.name}")
                print(f"Error: {result.stderr}")
                return False, training_time, 0, error_msg
                
            # Extract convergence epoch from output logs
            convergence_epoch = self._extract_convergence_epoch(config_path.stem)
            
            print(f"✅ Training completed for {config_path.name} in {training_time:.1f}s")
            return True, training_time, convergence_epoch, ""
            
        except subprocess.TimeoutExpired:
            error_msg = "Training timeout after 2 hours"
            print(f"⏱️ Training timeout for {config_path.name}")
            return False, time.time() - start_time, 0, error_msg
        except Exception as e:
            error_msg = f"Training error: {str(e)}"
            print(f"💥 Training error for {config_path.name}: {e}")
            return False, time.time() - start_time, 0, error_msg
    
    def _extract_convergence_epoch(self, experiment_name: str) -> int:
        """Extract the convergence epoch from training logs."""
        try:
            # Look for the best model checkpoint
            model_dir = self.outputs_dir / experiment_name / "pair_0_1" / "models"
            if model_dir.exists():
                # Find the best model file
                model_files = list(model_dir.glob("best_model_epoch*.pt"))
                if model_files:
                    # Extract epoch number from filename
                    epoch_nums = []
                    for f in model_files:
                        try:
                            epoch_num = int(f.stem.split("epoch")[1])
                            epoch_nums.append(epoch_num)
                        except (IndexError, ValueError):
                            continue
                    return max(epoch_nums) if epoch_nums else 0
        except Exception:
            pass
        return 0
    
    def train_all_configurations(self) -> list[TrainingResult]:
        """Train all transformer layer depth configurations."""
        print("=== Transformer Architecture Ablation Training (Phase 2: Layer Depth) ===")
        print("Training models with different transformer layer configurations")
        print("Using optimal attention heads value: 8 (from Phase 1)")
        
        # Configuration files for transformer layer ablation
        config_files = [
            "transformer_layers_1.yaml",
            "transformer_layers_2.yaml", 
            "transformer_layers_3.yaml",
            "transformer_layers_4.yaml"
        ]
        
        results = []
        
        for i, config_file in enumerate(config_files, 1):
            config_path = self.configs_dir / config_file
            experiment_name = config_path.stem
            
            print(f"\n--- Training {i}/{len(config_files)}: {experiment_name} ---")
            
            # Check if config file exists
            if not config_path.exists():
                print(f"❌ Config file not found: {config_path}")
                results.append(TrainingResult(
                    config_name=experiment_name,
                    success=False,
                    training_time=0.0,
                    convergence_epoch=0,
                    error_message="Config file not found"
                ))
                continue
            
            # Run training
            success, training_time, convergence_epoch, error_message = self.run_training(config_path)
            
            result = TrainingResult(
                config_name=experiment_name,
                success=success,
                training_time=training_time,
                convergence_epoch=convergence_epoch,
                error_message=error_message
            )
            
            results.append(result)
        
        return results
    
    def save_training_summary(self, results: list[TrainingResult]):
        """Save a summary of training results."""
        output_dir = Path("figures/ablation")
        output_dir.mkdir(exist_ok=True)
        
        summary_file = output_dir / "transformer_layers_training_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("Transformer Architecture Ablation (Phase 2: Layers) - Training Summary\n")
            f.write("=" * 70 + "\n\n")
            
            successful_count = sum(1 for r in results if r.success)
            f.write(f"Training Results: {successful_count}/{len(results)} successful\n")
            f.write(f"Using optimal attention heads: 8 (from Phase 1)\n\n")
            
            for result in results:
                status = "✅ SUCCESS" if result.success else "❌ FAILED"
                f.write(f"{result.config_name:<25} | {status:<10} | {result.training_time:8.1f}s")
                if result.success:
                    f.write(f" | Epoch {result.convergence_epoch}")
                else:
                    f.write(f" | {result.error_message}")
                f.write("\n")
            
            f.write(f"\nTotal training time: {sum(r.training_time for r in results):.1f}s\n")
            
            if successful_count > 0:
                f.write(f"\nNext step: Update transformer_ablation_evaluator.py config list and run evaluation.\n")
        
        print(f"\nTraining summary saved to: {summary_file}")


def main():
    """Main training execution."""
    trainer = TransformerAblationTrainer()
    
    # Train all configurations
    results = trainer.train_all_configurations()
    
    # Save summary
    trainer.save_training_summary(results)
    
    # Print final status
    successful_count = sum(1 for r in results if r.success)
    print(f"\n=== Training Complete: {successful_count}/{len(results)} successful ===")
    print("Phase 2: Transformer Layer Depth Ablation")
    
    if successful_count > 0:
        print("Ready for evaluation! Update config list in transformer_ablation_evaluator.py and run evaluation.")


if __name__ == "__main__":
    main() 