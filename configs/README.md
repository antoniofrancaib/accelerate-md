# Unified Configuration System

AccelMD uses a single unified configuration file for all experiments. Switch between ALDP and GMM experiments by changing the `experiment_type` field.

## Configuration File

**`experiment.yaml`** - Single configuration file for all experiment types
- Set `experiment_type: "aldp"` for ALDP cartesian experiments  
- Set `experiment_type: "gmm"` for Gaussian Mixture Model experiments
- Automatically generates appropriate experiment names
- Automatically sets `target.type` based on `experiment_type`

## How to Use

1. **Edit the experiment type** in `configs/experiment.yaml`:
   ```yaml
   experiment_type: "aldp"  # or "gmm"
   ```

2. **Run the experiment**:
   ```bash
   sbatch run_experiment.sh
   ```

3. **Customize parameters** (optional):
   - Edit the `aldp` or `gmm` sections for experiment-specific settings
   - Modify shared `pt`, `trainer`, or `device` settings
   - All experiments use the same PT and trainer settings

## Configuration Structure

```yaml
# Top-level experiment selection
experiment_type: "aldp"  # "aldp" or "gmm"
name: "unified_experiment_auto"  # Auto-generated if not specified
device: "cuda"

# Shared settings for all experiments
pt:
  temperatures: [300.0, 400.0, 500.0]
  num_chains: 32
  num_steps: 200000
  swap_interval: 500
  step_size: 1e-4

trainer:
  realnvp:
    model:
      hidden_dim: 512
      n_couplings: 150
    training: {...}

# Experiment-specific configurations (no overrides)
aldp:
  target:
    data_path: "./datasets/aldp/position_min_energy.pt"
    transform: cartesian
    env: implicit
  system: {...}

gmm:
  gmm_params:
    dim: 3
    n_mixes: 8
    loc_scaling: 5.0
    mode_arrangement: "random"
    grid_range: [-6.0, 6.0]
```

### How It Works

1. **Experiment type determines target**: `experiment_type: "aldp"` automatically sets `target.type: "aldp"`
2. **Shared settings**: All experiments use the same `pt` and `trainer` configurations
3. **Experiment-specific data only**: Only experiment-specific data paths and parameters are in the `aldp`/`gmm` sections
4. **No redundancy**: No duplicate information or overrides needed

### Auto-Generated Names

When using `name: "unified_experiment_auto"`, the system generates descriptive names:

- **ALDP**: `aldp_cart_{n_reps}rep_{n_couplings}coup_{hidden_dim}hidden`
- **GMM**: `gmm_{dim}dim_{n_mixes}mod_{n_reps}rep_{mode_arrangement}`

Examples:
- `aldp_cart_3rep_150coup_512hidden`
- `gmm_3dim_8mod_3rep_random`

## Key Features

### Simple Experiment Switching
Switch between ALDP and GMM experiments by changing one line:
```yaml
experiment_type: "gmm"  # Change from "aldp" to "gmm"
```

### Unified Settings
- **Same PT settings**: Both experiments use identical parallel tempering configuration
- **Same trainer settings**: Both experiments use identical model architecture and training parameters
- **Same evaluation settings**: Consistent evaluation across experiment types
- **Automatic target setup**: `target.type` automatically set based on `experiment_type`

### Clean Configuration
- No redundant `target.type` fields in experiment sections
- No override sections needed
- Only experiment-specific data and parameters specified
- Minimal configuration with maximum functionality

### Validation
- Validates experiment types (`aldp`, `gmm`)
- Ensures required configurations are present
- Provides clear error messages for misconfigurations

## Customization Examples

### Switch to GMM experiments:
```yaml
experiment_type: "gmm"  # That's it!
```

### Modify shared PT settings (affects both ALDP and GMM):
```yaml
pt:
  temperatures: [280.0, 320.0, 360.0, 400.0]  # New temperature schedule
  num_steps: 300000                           # Longer runs
```

### Use different model architecture (affects both experiments):
```yaml
trainer:
  realnvp:
    model:
      hidden_dim: 1024        # Larger network
      n_couplings: 200        # More coupling layers
```

### Modify GMM parameters only:
```yaml
gmm:
  gmm_params:
    dim: 5          # 5D instead of 3D
    n_mixes: 12     # More mixture components
```

## Troubleshooting

### "Missing 'experiment_type' field"
- Ensure your config file has `experiment_type: "aldp"` or `"gmm"` at the top level

### "No configuration found for experiment_type"
- Ensure the corresponding configuration section exists (`aldp:` or `gmm:`)
- Check that the section is not empty

### "Invalid experiment_type"
- `experiment_type` must be exactly `"aldp"` or `"gmm"` (case-sensitive)

# Guide to High-Dimensional GMM Experiments

This README provides guidelines for running GMM experiments with arbitrary dimensions. The codebase now supports running experiments on 2D, 5D, 60D, or any other dimensionality by simply changing the configuration.

## Configuration Guidelines

### Basic Configuration Changes for Higher Dimensions

1. **Dimension**: Set `gmm.dim` to the desired dimensionality (e.g., 5, 60, 100)
2. **Unified Grid Range**: Use `grid_range: [-6.0, 6.0]` instead of dimension-specific ranges
3. **Mode Arrangement**: For very high dimensions (>10D), prefer `mode_arrangement: "random"` over "grid"
4. **Number of Mixtures**: Consider reducing `n_mixes` for higher dimensions (e.g., 8-16 for 5D, 4-8 for 60D)
5. **RealNVP Coupling Layers**: Set `n_couplings` to at least match the dimensionality (or 1.5-2x for better performance)

### Example: Minimally Viable Configuration for 100D

```yaml
gmm:
  dim: 100                    # Set your desired dimension
  n_mixes: 4                  # Reduce for higher dimensions
  loc_scaling: 6.0
  custom_modes: false
  mode_arrangement: "random"  # Random works better for very high dimensions
  grid_range: [-6.0, 6.0]     # Unified range for all dimensions
  uniform_mode_scale: 0.2     # Controls spread of each Gaussian component

trainer:
  realnvp:
    model:
      dim: 100                # Will be auto-synced with gmm.dim
      hidden_dim: 512         # Increase for higher dimensions
      n_couplings: 100        # Rule of thumb: at least 1x the dimension
      use_permutation: true
```

## Troubleshooting High-Dimensional Runs

If your high-dimensional GMM experiment gets stuck or produces errors:

1. **Check Job Status**:
   ```
   ./check_job.py --job-id YOUR_JOB_ID
   ```

2. **Memory Issues**: Higher dimensions require more memory. For dimensions >100:
   - Reduce `n_samples` in the trainer configuration
   - Request more memory in your job submission script
   - Use `mode_arrangement: "random"` instead of "grid"

3. **Slow Dataset Creation**:
   - For dimensions >50, dataset creation can take longer
   - Monitor progress in debug logs
   - Consider reducing the number of samples temporarily for testing

4. **Visualization Limitations**:
   - For D>2, visualizations will show 2D projections
   - Higher-dimensional structures may not be fully visible in projections

## Example Configurations

See the following example configurations:
- `gmm.yaml` - Standard 2D GMM experiment
- `gmm5d.yaml` - 5D GMM experiment
- `gmm60d.yaml` - 60D GMM experiment

## Best Practices

1. **Start Small**: Test with lower dimensions first, then scale up
2. **Debug Flags**: Enable debug logging for better visibility
3. **Scale Appropriately**: Adjust `n_mixes`, `n_samples`, and architecture as dimensions increase
4. **Check Results**: Verify swap rates and other metrics to ensure the model is learning properly

If you encounter any other issues, please contact the development team. 