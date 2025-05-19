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