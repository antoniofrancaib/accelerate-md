## GMM PT Dataset

This section documents the specific GMM Parallel Tempering dataset we generated for training flow-based samplers.

### Configuration
- **config file:** `configs/pt/gmm.yaml`
- **num_steps:** 100000
- **burn_in:** 1000
- **swap_interval:** 100
- **thinning:** none

### GMM Distribution
- **dim:** 2
- **num_modes:** 5
- **loc_scaling:** 2.0
- **mixture weights:** uniform (all components equally weighted)
- **covariances:** identity matrices scaled by a factor (using softplus transformation on log_var)

### Parallel Tempering
- **num_replicas:** 10
- **temp_low:** 1.0
- **temp_high:** 100.0
- **schedule:** geometric ("geom")
- **num_chains:** 1 per replica

### MCMC Kernel
- **type:** Langevin dynamics with Metropolis-Hastings correction
- **step_size:** 1e-4
- **chain_length:** 100000
- **burn_in:** 1000
- **thinning:** none

### Dataset Generation Process
1. We initialized the GMM target distribution with 5 mixture components in 2D space
2. Setup a temperature ladder with 10 temperatures from 1.0 to 100.0 using geometric spacing
3. Sampled initial points from the GMM for each temperature and chain
4. Ran Parallel Tempering with swap proposals every 100 steps
5. Collected samples after a burn-in period of 1000 steps
6. Extracted adjacent temperature pairs by flattening the chain and step dimensions
7. Saved the final dataset with pairs of samples from adjacent temperatures

### Files Produced
- **raw trajectories:** `data/pt/gmm_PT_trajectories.npz`
  - Contains complete sampling trajectories for all temperatures
  - Shape: [num_temps=10, num_chains=1, n_samples=99000, dim=2]

- **paired samples:** `data/pt/gmm_pairs.npz`
  - Contains pairs of samples from adjacent temperatures
  - Each pair has shape [num_chains*n_samples=99000, dim=2]
  - 9 pairs in total (between 10 temperatures)

### Dataset Statistics
The dataset verification confirms that:
- All pairs have consistent shapes of (99000, 2)
- The variance of samples increases with temperature as expected
- Mean values remain relatively stable across temperatures
- The dataset contains no anomalies or issues
