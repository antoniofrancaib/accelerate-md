# Ablation Study Configuration Files

This directory contains configuration files for systematic ablation studies of transformer flow architectures.

## Two-Phase Iterative Ablation Approach

The ablation studies follow a two-phase iterative optimization approach:

### Phase 1: Attention Mechanisms ✅ COMPLETED
**Objective**: Find optimal number of attention heads  
**Result**: 8 attention heads achieved highest SAR (97.8±1.3%)

Configuration files tested:
- `transformer_attn_1.yaml` - Single attention head
- `transformer_attn_4.yaml` - 4 attention heads  
- `transformer_attn_8.yaml` - 8 attention heads (OPTIMAL)
- `transformer_attn_16.yaml` - 16 attention heads

### Phase 2: Transformer Architecture 🔄 CURRENT
**Objective**: Find optimal number of transformer layers using 8 attention heads  

Current configuration files:
- `transformer_layers_1.yaml` - 1 transformer layer
- `transformer_layers_2.yaml` - 2 transformer layers (baseline)
- `transformer_layers_3.yaml` - 3 transformer layers
- `transformer_layers_4.yaml` - 4 transformer layers

### Future Phases
After Phase 2 analysis, create configs based on optimal layer depth:
- Phase 3: Position Encoding (RFF dimensions and scales)
- Phase 4: MLP Configuration (different layer sizes)
- Phase 5: Flow Architecture (4, 8, 12, 16 layers)

## Configuration Design Principles

Each phase builds on previous optimal values:
1. **Start with baseline configuration** 
2. **Incorporate optimal values from previous phases**
3. **Vary only the current parameter under study**
4. **Identify optimal value** from results table with confidence intervals
5. **Update baseline** with optimal value
6. **Move to next parameter** and repeat

## Usage

### Phase 2: Transformer Layer Depth

Train the models:
```bash
python figures/ablation/transformer_ablation_trainer.py
```

Evaluate the models:
```bash
python figures/ablation/transformer_ablation_evaluator.py
```

Results will be saved to:
- Training: `figures/ablation/transformer_layers_training_summary.txt`
- Evaluation: `figures/ablation/transformer_layers_ablation_results.txt`

## Expected Output Format

The scripts will generate formatted tables with confidence intervals like:

```
Transformer Flow Architecture Ablation Study Results (Phase 2: Layer Depth)

Configuration                    | SAR (%)         | RTR            | ESS            | Energy Conservation
--------------------------------|-----------------|----------------|----------------|-------------------
Baseline                       |                 |                |                |
Vanilla PT (no flow)           |   24.3         |   2.43        |   29          |             1.000
                               |                 |                |                |
Flow-Enhanced Results          |                 |                |                |
1 transformer layer            |  XX.X±X.X      |  X.XX±X.XX    |  XXX±XX       |         X.XXX±X.XXX
2 transformer layers (baseline)|  XX.X±X.X      |  X.XX±X.XX    |  XXX±XX       |         X.XXX±X.XXX
3 transformer layers           |  XX.X±X.X      |  X.XX±X.XX    |  XXX±XX       |         X.XXX±X.XXX
4 transformer layers           |  XX.X±X.X      |  X.XX±X.XX    |  XXX±XX       |         X.XXX±X.XXX
```

## Notes

- Each experiment trains for up to 100 epochs with early stopping
- All models use optimal 8 attention heads from Phase 1
- Models are evaluated with 95% confidence intervals (10 independent runs)
- Training failures are handled gracefully with zero scores
- Scripts include timeout protection (2 hours per training run)
- Results include training time and convergence epoch information 