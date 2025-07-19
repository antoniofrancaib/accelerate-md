# Graph Architecture Improvements for PT Swap Flows

## Problem Diagnosis

The original graph architecture showed minimal improvement over naive swapping (0.3267 → 0.3314, only ~1.4%). Analysis revealed several key issues:

1. **Overly Conservative Transformations**: `±5%` scaling was too restrictive
2. **Simple Coupling Masks**: Alternating atoms didn't respect molecular structure  
3. **No Temperature Awareness**: Model couldn't condition on temperature differences
4. **Limited Feature Richness**: Only basic distance features
5. **Shallow Networks**: Single hidden layer MLPs insufficient for complexity

## Enhanced Graph Architecture Solutions

### 🎯 **1. Expressive Transformations**
- **Increased Scale Range**: `±5%` → `±50%` for meaningful coordinate changes
- **Configurable Scaling**: New `scale_range` parameter allows tuning
- **Deeper Networks**: 3-layer MLPs instead of single layer

```python
# Before: torch.tanh(raw_scales) * 0.05  # ±5%
# After:  torch.tanh(raw_scales) * self.scale_range  # ±50%
```

### 🧬 **2. Chemical-Aware Coupling Masks**
- **Phase 0**: Transform heavy atoms (C,N,O), preserve H structure
- **Phase 1**: Transform H atoms, preserve heavy atom backbone
- **Motivation**: Respects chemical hierarchy and molecular stability

```python
def _create_coupling_mask(self, atom_types, masked_elements, device):
    if self.phase == 0:
        coupling_mask = atom_types > 0  # Transform heavy atoms
    else:
        coupling_mask = atom_types == 0  # Transform hydrogens
```

### 🌡️ **3. Temperature Conditioning**
- **Direct Temperature Input**: Source/target temperatures as features
- **Physics-Informed**: Model learns temperature-dependent transformations
- **Broadcast**: Temperature features added to all atoms

```python
# Add temperature features: [source_temp, target_temp]
temp_features = torch.tensor([source_temp, target_temp], device=device)
temp_features = temp_features.expand(B, N, -1)  # [B, N, 2]
combined_features = torch.cat([node_features, coords, temp_features], dim=-1)
```

### 📐 **4. Richer Molecular Features**
Enhanced edge features from basic distance to 4D geometric representation:
- **Distance**: Bond length information
- **Angles**: Spatial orientation relative to reference
- **Dihedral Indicators**: Torsional geometry (simplified)  
- **Bond Type**: Chemical bond classification based on distance

```python
edge_features = torch.cat([
    distances,           # [E, 1] Bond lengths
    angles,              # [E, 1] Bond angles  
    dihedral_indicator,  # [E, 1] Torsional info
    bond_type,           # [E, 1] Chemical classification
], dim=-1)  # [E, 4]
```

### 🏗️ **5. Increased Model Capacity**
- **Flow Layers**: 8 → 12 layers for more complex transformations
- **Hidden Dimensions**: 512 → 1024 for richer representations
- **Atom Embeddings**: 128 → 256 dimensions
- **Message Passing**: 4 → 6 layers for deeper graph reasoning

### ⚙️ **6. Optimized Training**
- **More Data**: `subsample_rate: 100 → 10` (10x more training examples)
- **Lower Learning Rate**: `0.0001 → 0.00005` for stability
- **More Epochs**: `100 → 200` for convergence
- **Better Scheduling**: Warmup + patience adjustments

## Architecture Equivalence Testing

### 📊 **Systematic Comparison Framework**
Created three matched configurations for fair comparison:

1. **`configs/AA_simple.yaml`**: Baseline coordinate-to-coordinate flow
2. **`configs/AA.yaml`**: Enhanced graph architecture 
3. **`configs/AA_transformer.yaml`**: Attention-based architecture

**Key**: Identical training hyperparameters, data, and capacity for fair comparison.

### 🧪 **Testing Script: `test_architecture_equivalence.py`**
Automated testing pipeline that:
- Runs all three architectures systematically
- Extracts performance metrics (naive/flow acceptance)
- Computes improvement percentages  
- Analyzes training efficiency
- Generates recommendations

**Usage**:
```bash
conda activate accelmd
python test_architecture_equivalence.py
```

### 📈 **Evaluation Metrics**
- **Primary**: Flow acceptance improvement over naive
- **Secondary**: Absolute flow acceptance rate
- **Efficiency**: Improvement per training hour
- **Robustness**: Consistency across temperature pairs

## Expected Improvements

### 🎯 **Target Performance**
- **Baseline**: ~1.4% improvement (current graph)
- **Enhanced Graph**: >10% improvement target
- **Best Case**: 20-50% improvement over naive

### 🔬 **Scientific Insights**
The enhanced architecture tests key hypotheses:

1. **Inductive Biases vs Data-Driven Learning**:
   - Graph: Explicit molecular structure encoding
   - Transformer: Learned attention patterns
   - Simple: Minimal assumptions

2. **Temperature Conditioning Impact**:
   - Direct temperature features vs implicit learning

3. **Molecular Feature Richness**:
   - Geometric features vs coordinate-only approaches

## Next Steps & Ablation Studies

### 🔍 **Individual Component Testing**
1. **Scale Range Ablation**: Test `[0.05, 0.1, 0.2, 0.5]` to find optimum
2. **Temperature Conditioning**: On/off comparison
3. **Chemical Masking**: Chemical vs alternating masks
4. **Feature Richness**: Distance-only vs full geometric features
5. **Network Depth**: Shallow vs deep MLP comparison

### 🌡️ **Temperature Pair Studies**
- **Closer Pairs**: `[0,1]` vs `[1,2]` (different energy scales)
- **Wider Gaps**: `[0,2]`, `[0,3]` (larger temperature differences)
- **Cross-Peptide**: Train on AA, test on AK/AS

### 📊 **Data Scaling Studies**
- **Data Amount**: `subsample_rate = [1, 5, 10, 50, 100]`
- **Batch Size**: Impact on gradient quality
- **Training Length**: Convergence analysis

## Implementation Notes

### ⚠️ **Breaking Changes**
- Graph coupling layer constructor signature changed (new parameters)
- Forward method requires temperature arguments
- Message passing layer expects 4D edge features

### 🔧 **Backward Compatibility**
- All new parameters have sensible defaults
- Temperature conditioning can be disabled
- Fallback to distance-only features if coordinates unavailable

### 🚀 **Performance Considerations**
- Larger models require more GPU memory
- Deeper networks increase training time
- Richer features add computational overhead
- Consider mixed precision training for very large models

---

## Quick Start

1. **Test Enhanced Graph**:
   ```bash
   conda activate accelmd
   python main.py --config configs/AA.yaml --temp-pair 0 1
   ```

2. **Full Architecture Comparison**:
   ```bash
   python test_architecture_equivalence.py
   ```

3. **Monitor Progress**:
   - Check `architecture_comparison_results.csv` for intermediate results
   - Final results in `final_architecture_comparison.json` 