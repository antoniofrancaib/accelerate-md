Accelerating Molecular Dynamics with swap-flow proposals for Parallel Tempering. This repository trains normalizing flows that morph coordinates between adjacent temperatures to increase swap acceptance.

### Environment

```bash
conda env create -f environment.yml
conda activate accelmd
```

### Data layout

Place PT dipeptide datasets under `datasets/pt_dipeptides/<PEP>/` with:

```text
pt_<PEP>.pt          # PT trajectory tensor or dict (see shapes below)
atom_types.pt        # LongTensor [N] with atom type indices
adj_list.pt          # LongTensor [E,2] or [2,E] edges 
ref.pdb              # Reference PDB (used for dipeptide target)
```

PT tensor formats supported by the loader:
- [temps, chains, steps, coords] with coords = 3N
- [steps*chains, temps, coords]

Coordinates are reshaped to [B, N, 3]; optional preprocessing: centering, chirality filtering, random rotation augmentation.

### What’s implemented

- Training/evaluation via `main.py`
- Single-peptide and multi-peptide modes
- Architectures: simple (`PTSwapFlow`), graph (`PTSwapGraphFlow`), transformer (`PTSwapTransformerFlow`)
- Physics bases: generic dipeptide potential (implicit solvent). Energies run on CPU.
- Output management per temperature pair with best-checkpoint tracking

### Quick start

Single peptide (e.g., AA) using the simple flow:

```bash
conda activate accelmd && python -u main.py --config configs/AA_simple.yaml --temp-pair 0 1 
```

Multi‑peptide training (graph or transformer only):

```bash
conda activate accelmd && python -u main.py --config configs/multi_graph.yaml --temp-pair 0 1
```

Evaluate a saved checkpoint and report swap acceptance (prints mean ± CI over resampled sets):

```bash
conda activate accelmd && python -u main.py --config configs/multi_transformer.yaml \
  --evaluate --temp-pair 0 1 \
  --checkpoint outputs/<experiment>/pair_0_1/models/best_model_epochXXX.pt \
  --num-eval-samples 20000 --eval-repeats 5
```

Omit `--temp-pair` to iterate through all pairs listed in the config.

### Configuration overview

Configs live in `configs/`. Two ready-to-use examples are provided: `AA_simple.yaml`, `multi_graph.yaml`, `multi_transformer.yaml`.

Single‑peptide minimal example: `configs/AA_simple.yaml`

Multi‑peptide minimal example (requires `architecture: graph` or `transformer`): `configs/multi_graph.yaml` or `configs/multi_transformer.yaml`

Notes:
- In single mode, `peptide_code` auto-fills `data.pt_data_path` and `data.molecular_data_path`.
- In multi mode, datasets are discovered from `datasets/pt_dipeptides/<PEP>/` for all peptides listed.
- For `AX`, the ALDP target is used; otherwise the dipeptide target uses `ref.pdb` with implicit solvent.
- For the simple architecture, `num_atoms` is inferred from data and injected before model build.

### Training and evaluation details

The trainer minimises a weighted sum of bidirectional NLL (low→high, high→low) and an acceptance‑oriented loss. Weights can be scheduled (`nll_start/end`, `acc_start/end`, `warmup_epochs`). Gradients are clipped (`clip_grad_norm`). LR scheduling uses `ReduceLROnPlateau` on validation loss. Energies and Boltzmann bases are evaluated on CPU. During training a checkpoint is saved whenever validation loss improves; after training, the most recent best checkpoint is used for evaluation.

Evaluation computes naïve and flow‑based swap acceptance over resampled subsets; in multi‑peptide mode, each peptide is evaluated separately and results are printed. Internally, metrics can be saved to `metrics/swap_acceptance.json` when invoked with saving enabled.


### CLI reference (`main.py`)

```text
--config PATH                 YAML config
--temp-pair i j               Train/evaluate this temperature pair (omit to run all)
--epochs N                    Override num_epochs for quick tests (training only)
--evaluate                    Run evaluation instead of training
--checkpoint PATH             Checkpoint to load for evaluation or resume
--num-eval-samples N          Samples per evaluation run (default 20000)
--eval-repeats K              Resampled runs to estimate CI (default 1)
--resume PATH                 Resume training from checkpoint
```

### Troubleshooting

- If evaluation complains about missing files, check `datasets/pt_dipeptides/<PEP>/` contains `pt_<PEP>.pt`, `atom_types.pt`, `adj_list.pt`, and `ref.pdb`.
- Temperature indices in `temperature_pairs` refer to positions in `temperatures.values` (not Kelvin).
- OpenMM runs on CPU; adjust `system.n_threads` if needed.

### License

MIT
