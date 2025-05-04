import numpy as np
import torch
import os
import tempfile
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.train_flow import GMMPairDataset, make_pair_loader

def _mock_npz(tmp):
    temps = np.arange(3)          # T0, T1, T2 → 2 transitions
    pairs = np.random.randn(2, 2, 100, 2)   # [k, {src,tgt}, N, dim]
    path = os.path.join(tmp, "mock.npz")
    np.savez(path, pairs=pairs, temps=temps)
    return path

def test_dataset_shapes(tmp_path):
    f = _mock_npz(tmp_path)
    ds = GMMPairDataset(f, transition_idx=0)
    assert len(ds) == 100
    x, y = ds[0]
    assert x.shape == (2,) and y.shape == (2,)

def test_dataloader_batch(tmp_path):
    f = _mock_npz(tmp_path)
    train_loader, val_loader = make_pair_loader(f, 1, batch_size=32, val_split=0.2, seed=0)
    xb, yb = next(iter(train_loader))
    assert xb.shape == (32, 2)
    assert yb.shape == (32, 2) 