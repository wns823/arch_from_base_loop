"""Base model (your existing architecture). Agents copy and edit this file. REPLACE with your model.

Conventions the harness relies on:
- class Model(nn.Module), built as Model(**data_loader.model_init_kwargs()).
- CONDITIONAL (module-level): False = the runner passes None for the condition (see data_loader.model_args).
- UPPER_CASE module-level constants are read as hyperparameters and shown in the LEDGER.
- Everything in this one file (agents may only write model.py and train_config.py)."""
import torch
import torch.nn as nn

CONDITIONAL = False
WIDTH = 64


class Model(nn.Module):
    def __init__(self, ctx_dim=8):
        super().__init__()
        self.tower = nn.Sequential(nn.Conv1d(4, WIDTH, 7, padding=3), nn.GELU(),
                                   nn.Conv1d(WIDTH, WIDTH, 5, padding=2), nn.GELU())
        self.head = nn.Linear(WIDTH, 1)

    def forward(self, seq, ctx=None):
        """seq [b, 4, 64] -> prediction [b]. ctx [b, 8] is ignored by the base model."""
        return self.head(self.tower(seq).amax(-1)).squeeze(-1)
