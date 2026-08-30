"""SCP682-ChemState-MODZ model definition."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChemStateMODZ(nn.Module):
    def __init__(self, n_sites: int, n_cells: int, fingerprint_dim: int, dropout: float = 0.05):
        super().__init__()
        self.n_sites = n_sites
        self.n_cells = n_cells
        self.chemical_first = nn.Linear(fingerprint_dim, 128, bias=False)
        self.chemical_output = nn.Linear(128, 64)
        self.encoder_first = nn.Linear(n_sites * 2 + 64, 128, bias=False)
        self.encoder_hidden = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.3),
            nn.Dropout(dropout),
        )
        self.encoder_output = nn.Linear(64, 64)
        context_dim = n_cells + 3
        self.decoder_first = nn.Sequential(
            nn.Linear(64 + 64 + context_dim, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.3),
            nn.Dropout(dropout),
        )
        self.decoder_hidden = nn.Sequential(
            nn.Linear(64, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.3),
            nn.Dropout(dropout),
        )
        self.decoder_output = nn.Linear(128, n_sites)

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        baseline = inputs["baseline"]
        chemical = self.chemical_output(self.chemical_first(inputs["fingerprint"]))
        state = torch.cat([baseline, inputs["input_mask"].float(), chemical], dim=1)
        latent = self.encoder_output(self.encoder_hidden(self.encoder_first(state)))
        cell = F.one_hot(inputs["cell_ids"], num_classes=self.n_cells).float()
        context = torch.cat([
            cell,
            inputs["condition"],
            -torch.ones((len(cell), 1), dtype=cell.dtype, device=cell.device),
        ], dim=1)
        decoded = self.decoder_output(
            self.decoder_hidden(self.decoder_first(torch.cat([latent, chemical, context], dim=1)))
        )
        return decoded - baseline
