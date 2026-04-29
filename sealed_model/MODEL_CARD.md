# RNA2Phospho dual-output final v1

Input: bulk RNA expression matrix with gene symbols.

Outputs:
- CPTAC/PDC mass-spectrometry phosphosite predictions.
- TCGA/TCPA phospho-RPPA antibody predictions.

The two output heads are separate because phosphosite mass spectrometry and RPPA antibody signals have different measurement semantics.

Samples: 7465 total, 1431 CPTAC/PDC, 6034 TCGA/TCPA.
Input genes: 33233.
CPTAC phosphosite outputs: 16049.
TCPA phospho-antibody outputs: 76.
Checkpoint SHA256: e30d91a80cd39f57ed463a9a1866a7ea0c31efd42d7f156863f6ae5cd01b6ebf.
