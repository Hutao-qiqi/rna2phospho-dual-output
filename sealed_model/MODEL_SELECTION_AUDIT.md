# RNA2Phospho best deployable model v2 with total protein

Final model is a multi-engine deployable contract from one RNA input. It outputs four matrices:

- CPTAC/PDC mass-spectrometry phosphosite predictions.
- CPTAC/PDC mass-spectrometry total-protein predictions.
- TCPA phospho-RPPA antibody predictions.
- TCPA total-RPPA antibody predictions.

## Selected CPTAC/PDC phosphosite engine

`20260426_cptac_adversarial_ensemble_stack_v1`

Locked gene-site target contract. Median target Spearman = 0.4305588762701733. Targets with Spearman >= 0.5 = 4506.

## Selected CPTAC/PDC total-protein engine

`20260428_cptac_total_proteome_film_vae_z_direct_residual_v1`

Median target Spearman = 0.4846051523389646. Targets with Spearman >= 0.5 = 5222.

A v2 retrain on expanded `pancancer_multi_task_locked_v2` was performed, but its median target Spearman was 0.4729714323757167, so it is not selected.

## Selected TCPA/RPPA engine

`20260428_tcpa_pancancer_rppa_film_vae_z_direct_residual_v1`

Total antibody median Spearman = 0.6652301230803972. Phospho-antibody median Spearman = 0.6364467170343006.

## Rejected prototype

`20260429_rna2phospho_dual_output_final_v1` remains rejected because its CPTAC phosphosite median Spearman was only 0.30146750524109006.
