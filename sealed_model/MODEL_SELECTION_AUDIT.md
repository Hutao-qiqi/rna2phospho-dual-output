# RNA2Phospho best deployable model v1

Selected model is a two-engine deployable contract, not the rejected single-checkpoint prototype.

## Selected CPTAC/PDC phosphosite engine

`20260426_cptac_adversarial_ensemble_stack_v1`

Reason: best strict locked gene-site CPTAC/PDC model. Median target Spearman = 0.4305588762701733; 4506 sites >= 0.5. The ensemble uses cross-fitted per-target correlation weights over five component models. Component checkpoint availability was verified. Rerun checkpoints for baseline, target-stratified, and per-cancer branches reproduce original OOF predictions exactly.

## Rejected CPTAC high-number model

`20260426_cptac_pancancer_phosphoproteome_film_v3` has median Spearman = 0.5111875187448232, but it uses the older unlocked raw/refseq-like phosphosite table and raw logratio target contract. It is not selected as the main deployable model.

## Selected TCPA phospho-RPPA engine

`20260428_tcpa_pancancer_rppa_film_vae_z_direct_residual_v1`

Reason: best TCPA/RPPA phospho-antibody model. Phospho-antibody median Spearman = 0.6364467170343006; 70 phospho antibodies >= 0.5.

## Rejected joint prototype

`20260429_rna2phospho_dual_output_final_v1` is rejected. Its CPTAC phosphosite median Spearman is 0.30146750524109006, below the selected CPTAC engine. It is retained only as an interface prototype and must not be cited as the final model.
