from pathlib import Path
import json
root = Path(r'D:\data\lsy\vm_lsy_parent\lsy')
base = root / r'01_data\single_cell\intermediate\scfoundation_embeddings\fullgene_cell_all_v1_shards'
ds = 'iccite_seq_tcell_2025'
sh0 = base / 'shard0_of_2' / ds
sh1 = base / 'shard1_of_2' / ds
bar = root / r'01_data\single_cell\intermediate\paired_matrices\iccite_seq_tcell_2025\rna_full_counts\rna_full_counts_barcodes.tsv'
cells = [x.strip() for x in bar.read_text(encoding='utf-8').splitlines() if x.strip()]
sh0.mkdir(parents=True, exist_ok=True)
emb = sh0 / 'embeddings.npy'
if not emb.exists():
    raise SystemExit(f'missing {emb}')
sh0_cells = cells[0::2]
(sh0 / 'barcodes.tsv').write_text('\n'.join(sh0_cells) + '\n', encoding='utf-8')
if (sh1 / 'genes_used.tsv').exists():
    (sh0 / 'genes_used.tsv').write_text((sh1 / 'genes_used.tsv').read_text(encoding='utf-8'), encoding='utf-8')
meta = json.loads((sh1 / 'embedding_metadata.json').read_text(encoding='utf-8'))
meta['n_cells'] = len(sh0_cells)
meta['shard_index'] = 0
meta['complete'] = True
(sh0 / 'embedding_metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
print('repaired', len(sh0_cells))
