"""Record completed full-embedding provenance and a reusable sample index."""
from common import *


def main():
    cache=OUT/'cache/esm2_residue'
    completion=json.loads((cache/'completion.json').read_text())
    index=json.loads((cache/'index.json').read_text())
    df=pd.read_csv(SOURCE)
    splits=dict(np.load(SPLIT)); split_label=np.empty(len(df),dtype=object)
    for k,ids in splits.items(): split_label[ids]=k.removesuffix('_idx')
    mapping=np.asarray(index['row_mapping'])
    pd.DataFrame(dict(row_id=np.arange(len(df)),sample_id=df.iloc[:,0],split=split_label,
        unique_sequence_index=mapping,sequence_length=df.Sequence.str.len(),
        sequence_sha256=[index['sequence_sha256'][i] for i in mapping],
        embedding_file=[f'{i:05d}.npy' for i in mapping])).to_csv(cache/'sample_index.csv',index=False)
    digest_path=cache/'file_sha256.json'
    if not digest_path.exists():
        dump(digest_path,{p.name:sha(p) for p in sorted(cache.glob('*.npy'))})
    run=json.loads((OUT/'reports/run_manifest.json').read_text())
    run['residue_cache']=dict(path=str(cache),completion=completion,index_sha256=sha(cache/'index.json'),
        file_hash_manifest=str(digest_path),file_hash_manifest_sha256=sha(digest_path),sample_index_sha256=sha(cache/'sample_index.csv'))
    dump(OUT/'reports/run_manifest.json',run)
    with (OUT/'reports/environment_audit.md').open('a') as f:
        f.write('\n## Completed full embedding cache\n\n'+json.dumps(run['residue_cache'],indent=2)+'\n')
    print(json.dumps(run['residue_cache'],indent=2))


if __name__=='__main__': main()
