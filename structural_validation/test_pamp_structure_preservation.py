import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import io
import numpy as np
import pandas as pd
import run_pamp_structure_preservation as r
import pamp_structure_statistics as s

def pdb_text(n=20,shift=None):
    shift=np.zeros(3) if shift is None else np.asarray(shift)
    lines=[]
    rng=np.random.default_rng(4)
    for i in range(n):
        for j,name in enumerate(['N','CA','C','O']):
            p=np.array([i*3.8,np.sin(i),np.cos(i)])+rng.normal(0,.2,3)+shift
            lines.append(f'ATOM  {i*4+j+1:5d} {name:^4s} ALA A{i+1:4d}    {p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}{1:6.2f}{80:6.2f}          {name[0]:>2s}')
    return '\n'.join(lines)+'\nTER\nEND\n'

class MetricsTests(unittest.TestCase):
    def test_rigid_translation_and_tm_score(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/'a.pdb',Path(d)/'b.pdb'
            a.write_text(pdb_text()); b.write_text(pdb_text(shift=[10,20,30]))
            r.validate_pdb(a,'A'*20)
            m=r.backbone_rmsd(a,b,10)
            self.assertLess(m['backbone_rmsd_angstrom'],1e-10)
            self.assertLess(m['mutation_site_ca_displacement_angstrom'],1e-10)
            self.assertAlmostEqual(r.usalign(a,b,'/root/tools/USalign')['tm_score'],1)
            # Missing backbone atoms must fail, not silently shift mutation index.
            b.write_text('\n'.join(b.read_text().splitlines()[1:]))
            with self.assertRaises(RuntimeError): r.backbone_rmsd(a,b,10)

    def test_metric_failure_never_marked_success(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d); p=out/'conf.npy'; np.save(p,np.full(20,80,dtype=np.float32))
            wt='A'*20; mut='C'+'A'*19
            cache={v:{'status':'success','plddt_path':str(p),'pdb_path':str(out/'missing.pdb')} for v in [wt,mut]}
            with patch.object(r,'usalign',side_effect=ValueError('injected alignment failure')):
                result=r.pair_metrics(out,wt,mut,cache,'/root/tools/USalign')
            self.assertEqual(result['structure_status'],'metric_error')
            self.assertIn('injected alignment failure',result['structure_error'])
            self.assertEqual(result['delta_mean_plddt'],0)
            cache[mut]={'status':'prediction_error','error':'injected folding failure'}
            result=r.pair_metrics(out,wt,mut,cache,'/root/tools/USalign')
            self.assertEqual(result['structure_status'],'prediction_error')
            self.assertEqual(result['wt_mean_plddt'],80)
            self.assertNotIn('mut_mean_plddt',result)

    def test_cache_requires_valid_files_and_configuration(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d); seq='A'*20
            _,p,n,j=r.cache_paths(out,seq)
            r.atomic_text(p,pdb_text()); buf=io.BytesIO(); np.save(buf,np.full(20,80.))
            r.atomic_write(n,buf.getvalue())
            meta={'settings':r.SETTINGS,'sequence':seq,'pdb_sha256':r.file_sha(p),'plddt_sha256':r.file_sha(n)}
            r.save_json(j,meta)
            self.assertIsNotNone(r.cached_structure(out,seq))
            r.atomic_text(p,'CORRUPTED')
            self.assertIsNone(r.cached_structure(out,seq))

    def test_duplicate_rows_negative_activity_and_failure_retained(self):
        df=r.load_rows(Path('pamp_ensemble_outputs/pamp_ensemble_full_results.csv'))
        self.assertEqual(len(df),1650)
        self.assertEqual((pd.to_numeric(df.delta_log2_pred)<0).sum(),397)
        with tempfile.TemporaryDirectory() as d:
            res=r.make_results(df,Path(d),{},'/root/tools/USalign')
        self.assertEqual(res.cache_row.tolist(),df.cache_row.tolist())
        self.assertTrue(res.structure_status.eq('pending').all())

    def test_bootstrap_degenerate_and_exact_linear_correlation(self):
        df=pd.DataFrame({'orig_seq':['AA','AA','CC','DD','EE','FF'],
                         'delta_log2_pred':[-2,-1,0,1,2,3],'delta_mean_plddt':[-4,-2,0,2,4,6]})
        c=s.correlations(df,50,2026)
        np.testing.assert_allclose(c.coefficient,1)
        np.testing.assert_allclose(c.wt_cluster_bootstrap95_low,1)
        t=s.stats_table(df,50,2026)
        row=t[t.metric=='delta_mean_plddt'].iloc[0]
        self.assertEqual(row['mean'],1)
        empty=s.stats_table(df.iloc[:0],10,2026)
        self.assertTrue(empty['n'].eq(0).all())

    def test_final_artifacts_keep_failures_and_negative_delta(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            df=pd.DataFrame({'cache_row':[1,2,3], 'orig_seq':['AA','CC','DD'],
                             'mut_seq':['AC','CD','DE'],'delta_log2_pred':[-1,2,-3],
                             'structure_status':['success','success','prediction_error'],
                             'delta_mean_plddt':[-4,5,np.nan]})
            s.finalize(df,out,20,2026)
            import json
            summary=json.loads((out/'summary_metrics.json').read_text())
            self.assertEqual(summary['n_input_rows'],3)
            self.assertEqual(summary['n_failed_rows'],1)
            self.assertEqual(summary['negative_activity_rows_retained'],2)
            self.assertAlmostEqual(summary['failure_rate'],1/3)
            self.assertEqual(len(pd.read_csv(out/'failures.csv')),1)
            self.assertTrue((out/'structural_correlations.csv').exists())

if __name__=='__main__': unittest.main()
