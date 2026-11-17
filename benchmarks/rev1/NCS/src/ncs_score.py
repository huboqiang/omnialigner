"""Frozen pair scoring; unchanged numerical function from the validated run."""
import math, warnings
import ncs_common as c

def score_pair(sb,a,b,genes):
    import numpy as np
    grid=sb.overlap_bygrids;ga,gb=grid(a,b,'spatial','spatial',10,10)
    maska=ga.obs.grid_label.notna()&(ga.obs.grid_label!='NonOverlap');maskb=gb.obs.grid_label.notna()&(gb.obs.grid_label!='NonOverlap')
    # Native functions do not mutate their gridded inputs. Avoid 500 repeated
    # full spatial binning operations; the frozen numerical functions stay intact.
    sb.overlap_bygrids=lambda *args,**kw:(ga,gb)
    funcs=[sb.PCC_bygrid,sb.cos_sim_bygrid,sb.ssim_bygrid,sb.MI_bygrid,sb.ASSD_bygrid]
    records=[]
    try:
        for gene in genes:
            row={'gene':gene}
            for key,fn in zip(c.METRICS[:5],funcs):
                np.random.seed(0)
                with warnings.catch_warnings(record=True)as ws:
                    warnings.simplefilter('always')
                    try:
                        v=float(fn(a,b,gene));row[key]=v if math.isfinite(v)else None;row[key+'_error']='' if math.isfinite(v)else 'native_nonfinite'
                    except Exception as exc:row[key]=None;row[key+'_error']=repr(exc)
                row[key+'_warnings']=' | '.join(str(w.message)for w in ws)
            records.append(row)
    finally:sb.overlap_bygrids=grid
    result={k:c.strict_mean([r[k]for r in records],len(genes))for k in c.METRICS[:5]}
    with warnings.catch_warnings(record=True)as ws:
        warnings.simplefilter('always')
        try:v=float(sb.Overlap_accuracy(a,b,'spatial'));result['Accuracy']=v if math.isfinite(v)else None;accuracy_error='' if math.isfinite(v)else 'native_nonfinite'
        except Exception as exc:result['Accuracy']=None;accuracy_error=repr(exc)
    support=dict(reference_cells=a.n_obs,moving_cells=b.n_obs,reference_covered=int(maska.sum()),moving_covered=int(maskb.sum()),reference_valid_grids=int(ga.obs.loc[maska,'grid_label'].nunique()),moving_valid_grids=int(gb.obs.loc[maskb,'grid_label'].nunique()),native_insufficient_grid_fallback=bool(ga.obs.loc[maska,'grid_label'].nunique()<2 or gb.obs.loc[maskb,'grid_label'].nunique()<2),accuracy_error=accuracy_error,accuracy_warnings=' | '.join(str(w.message)for w in ws))
    return result,records,support
