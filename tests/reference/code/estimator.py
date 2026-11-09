"""Call the unchanged directed Ras DHAM and corrected MFPT optimizer."""
import numpy as np
import config
from dham import DHAM,count_transitions_one
from util import try_and_optim_M

def counts(cv):
    cv=np.asarray(cv)
    if not np.isfinite(cv).all() or np.any(cv<config.qspace[0]) or np.any(cv>=config.qspace[-1]):
        raise RuntimeError('CV outside the fixed model grid; preserve run for review')
    # Exactly the Ras convention: no -1; no pair joins different bias windows.
    return count_transitions_one(np.digitize(cv,config.qspace),config.DHAM_num_bins,config.dham_lag_steps)

def decide(history,params,last,seed,window):
    C=np.stack(history)
    triples=np.asarray(params).reshape(-1,3,config.num_gaussian).transpose(0,2,1)
    d=DHAM(triples,config.DHAM_num_bins)
    d.qspace=config.qspace
    total=C.sum(0)
    visits=C.sum(2)
    # build_MM is used directly: avoid an unnecessary global stationary-vector
    # calculation for a potentially reducible directed graph (not the objective).
    M=np.asarray(d.build_MM(total,visits,biased=True).T,dtype=float)
    keep=np.flatnonzero(M.any(0)&M.any(1))
    if not len(keep):raise RuntimeError('No observed working chain')
    working=M[np.ix_(keep,keep)]
    if not np.allclose(working.sum(0),1,rtol=0,atol=1e-10):
        raise RuntimeError('Trimming discarded probability; no renormalized invented graph')
    start=int(np.digitize(last,config.qspace))
    final=int(np.digitize(config.target_distance,config.qspace)-1)
    target=int(keep[np.argmin(abs(keep-final))])
    ii,jj=np.indices(total.shape);off=total[ii!=jj].sum()
    fraction=float(total[abs(ii-jj)>1].sum()/off) if off else 0.
    if fraction!=0:raise RuntimeError('Non-neighbor transitions: Ras neighbor-only tilt would discard evidence')
    new,info=try_and_optim_M(working,keep,num_gaussian=config.num_gaussian,start_index=start,end_index=target,
                           qspace=config.qspace,rng=np.random.default_rng([seed,window+1]))
    result=dict(qspace=config.qspace,counts=total,M_working=working,working_indices=keep,
                last_visited_state=start,closest_index=target,params_this_prop=params[-1],next_params=new,
                fraction_nonadjacent=fraction,**info)
    return new,result
