import glob, numpy as np, pandas as pd, sys
sys.path.insert(0,'/home/m/claude_work/soap')
from soap_frame import soap_for_frame
sets={'5050':'/mnt/e/Projects/Amorphous/GitHub/Atomic_Simulation_Post-Processing_Pipeline/files/snapshots_5050/*.lammpstrj',
      '4654':'/mnt/e/Projects/Amorphous/GitHub/Atomic_Simulation_Post-Processing_Pipeline/files/snapshots_4654/*.lammpstrj',
      '6436':'/mnt/e/Projects/Amorphous/GitHub/Atomic_Simulation_Post-Processing_Pipeline/files/snapshots_6436/*.lammpstrj',
      'target':'/mnt/e/Projects/Amorphous/GitHub/Machine_Learning_Pipeline_for_Materials_Science/data/polyamorphous/*.lammpstrj'}
for name,pat in sets.items():
    files=sorted(glob.glob(pat))
    acc=None; cnt=0; ids0=None
    for fn in files:
        ids,X=soap_for_frame(fn, n_jobs=8)
        if ids0 is None: ids0=ids; acc=np.zeros_like(X,dtype=np.float64)
        assert np.array_equal(ids,ids0)
        acc+=X; cnt+=1
    avg=acc/cnt
    cols=[f'soap_{k:03d}' for k in range(avg.shape[1])]
    df=pd.DataFrame(avg,columns=cols); df.insert(0,'id',ids0)
    df.to_csv(f'/home/m/claude_work/soap/soap_avg_{name}.csv',index=False)
    print(name,len(files),'frames',df.shape,flush=True)
