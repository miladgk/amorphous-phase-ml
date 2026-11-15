import sys, time, numpy as np
from ase import Atoms
from dscribe.descriptors import SOAP

def read_dump(fn):
    with open(fn) as f:
        lines = f.readlines()
    n = int(lines[3])
    bounds = [list(map(float, lines[i].split())) for i in (5, 6, 7)]
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    hdr = lines[8].split()[2:]
    d = np.loadtxt(lines[9:9+n])
    ids = d[:, hdr.index('id')].astype(int)
    pos = d[:, [hdr.index('x'), hdr.index('y'), hdr.index('z')]]
    o = np.argsort(ids)
    return ids[o], pos[o], lo, hi - lo

def soap_for_frame(fn, n_jobs=8):
    ids, pos, lo, L = read_dump(fn)
    pos = np.mod(pos - lo, L)                       # wrap into periodic cell
    atoms = Atoms(symbols=['Cu'] * len(ids), positions=pos, cell=L, pbc=True)
    # species-blind: a single species channel, every neighbour contributes equally
    soap = SOAP(r_cut=5.0, n_max=8, l_max=6, sigma=0.4, species=['Cu'], periodic=True, average='off', sparse=False)
    X = soap.create(atoms, n_jobs=n_jobs)
    return ids, X

if __name__ == '__main__':
    fn, out = sys.argv[1], sys.argv[2]
    t = time.time()
    ids, X = soap_for_frame(fn)
    np.save(out + '.ids.npy', ids); np.save(out + '.npy', X.astype(np.float32))
    print('frame', fn, X.shape, 'time', round(time.time() - t, 1), 's', flush=True)
