__author__ = "Andrew DiFabbio"
__email__ = "avdifabbio@gmail.com"

# IMPORTS
# experimental parameters
import exp_params

# necessary for processing
import numpy as np

# necessary for file management/utility
import argparse
import os


def raw_excess(wd, t, fl_ch, sc_ch, n_orb, pool = 1):
    """Calculate the raw excess rate normalized to fluorescence rate, along with the statistical 
    uncertainty of these, using data formatted by channel."""
    D = os.path.join(os.path.join(wd, 'reformatted_by_channel'), t)
    coin = np.genfromtxt(os.path.join(D, f'coincidence_{fl_ch}_{sc_ch}.csv'), delimiter = ',')
    accs = np.genfromtxt(os.path.join(D, f'accidentals_{fl_ch}_{sc_ch}.csv'), delimiter = ',')/n_orb
    fluo = np.genfromtxt(os.path.join(D, 'fluorescence.csv'), delimiter = ',')[:,fl_ch]
    
    n_run, n_bin = coin.shape
    fluo = fluo.reshape((n_run,1))

    # remove rows of all zeros 
    # (apparently, there are some netcdfs that take up the right
    # amount of space in memory but don't contain any events.)
    coin = coin[~np.all(coin == 0, axis=1)]
    accs = accs[~np.all(accs == 0, axis=1)]
    fluo = fluo[~np.all(fluo == 0, axis=1)]
    n_run, n_bin = coin.shape
    N = n_run//pool

    # pool rows as desired
    coin = np.sum(coin[:pool*N,:].reshape((N, pool, n_bin)), axis = 1)
    accs = np.sum(accs[:pool*N,:].reshape((N, pool, n_bin)), axis = 1)
    fluo = np.sum(fluo[:pool*N,:].reshape((N, pool)), axis = 1).reshape((N,1))
    excess = (coin - accs)/fluo

    E = np.mean(excess, axis = 0)
    dE = np.std(excess, axis = 0, ddof = 1)/np.sqrt(N)

    return E, dE

def atten_src(E, src):
    """Calculate the attenuation due to src of a signal at energy E (which can be a scalar or array)."""
    src_d = np.genfromtxt(os.path.join(exp_params.ATTENUATION_PATH, f'{src}.csv'), delimiter = ',').T
    src_E = src_d[0,:]
    src_T = src_d[1,:]

    # if it is passed an array (which most of the time it should be)
    if isinstance(E, np.ndarray):
        N = E.size
        T = np.zeros(N)

        for i in range(N):
            e = E[i]
            try:
                l_ind = int(np.min(np.where(src_E == np.max(src_E[src_E < e]))))
                h_ind = int(np.min(np.where(src_E == np.min(src_E[src_E > e]))))

                el, eh = src_E[l_ind], src_E[h_ind]
                tl, th = src_T[l_ind], src_T[h_ind]
                T[i]+=(tl + (th-tl)*(e-el)/(eh-el))

            except:
                if e <= np.min(src_E): T[i]+=src_T[np.argmin(src_E)]
                if e >= np.max(src_E): T[i]+=src_T[np.argmax(src_E)]

        return T
    
    # if it is passed a scalar
    else:
        try:
            l_ind = int(np.min(np.where(src_E == np.max(src_E[src_E < E]))))
            h_ind = int(np.min(np.where(src_E == np.min(src_E[src_E > E]))))
        except:
            if E <= np.min(src_E): return src_T[np.argmin(src_E)]
            if E >= np.max(src_E): return src_T[np.argmax(src_E)]

        el, eh = src_E[l_ind], src_E[h_ind]
        tl, th = src_T[l_ind], src_T[h_ind]

        return tl + (th-tl)*(e-el)/(eh-el)

def atten(E, sources):
    """Calculate the compounded attenuation due to a variety of sources."""
    A = 1
    for src in sources: A*=atten_src(E, src)
    return A

def normalized_excess(wd, t, n_ch, n_orb, pool, e_bin, n_bin, e_l, e_h, src = []):
    """Calculate the normalized excess, dividing out attenuation, angular factors,
    and solid angle."""
    En = exp_params.XMAP_ENERGY_UNIT*e_bin*np.arange(n_bin)
    L = int(e_l/(exp_params.XMAP_ENERGY_UNIT*e_bin))
    H = int(e_h/(exp_params.XMAP_ENERGY_UNIT*e_bin))
   
    En = En[L:H]
    new_bin = H-L
    if not t == 'empty':
        ATTEN = atten(En, [*src, t])
    else:
        ATTEN = atten(En, src)

    RES = np.zeros((n_ch*(n_ch - 1), new_bin))
    ERR = np.zeros((n_ch*(n_ch - 1), new_bin))
    
    i = 0
    for sc_ch in range(n_ch):
        ATTEN_CH = e_bin*atten_src(En, exp_params.CHANNEL_DETECTOR[sc_ch])
        for fl_ch in range(n_ch):
            if fl_ch == sc_ch: continue

            Ex, dEx = raw_excess(wd, t, fl_ch, sc_ch, n_orb, pool)
            Ex = Ex[L:H]/ATTEN_CH
            dEx = dEx[L:H]/ATTEN_CH

            RES[i,:]+=Ex/(exp_params.CHANNEL_ANGULAR_COEFF[sc_ch]*exp_params.CHANNEL_SOLID_ANGLE[sc_ch])
            ERR[i,:]+=dEx/(exp_params.CHANNEL_ANGULAR_COEFF[sc_ch]*exp_params.CHANNEL_SOLID_ANGLE[sc_ch])
            i+=1

    RES/=ATTEN
    ERR/=ATTEN

    E = np.sum(RES/ERR**2, axis = 0)/np.sum(1/ERR**2, axis = 0)
    dE = np.sqrt(1/np.sum(1/ERR**2, axis = 0))

    return np.concatenate((En.reshape(1, new_bin), E.reshape(1, new_bin), dE.reshape(1, new_bin)), axis = 0).T

def reduce(wd, n_ch, n_orb, pool, e_bin, n_bin, e_l, e_h, src = []):
    # If it doesn't already exist, create the folders for reduced results.
    wdir = os.path.join(wd,'reduced_excess')
    if not os.path.isdir(wdir): os.mkdir(wdir)

    for t in exp_params.THICKNESS:
        print(f'Performing Reduction for {t}...')
        np.savetxt(os.path.join(wdir, f'reduced_{t}_{e_l}_{e_h}_{pool}.csv'), 
                   normalized_excess(wd, t, n_ch, n_orb, pool, e_bin, n_bin, e_l, e_h, src), delimiter = ',')
    
    with open(os.path.join(wdir,f'reduction_parameters_{e_l}_{e_h}_{pool}.txt'),'w') as f:
        f.write('Runtime Parameters: \n')
        f.write(f'energy window: {e_bin} xMAP units \n')
        f.write(f'offsets Used: {n_orb} \n')
        f.write(f'histogram bins for raw data: {n_bin} \n')
        f.write(f'netcdfs pooled per batch: {pool} \n')
        f.write(f'energy bounds: [{e_l}, {e_h}] keV \n')
        f.write(f'sources of attentuation: {src} \n')
        f.write('Note that self absorption and detector efficiency are implicity included.')

    print(f'SAVED RESULTS TO: {wdir}')


if __name__ == '__main__':
    # Parsing command line arguments
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest = 'command')

    # reduce arguments
    """USAGE: python reduce_data.py reduce <WORKING DIRECTORY> <NUMBER OF CHANNELS> <NUMBER OF ORBITS> \\
    <NETCDF PER BATCH> <ENERGY BIN> <NUMBER OF BINS> <LOWER ENERGY BOUND> <UPPER ENERGY BOUND> <SOURCES OF \\
    ATTENUATION SEPARATED BY SPACES>"""

    reduce_p = sp.add_parser('reduce', help = 'Calculate and save the normalized excess and associated uncertainty')
    reduce_p.add_argument(dest = 'wd', help = 'Working directory')
    reduce_p.add_argument(dest = 'n_ch', type = int, help = 'Number of detector channels')
    reduce_p.add_argument(dest = 'n_orb', type = int, help = 'Number of orbits over which accidentals were taken')
    reduce_p.add_argument(dest = 'pool', type = int, help = 'Number of netcdfs to pool for a batch')
    reduce_p.add_argument(dest = 'e_bin', type = int, help = 'Size of energy bins in xMAP units')
    reduce_p.add_argument(dest = 'n_bin', type = int, help = 'Number of bins in raw histogram')
    reduce_p.add_argument(dest = 'e_l', type = float, help = 'Lower bound for scattered energies in keV')
    reduce_p.add_argument(dest = 'e_h', type = float, help = 'Upper bound for scattered energies in keV')
    reduce_p.add_argument(dest = 'src', nargs = '*', help = 'Sources of attenuation. For example: He Kapton')

    a = p.parse_args()
    if a.command == 'reduce':
        reduce(a.wd, a.n_ch, a.n_orb, a.pool, a.e_bin, a.n_bin, a.e_l, a.e_h, a.src)
