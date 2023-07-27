"""Processing script for IAB. Modeled after the original script by Chase Goddard 
(email: cwg45@cornell.edu). Designed to run faster with less memory allocation."""

__author__ = "Andrew DiFabbio"
__email__ = "avd38@cornell.edu"

#imports
import exp_params
import processor
from time import perf_counter
from joblib import Parallel, delayed
import numpy as np
import argparse
import os

def reduce_data(r_dir,w_dir,w_name,t_bin,e_bin,fl_l,fl_h,n_bins,n_ch,n_orb,n_cor,sim=False):
    """Reduce the data in parallel according to the parameters provided and save the result to """
    begin_run_time=perf_counter()
    print(f'Running on {r_dir}')
    
    # parameters used to properly bin data later
    t_factor = exp_params.XMAP_TIME_UNIT/t_bin
    e_factor=1/e_bin
    t_orb = int(exp_params.ORBITAL_PERIOD/exp_params.XMAP_TIME_UNIT)

    # process all files in parallel
    p=processor.Processor(t_factor,e_factor,t_orb,fl_l,fl_h,n_bins,n_ch,n_orb)
    r=[os.path.join(r_dir,s) for s in os.listdir(r_dir)]
    r=[s for s in r if os.path.getsize(s)>10000]
    reduced = Parallel(n_jobs = n_cor)(delayed(p.process_file)(path) for path in r)

    reduced_coin = np.zeros((n_ch*n_ch,n_bins),dtype=int)
    reduced_acoin = np.zeros((n_ch*n_ch,n_bins),dtype=int)
    fluor_count = np.zeros((n_ch),dtype=int)

    for res in reduced:
        reduced_coin+=res[0]
        reduced_acoin+=res[1]
        fluor_count+=res[2]
        
    # write results to the specified path
    np.savetxt(os.path.join(w_dir,w_name+'_coincidence.csv'),reduced_coin,delimiter=',',fmt='%i')
    np.savetxt(os.path.join(w_dir,w_name+'_accidentals.csv'),reduced_acoin,delimiter=',',fmt='%i')
    np.savetxt(os.path.join(w_dir,w_name+'_fluor_count.csv'),fluor_count,delimiter=',',fmt='%i')

    # write run arguments to a text file
    runtime=round(perf_counter()-begin_run_time,3)
    with open(os.path.join(w_dir,w_name+'_log.txt'),'w') as f:
        f.write('CODA Log File \n')
        f.write(f'Log file for {os.path.join(w_dir,w_name)} \n')
        f.write(f'Running on folder {r_dir} \n')
        f.write('\n')
        f.write('Runtime Parameters: \n')
        f.write(f'time window: {t_bin} ns \n')
        f.write(f'energy window: {e_bin} xMAP units \n')
        f.write(f'time offset: {t_orb*20} ns \n')
        f.write(f'Offsets Used: {n_orb} \n')
        f.write(f'fluorescence region: [{fl_l}, {fl_h}] x MAP units \n')
        f.write(f'histogram bins: {n_bins} \n')
        f.write(f'completed in {runtime} s')

    print("COMPLETED. Took {} s".format(runtime))

if __name__== '__main__':
    # Parsing command line arguments
    p = argparse.ArgumentParser()

    # read/write location arguments
    p.add_argument(help='xMAP data directory',dest='r_dir')
    p.add_argument(help='directory to write output .csv to',dest='w_dir') 
    p.add_argument(help='name of output .csv',dest='w_name') 
    # processing arguments
    p.add_argument(help='length of time bin (ns)',type=int,dest='t_bin')
    p.add_argument(help='length of energy bin (xMAP units)',type=int,dest='e_bin')
    p.add_argument(help='beginning of fluorescence region in xMAP units',type=int,dest='fl_l')
    p.add_argument(help='end of fluorescence region in xMAP units',type=int,dest='fl_h')
    p.add_argument(help='number of bins in final histogram',type=int,dest='n_bins') 
    p.add_argument(help='number of detector channels',type=int,dest='n_ch')
    p.add_argument(help='number of orbits over which to average accidentals',type=int,dest='n_orb')
    p.add_argument(help='number of cores to parallelize over',type=int,dest='n_cor')
    p.add_argument(help='--sim indicates whether the data is simulated',action='store_true',dest='--sim')
    a=p.parse_args()

    reduce_data(a.r_dir,a.w_dir,a.w_name,a.t_bin,a.e_bin,a.fl_l,a.fl_h,a.n_bins,a.n_ch,a.n_orb,a.n_cor,a.sim)