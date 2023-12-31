__author__ = "Andrew DiFabbio"
__email__ = "avd38@cornell.edu"

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
    p=processor.Processor(t_factor,e_factor,t_orb,fl_l,fl_h,n_bins,n_ch,n_orb,sim)
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

def tally_accidentals(off_l,off_h,off_s,r_dir,w_dir,w_name,t_bin,fl_l,fl_h,n_ch,n_cor,sim=False):
    """Reduce the data in parallel according to the parameters provided and save the result to """
    begin_run_time=perf_counter()
    print(f'Running on {r_dir}')
    
    # parameters used to properly bin data later
    t_factor = exp_params.XMAP_TIME_UNIT/t_bin

    # process all files in parallel
    p=processor.Processor(t_factor,1,1,fl_l,fl_h,1,n_ch,1,sim)
    r=[os.path.join(r_dir,s) for s in os.listdir(r_dir)]
    r=[s for s in r if os.path.getsize(s)>10000]

    # -----------THE PARALLELIZATION FOR THIS CURRENTLY DOESN'T WORK AND NEEDS TO BE FIXED-----------
    # r=[(pth,off_l/exp_params.XMAP_TIME_UNIT,off_h/exp_params.XMAP_TIME_UNIT,off_s/exp_params.XMAP_TIME_UNIT) for pth in r]
    # reduced = Parallel(n_jobs = n_cor)(delayed(p.accidentals_given_offset)(pth,a,b,c) for pth,a,b,c in r)
    reduced=[p.accidentals_given_offset(r[0],off_l/exp_params.XMAP_TIME_UNIT,off_h/exp_params.XMAP_TIME_UNIT,off_s/exp_params.XMAP_TIME_UNIT)]

    tally=np.zeros(int((off_h-off_l)/off_s))
    for t in reduced: tally+=t

    if not os.path.isdir(os.path.join(w_dir,'acc_tally/')): os.mkdir(os.path.join(w_dir,'acc_tally/'))
    np.savetxt(os.path.join(os.path.join(w_dir,'acc_tally/'),w_name+'_acc_tally.csv'),tally,delimiter=',',fmt='%i')

    # write run arguments to a text file
    runtime=round(perf_counter()-begin_run_time,3)
    with open(os.path.join(os.path.join(w_dir,'acc_tally/'),w_name+'_acc_tally_log.txt'),'w') as f:
        f.write('CODA Log File \n')
        f.write(f'Accidental Tally Log file for {w_name} \n')
        f.write(f'Running on folder {r_dir} \n')
        f.write('\n')
        f.write('Runtime Parameters: \n')
        f.write(f'off_l,off_h,off_s (ns): {off_l},{off_h},{off_s} \n')
        f.write(f'time window: {t_bin} ns \n')
        f.write(f'fluorescence region: [{fl_l}, {fl_h}] x MAP units \n')
        f.write(f'completed in {runtime} s')

    plt.scatter(np.arange(off_l,off_h,off_s),tally,color='k')
    plt.title(f'{w_name} Accidental Counts Varying Time Offset')
    plt.xlabel('Time Offset (ns)')
    plt.ylabel('Accidental Count')
    plt.savefig(os.path.join(os.path.join(w_dir,'acc_tally/'),f'{w_name}_acc_tally_fig.png'))

    print("COMPLETED. Took {} s".format(runtime))

if __name__== '__main__':
    # Parsing command line arguments
    p = argparse.ArgumentParser()
    sp=p.add_subparsers(dest='command')

    # reduce command
    """USAGE: python <PATH TO fl_coin>/fl_coin/reduce.py reduce <PATH TO RAW DATA> <PATH TO WRITE DIRECTORY> \\
    <RUN NAME> <TIME BIN> <ENERGY BIN> <FLUO ENERGY LOWER BOUND> <FLUO ENERGY UPPER BOUND> <NUMBER OF BINS> \\
    <NUMBER OF CHANNELS> <NUMBER OF ORBITS> <NUMBER OF CORES> <--sim>"""

    reduce_p=sp.add_parser('reduce',help='reduce the data from a single folder')
    reduce_p.add_argument(help='xMAP data directory',dest='r_dir')
    reduce_p.add_argument(help='directory to write output .csv to',dest='w_dir') 
    reduce_p.add_argument(help='name of output .csv',dest='w_name') 
    reduce_p.add_argument(help='length of time bin (ns)',type=int,dest='t_bin')
    reduce_p.add_argument(help='length of energy bin (xMAP units)',type=int,dest='e_bin')
    reduce_p.add_argument(help='beginning of fluorescence region in xMAP units',type=int,dest='fl_l')
    reduce_p.add_argument(help='end of fluorescence region in xMAP units',type=int,dest='fl_h')
    reduce_p.add_argument(help='number of bins in final histogram',type=int,dest='n_bins') 
    reduce_p.add_argument(help='number of detector channels',type=int,dest='n_ch')
    reduce_p.add_argument(help='number of orbits over which to average accidentals',type=int,dest='n_orb')
    reduce_p.add_argument(help='number of cores to parallelize over',type=int,dest='n_cor')
    reduce_p.add_argument('--sim',help='--sim indicates whether the data is simulated',action='store_true')
    
    # tallyacc command
    """USAGE: python <PATH TO fl_coin>/fl_coin/reduce.py tallyacc <TIME OFFSET LOWER BOUND> <TIME OFFSET UPPER BOUND> \\
    <STEP SIZE BETWEEN TIME OFFSETS> <PATH TO RAW DATA> <PATH TO WRITE DIRECTORY> <RUN NAME> <TIME BIN> \\
    <FLUO ENERGY LOWER BOUND> <FLUO ENERGY UPPER BOUND> <NUMBER OF CHANNELS> <NUMBER OF CORES> <--sim>"""

    tally_acc_p=sp.add_parser('tallyacc',help='tally the accidentals for runs in a folder given offsets')
    tally_acc_p.add_argument(help='lowest time offset to use in nanoseconds',type=int,dest='off_l')
    tally_acc_p.add_argument(help='highest time offset to use in nanoseconds',type=int,dest='off_h')
    tally_acc_p.add_argument(help='step between off_l and off_h in nanoseconds',type=int,dest='off_s')
    tally_acc_p.add_argument(help='xMAP data directory',dest='r_dir')
    tally_acc_p.add_argument(help='directory to write output .csv to',dest='w_dir') 
    tally_acc_p.add_argument(help='name of output .csv',dest='w_name') 
    tally_acc_p.add_argument(help='length of time bin (ns)',type=int,dest='t_bin')
    tally_acc_p.add_argument(help='beginning of fluorescence region in xMAP units',type=int,dest='fl_l')
    tally_acc_p.add_argument(help='end of fluorescence region in xMAP units',type=int,dest='fl_h')
    tally_acc_p.add_argument(help='number of detector channels',type=int,dest='n_ch')
    tally_acc_p.add_argument(help='number of cores to parallelize over',type=int,dest='n_cor')
    tally_acc_p.add_argument('--sim',help='--sim indicates whether the data is simulated',action='store_true')
    
    a=p.parse_args()
    if a.command=='reduce':
        reduce_data(a.r_dir,a.w_dir,a.w_name,a.t_bin,a.e_bin,a.fl_l,a.fl_h,a.n_bins,a.n_ch,a.n_orb,a.n_cor,a.sim)
    elif a.command=='tallyacc':
        from matplotlib import pyplot as plt
        tally_accidentals(a.off_l,a.off_h,a.off_s,a.r_dir,a.w_dir,a.w_name,a.t_bin,a.fl_l,a.fl_h,a.n_ch,a.n_cor,a.sim)
