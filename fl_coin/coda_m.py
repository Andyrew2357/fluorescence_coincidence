__author__ = "Andrew DiFabbio"
__email__ = "avd38@cornell.edu"

import reduce
import argparse
import os
import multiprocessing
from time import perf_counter

def main(r_dir,w_dir,t_bin,e_bin,fl_l,fl_h,n_bins,n_ch,n_orb,master_csv):
    begin_run_time=perf_counter()
    for t in ['40nm/','80nm/','160nm/','320nm/','empty/']:
        p=os.path.join(w_dir,t)
        if os.path.isdir(p): continue
        os.mkdir(p)

    n_cor=multiprocessing.cpu_count()
    with open(master_csv,'r+') as master:
        for l in master:
            run,thickness=l.strip().split(',')
            w_name=f'{thickness}/{run}'
            data_dir=os.path.join(r_dir,'%s/list/' % run)
            reduce.reduce_data(data_dir,w_dir,w_name,t_bin,e_bin,fl_l,fl_h,n_bins,n_ch,n_orb,n_cor)

    runtime=round((perf_counter()-begin_run_time)/3600,3)
    with open(os.path.join(w_dir,'reduction_parameters.txt'),'w') as f:
        f.write('Runtime Parameters: \n')
        f.write(f'time window: {t_bin} ns \n')
        f.write(f'energy window: {e_bin} xMAP units \n')
        f.write(f'Offsets Used: {n_orb} \n')
        f.write(f'fluorescence region: [{fl_l}, {fl_h}] x MAP units \n')
        f.write(f'histogram bins: {n_bins} \n')
        f.write(f'completed in {runtime} hrs')

    print("COMPLETED. Took {} hrs".format(runtime))

if __name__== '__main__':
    # Parsing command line arguments
    p = argparse.ArgumentParser()

    # read/write location arguments
    p.add_argument(help='xMAP data directory',dest='r_dir')
    p.add_argument(help='directory to write output .csv to',dest='w_dir') 
    # processing arguments
    p.add_argument(help='length of time bin (ns)',type=int,dest='t_bin')
    p.add_argument(help='length of energy bin (xMAP units)',type=int,dest='e_bin')
    p.add_argument(help='beginning of fluorescence region in xMAP units',type=int,dest='fl_l')
    p.add_argument(help='end of fluorescence region in xMAP units',type=int,dest='fl_h')
    p.add_argument(help='number of bins in final histogram',type=int,dest='n_bins') 
    p.add_argument(help='number of detector channels',type=int,dest='n_ch')
    p.add_argument(help='number of orbits over which to average accidentals',type=int,dest='n_orb')
    p.add_argument(help='path to a csv detailing runs to be processed',dest='master_csv')
    a=p.parse_args()

    main(a.r_dir,a.w_dir,a.t_bin,a.e_bin,a.fl_l,a.fl_h,a.n_bins,a.n_ch,a.n_orb,a.master_csv)