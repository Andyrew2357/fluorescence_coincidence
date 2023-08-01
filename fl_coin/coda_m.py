__author__ = "Andrew DiFabbio"
__email__ = "avd38@cornell.edu"

import reduce
import calc_exc
import processor
import exp_params
import argparse
import numpy as np
import os
import multiprocessing
import time

def reduce_main(r_dir,w_dir,t_bin,e_bin,fl_l,fl_h,n_bins,n_ch,n_orb,master_csv):
    begin_run_time=time.perf_counter()
    if not os.path.isdir(os.path.join(a.w_dir,'data/')): os.mkdir(os.path.join(a.w_dir,'data/'))
    for t in ['data/40nm/','data/80nm/','data/160nm/','data/320nm/','data/empty/']:
        p=os.path.join(a.w_dir,t)
        if not os.path.isdir(p): os.mkdir(p)

    n_cor=multiprocessing.cpu_count()
    with open(master_csv,'r+') as master:
        for l in master:
            run,thickness=l.strip().split(',')
            w_name=f'data/{thickness}/{run}'
            data_dir=os.path.join(r_dir,'%s/list/' % run)
            reduce.reduce_data(data_dir,w_dir,w_name,t_bin,e_bin,fl_l,fl_h,n_bins,n_ch,n_orb,n_cor)

    calc_exc.combine_data(w_dir,n_ch,n_bins,n_orb,e_bin,False) #currently hardcoding to skip attentuation correction

    runtime=round((time.perf_counter()-begin_run_time)/3600,3)
    with open(os.path.join(w_dir,'reduction_parameters.txt'),'w') as f:
        f.write('Runtime Parameters: \n')
        f.write(f'time window: {t_bin} ns \n')
        f.write(f'energy window: {e_bin} xMAP units \n')
        f.write(f'Offsets Used: {n_orb} \n')
        f.write(f'fluorescence region: [{fl_l}, {fl_h}] x MAP units \n')
        f.write(f'histogram bins: {n_bins} \n')
        f.write(f'completed in {runtime} hrs')

    print("COMPLETED. Took {} hrs".format(runtime))

def monitor_main(r_dir,w_dir,w_name,t_bin,e_bin,fl_l,fl_h,n_bins,n_ch,n_orb,sleep_t):
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
    
    coin=np.zeros((n_ch*n_ch,n_bins),dtype=int)
    acoin=np.zeros((n_ch*n_ch,n_bins),dtype=int)
    fl_ct=np.zeros((n_ch,1),dtype=int)

    t_factor = exp_params.XMAP_TIME_UNIT/t_bin
    e_factor=1/e_bin
    t_orb = int(exp_params.ORBITAL_PERIOD/exp_params.XMAP_TIME_UNIT)
    p=processor.Processor(t_factor,e_factor,t_orb,fl_l,fl_h,n_bins,n_ch,n_orb)

    print(f'Initiating Folder Monitor on {r_dir}')
    processed={}
    try:
        while True:
            print('Looking for new files...')
            curr=[l for l in os.listdir(r_dir) if not l in processed]
            if len(curr)==0:
                print('None Found.')
                time.sleep(sleep_t)
                continue
            processed.add(curr)
            
            print(f'Processing {curr} ...')
            fcoin,facoin,ffl_ct=p.process_file(os.path.join(r_dir,curr))
            coin+=fcoin
            acoin+=facoin
            fl_ct+=ffl_ct

            print(f'Finished Processing. Writing to {w_dir}')
            np.savetxt(os.path.join(w_dir,w_name+'_coincidence.csv'),coin,delimiter=',',fmt='%i')
            np.savetxt(os.path.join(w_dir,w_name+'_accidentals.csv'),acoin,delimiter=',',fmt='%i')
            np.savetxt(os.path.join(w_dir,w_name+'_fluor_count.csv'),fl_ct,delimiter=',',fmt='%i')

    except KeyboardInterrupt:
        print(f'Folder Monitor on {r_dir} Terminated.')

if __name__== '__main__':
    # Parsing command line arguments
    p = argparse.ArgumentParser()
    sp=p.add_subparsers(dest='command')

    # reduce arguments
    reduce_p=sp.add_parser('reduce',help='reduce a dataset that has already been acquired')
    reduce_p.add_argument(help='xMAP data directory',dest='r_dir')
    reduce_p.add_argument(help='directory to write output .csv to',dest='w_dir') 
    reduce_p.add_argument(help='length of time bin (ns)',type=int,dest='t_bin')
    reduce_p.add_argument(help='length of energy bin (xMAP units)',type=int,dest='e_bin')
    reduce_p.add_argument(help='beginning of fluorescence region in xMAP units',type=int,dest='fl_l')
    reduce_p.add_argument(help='end of fluorescence region in xMAP units',type=int,dest='fl_h')
    reduce_p.add_argument(help='number of bins in final histogram',type=int,dest='n_bins') 
    reduce_p.add_argument(help='number of detector channels',type=int,dest='n_ch')
    reduce_p.add_argument(help='number of orbits over which to average accidentals',type=int,dest='n_orb')
    reduce_p.add_argument(help='path to a csv detailing runs to be processed',dest='master_csv')

    # monitor arguments
    monitor_p=sp.add_parser('monitor',help='monitor a folder for realtime data collection and reduction')
    monitor_p.add_argument(help='xMAP data directory',dest='r_dir')
    monitor_p.add_argument(help='directory to write output .csv to',dest='w_dir')
    monitor_p.add_argument(help='name of the run',dest='w_name') 
    monitor_p.add_argument(help='length of time bin (ns)',type=int,dest='t_bin')
    monitor_p.add_argument(help='length of energy bin (xMAP units)',type=int,dest='e_bin')
    monitor_p.add_argument(help='beginning of fluorescence region in xMAP units',type=int,dest='fl_l')
    monitor_p.add_argument(help='end of fluorescence region in xMAP units',type=int,dest='fl_h')
    monitor_p.add_argument(help='number of bins in final histogram',type=int,dest='n_bins') 
    monitor_p.add_argument(help='number of detector channels',type=int,dest='n_ch')
    monitor_p.add_argument(help='number of orbits over which to average accidentals',type=int,dest='n_orb')
    monitor_p.add_argument(help='time to sleep between checking for new data (s)',dest='sleep_t')
    
    a=p.parse_args()
    if a.command=='reduce':
        reduce_main(a.r_dir,a.w_dir,a.t_bin,a.e_bin,a.fl_l,a.fl_h,a.n_bins,a.n_ch,a.n_orb,a.master_csv)
    elif a.command=='monitor':
        monitor_main(a.r_dir,a.w_dir,a.w_name,a.t_bin,a.e_bin,a.fl_l,a.fl_h,a.n_bins,a.n_ch,a.n_orb,a.sleep_t)
