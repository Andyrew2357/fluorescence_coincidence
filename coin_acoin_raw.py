__author__ = "Andrew DiFabbio"
__email__ = "avdifabbio@gmail.com"

# IMPORTS
# experimental parameters
import exp_params

# necessary for processing
from scipy.io import netcdf_file
from joblib import Parallel, delayed
import multiprocessing
import numpy as np

# necessary for file management/utility
from time import perf_counter
import argparse
import time
import shutil
import os


def readnetcdf(path,simulated=False):
    """Read in a netcdf (.nc) file from disk and return the data contained
    in the form of a numpy structured array."""
    if simulated: return np.load(path)
    
    f = netcdf_file(path, 'r', mmap=False)
    tmp_data = f.variables['array_data']
    data = tmp_data[:].copy().astype('uint16') #copy data into memory from disk
    f.close() #close netcdf file
    
    data = data.ravel() #flatten data to 1 dimension 
    
    #Get number of events. 2^16 scales word 67 according to xMAP file format
    #(i.e. word 67 contains more significant digits of the num_events field)
    #word 66 and 67 are the header words that contain the number of events
    num_events =  data[66].astype('int32')+(2**16)*(data[67].astype('int32'))
    offset = 256
    
    #set up vectors to store data
    E = np.zeros(num_events)
    channel = np.zeros(num_events)
    time = np.zeros(num_events, dtype='float64')
            
    for i in range(0,num_events):
        #xMAP stores data in specific bits of these words
        word1 = data[offset+3*i]
        word2 = data[offset+3*i+1]
        word3 = data[offset+3*i+2]
        
        #extract channel bits (13-15 of word 1)
        channel[i] = np.bitwise_and(np.right_shift(word1, 13), 3)
        #extract energy bits (0-12 of word 1)
        E[i] = np.bitwise_and(word1, 0x1fff)
        #extract time bits (word3 contains MSB)
        time[i] = (word3 * (2**16) + word2) #* time_constant
                
    #package data into table format (as numpy structured array)
    return np.array(list(zip(time, E, channel)),
            dtype = {'names':['time', 'E', 'channel'],
            'formats':['float64', 'int32', 'int32']})

def process_file(path, t_factor, e_factor, t_orb, fl_l, fl_h, n_bins, n_ch, n_orb, sim=False):
    """Processes the relevant file for coincidences and accidentals and add this info to the reduced data array. 
    Results take the format <coincidences>,<accidentals>,<fluorescence counts>."""
    print(path)

    reduced_coin = np.zeros((n_ch*n_ch, n_bins), dtype=int)
    reduced_acoin = np.zeros((n_ch*n_ch, n_bins), dtype=int)
    fluor_count = np.zeros(n_ch, dtype=int)
    evts = readnetcdf(path, sim)

    fl=set()
    for t, E, ch in evts:
        if fl_l < E < fl_h: 
            fl.add((ch, int(t*t_factor)))
            fluor_count[ch]+=1

    for t, E, ch in evts:
        for fl_ch in range(n_ch):
            if fl_ch == ch: continue
            if (fl_ch, int(t*t_factor)) in fl: reduced_coin[n_ch*fl_ch+ch, int(E*e_factor)]+=1
            for i in range(1, n_orb+1):
                t_off=int((t + i*t_orb)*t_factor)
                if (fl_ch, t_off) in fl: reduced_acoin[n_ch*fl_ch + ch, int(E*e_factor)]+=1

    return reduced_coin,reduced_acoin,fluor_count


# FIX THISL:KJA:LKDJ:ALKDJ:ALKDJ
def run_raw_counts(r_dir,w_dir,w_name,t_bin,e_bin,fl_l,fl_h,n_bins,n_ch,n_orb,n_cor,sim=False):
    """Reduce the data from a given folder in parallel according to the parameters provided and save the result 
    to a series of CSV files for each netcdf"""
    begin_run_time = perf_counter()
    print(f'Running on {r_dir}')
    
    # parameters used to properly bin data later
    t_factor = exp_params.XMAP_TIME_UNIT/t_bin
    e_factor = 1/e_bin
    t_orb = int(exp_params.ORBITAL_PERIOD/exp_params.XMAP_TIME_UNIT)

    # process all files in parallel
    to_process = [os.path.join(r_dir, s) for s in os.listdir(r_dir)]
    to_process = [s for s in to_process if os.path.getsize(s)>10000]
    A = (t_factor, e_factor, t_orb, fl_l, fl_h, n_bins, n_ch, n_orb, sim)
    r = [(s, *A) for s in to_process]
    reduced = Parallel(n_jobs = n_cor)(delayed(process_file)(*args) for args in r)

    for res, fnam in zip(reduced, to_process):
        reduced_coin, reduced_acoin,fluor_count=res
        identifier = os.path.relpath(fnam, r_dir)
            
        np.savetxt(os.path.join(w_dir,f'{w_name}_{identifier}_coincidence.csv'), reduced_coin, delimiter = ',', fmt = '%i')
        np.savetxt(os.path.join(w_dir,f'{w_name}_{identifier}_accidentals.csv'), reduced_acoin, delimiter = ',', fmt = '%i')
        np.savetxt(os.path.join(w_dir,f'{w_name}_{identifier}_fluor_count.csv'), fluor_count, delimiter = ',', fmt = '%i')

    # write run arguments to a text file
    runtime = round(perf_counter() - begin_run_time, 3)
    with open(os.path.join(w_dir, w_name + '_log.txt'), 'w') as f:
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

def raw_main(r_dir,w_dir,t_bin,e_bin,fl_l,fl_h,n_bins,n_ch,n_orb,master_csv):
    """Call run_raw_counts on a series of folders specified by master_csv"""
    begin_run_time = time.perf_counter()

    # If they do not already exist, create the folders for reduced raw counts.
    raw_dir = os.path.join(w_dir,'raw_counts')
    if not os.path.isdir(raw_dir): os.mkdir(raw_dir)
    for t in exp_params.THICKNESS:
        p = os.path.join(raw_dir,t)
        if not os.path.isdir(p): os.mkdir(p)

    # Repeatedly call reduce_data to process each run specified by the master CSV.
    n_cor=multiprocessing.cpu_count()
    with open(master_csv,'r+') as master:
        for l in master:
            run,thickness=l.strip().split(',')
            w_name=f'{thickness}\\{run}'
            data_dir=os.path.join(r_dir,'%s\\list' % run)
            run_raw_counts(data_dir,raw_dir,w_name,t_bin,e_bin,fl_l,fl_h,n_bins,n_ch,n_orb,n_cor)

    # Write the reduction parameters and total run-time to a TXT.
    runtime=round((time.perf_counter()-begin_run_time)/3600,3)
    with open(os.path.join(w_dir,'reduction_parameters.txt'),'w') as f:
        f.write('Runtime Parameters: \n')
        f.write(f'time window: {t_bin} ns \n')
        f.write(f'energy window: {e_bin} xMAP units \n')
        f.write(f'offsets Used: {n_orb} \n')
        f.write(f'fluorescence region: [{fl_l}, {fl_h}] xMAP units \n')
        f.write(f'histogram bins: {n_bins} \n')
        f.write(f'completed in {runtime} hrs')

    shutil.copyfile(master_csv, os.path.join(raw_dir, 'master_csv_copy.csv'))

    print("COMPLETED. Took {} hrs".format(runtime))

def reformat_by_channel(wd, n_ch, n_bins, blacklist=[]):
    """Using raw_counts in the working directory, create histograms of the 
    number of coincidence and accidental counts for each pair of detectors"""
    print('BLACKLIST:')
    print(blacklist)

    raw_dir = os.path.join(wd, 'raw_counts')
    ref_dir = os.path.join(wd, 'reformatted_by_channel')
    if not os.path.isdir(ref_dir): os.mkdir(ref_dir)

    # Process everything in order of thickness
    for t in exp_params.THICKNESS:
        reformatted = {}
        D = os.path.join(raw_dir, t)
        P = os.path.join(ref_dir,t)
        if not os.path.isdir(P): os.mkdir(P)

        for fname in os.listdir(D):
            # If the run is blacklisted continue
            blacklisted = False
            for b in blacklist:
                if b in fname: blacklisted = True
            if blacklisted: continue

            if 'coincidence' in fname or 'accidentals' in fname:
                counts = np.genfromtxt(os.path.join(D, fname), delimiter = ',')

                for fl_ch in range(n_ch):
                    for sc_ch in range(n_ch):
                        if fl_ch == sc_ch: continue
                        tag = f'coincidence_{fl_ch}_{sc_ch}' if ('coincidence' in fname) else f'accidentals_{fl_ch}_{sc_ch}'

                        try:
                            reformatted[tag] = np.concatenate((reformatted[tag], counts[n_ch*fl_ch + sc_ch, :].reshape(1, n_bins)), axis = 0)
                        except:
                            reformatted[tag] = counts[n_ch*fl_ch + sc_ch, :].reshape(1, n_bins)
                            
                continue

            if 'fluor_count' in fname:
                fluor = np.genfromtxt(os.path.join(D, fname), delimiter = ',').reshape(1, n_ch)
                try:
                    reformatted['fluorescence'] = np.concatenate((reformatted['fluorescence'], fluor), axis = 0)
                except:
                    reformatted['fluorescence'] = fluor

                continue
        
        for key in reformatted:
            np.savetxt(os.path.join(P, f'{key}.csv'), reformatted[key], delimiter = ',', fmt = '%i')


if __name__ == '__main__':
    # Parsing command line arguments
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest = 'command')

    # raw arguments
    """USAGE: python coin_acoin_raw.py raw <PATH TO RAW DATA> <PATH TO WRITE DIRECTORY> <TIME BIN> \\
    <ENERGY BIN> <FLUO ENERGY LOWER BOUND> <FLUO ENERGY UPPER BOUND> <NUMBER OF BINS> <NUMBER OF CHANNELS> \\
    <NUMBER OF ORBITS> <PATH TO MASTER CSV>"""

    raw_p = sp.add_parser('raw', help = 'Reduce a dataset of events to a raw coincidence and accidental counts')
    raw_p.add_argument(dest = 'r_dir', help = 'xMAP data directory')
    raw_p.add_argument(dest = 'w_dir', help = 'Directory to write raw coincidence and accidental counts')
    raw_p.add_argument(dest = 't_bin', type = int, help = 'Size of time bins (ns)')
    raw_p.add_argument(dest = 'e_bin', type = int, help = 'Size of energy bins (xMAP units)')
    raw_p.add_argument(dest = 'fl_l', type = int, help = 'Lower bound of the fluorescence energies in xMAP units')
    raw_p.add_argument(dest = 'fl_h', type = int, help = 'Upper bound of the fluorescence energies in xMAP units')
    raw_p.add_argument(dest = 'n_bins', type = int, help = 'Number of energy bins in the final histogram')
    raw_p.add_argument(dest = 'n_ch', type = int, help = 'Number of detector channels')
    raw_p.add_argument(dest = 'n_orb', type = int, help = 'Number of orbits over which to average accidentals')
    raw_p.add_argument(dest = 'master_csv', help = 'CSV detailing runs to be processed')

    # reformat arguments
    """USAGE: python coin_acoin_raw.py reformat <WORKING DIRECTORY> <NUMBER OF CHANNELS> <BLACKLISTED RUNS SEPARATED \\
    BY SPACES>"""

    reformat_p = sp.add_parser('reformat', help = """Using raw_counts in the working directory, create histograms of the 
                               number of coincidence and accidental counts for each pair of detectors""")
    reformat_p.add_argument(dest = 'wd', help = 'Working directory')
    reformat_p.add_argument(dest = 'n_ch', type = int, help = 'Number of detector channels')
    reformat_p.add_argument(dest = 'n_bins', type = int, help = 'Number of energy bins in the final histogram')
    reformat_p.add_argument(dest = 'blacklist', nargs = '*', help = 'Blacklisted runs. The routine will ignore runs with these names.')

    a = p.parse_args()
    if a.command == 'raw':
        raw_main(a.r_dir, a.w_dir, a.t_bin, a.e_bin, a.fl_l, a.fl_h, a.n_bins, a.n_ch, a.n_orb, a.master_csv)
    elif a.command == 'reformat':
        reformat_by_channel(a.wd, a.n_ch, a.n_bins, a.blacklist)