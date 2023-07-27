__author__ = "Andrew DiFabbio"
__email__ = "avd38@cornell.edu"

import argparse
import numpy as np
import os
from matplotlib import pyplot as plt
import exp_params

np.seterr(divide='ignore', invalid='ignore')

def sum_over_results(path,num_ch,num_bins,verbose=False):
    fl_ct=np.zeros((num_ch,1))
    coin=np.zeros((num_ch*num_ch,num_bins))
    acoin=np.zeros((num_ch*num_ch,num_bins))
    
    for l in os.listdir(path):
        if '.txt' in l: continue

        if 'fluor_count' in l:
            fl_ct+=np.genfromtxt(os.path.join(path,l),delimiter=',').reshape((4,1))
        elif 'coincidence' in l:
            coin+=np.genfromtxt(os.path.join(path,l),delimiter=',')
        elif 'accidentals' in l:
            acoin+=np.genfromtxt(os.path.join(path,l),delimiter=',')
    
    if not verbose:
        coin=np.sum(coin.reshape((num_ch,num_ch,num_bins)),axis=1)
        acoin=np.sum(acoin.reshape((num_ch,num_ch,num_bins)),axis=1)

    return fl_ct,coin,acoin

def get_ang_corr(num_ch):
    ang_corr=np.zeros((num_ch,1))
    for r in range(num_ch):
        for ch in range(num_ch):
            if r==ch: continue
            ang_corr[r]+=exp_params.CHANNEL_ANGULAR_COEFF[ch]*exp_params.CHANNEL_SOLID_ANGLE[ch]

    return np.power(ang_corr,-1)

def combine_summed_results(data,num_ch,num_orb,e_bin):
    ang_corr=get_ang_corr(num_ch)
    fl_ct,coin,acoin=data

    excesses=ang_corr*(coin-acoin/num_orb)/fl_ct
    variances=ang_corr**2*((coin+acoin/num_orb**2)/fl_ct**2 + (coin-acoin/num_orb)**2/fl_ct**3)
    
    norm_vars=np.power(np.sum(np.power(variances,-1),axis=0),-1)
    
    return (norm_vars*np.sum(excesses/variances,axis=0))/(e_bin*exp_params.XMAP_ENERGY_UNIT), \
        np.sqrt(norm_vars)/(e_bin*exp_params.XMAP_ENERGY_UNIT)

def combine_folder(path,num_ch,num_bins,num_orb,e_bin):
    return combine_summed_results(sum_over_results(path,num_ch,num_bins),num_ch,num_orb,e_bin)

def correct_atten(data,e_bin):
    # NEEDS TO BE IMPLEMENTED
    return data

def combine_data(wdir,num_ch,num_bins,num_orb,e_bin,atten=True,write=True):
    result_excesses=np.zeros((5,num_bins))
    result_errors=np.zeros((5,num_bins))
    for r,t in enumerate(['40nm/','80nm/','160nm/','320nm/','empty/']):
        exc,err=combine_folder(os.path.join(wdir,t),num_ch,num_bins,num_orb,e_bin)
        result_excesses[r,:]+=exc
        result_errors[r:]+=err
    if atten: result=correct_atten(result,e_bin)

    if write: 
        np.savetxt(os.path.join(wdir,'combined_result_excesses.csv'),result_excesses,delimiter=',')
        np.savetxt(os.path.join(wdir,'combined_result_errors.csv'),result_errors,delimiter=',')
    else:
        return result_excesses,result_errors
    
if __name__=='__main__':
    # Parsing command line arguments
    p = argparse.ArgumentParser()

    # read/write location arguments
    p.add_argument(help='working directory',dest='wdir') 
    # processing arguments
    p.add_argument(help='length of energy bin (xMAP units)',type=int,dest='e_bin')
    p.add_argument(help='number of bins in final histogram',type=int,dest='n_bins') 
    p.add_argument(help='number of detector channels',type=int,dest='n_ch')
    p.add_argument(help='number of orbits over which to average accidentals',type=int,dest='n_orb')
    p.add_argument('--skipwrite',help='do not write the result to a CSV',action='store_false')
    p.add_argument('--noatten',help='do not correct for attenuation',action='store_false')
    a=p.parse_args()

    combine_data(a.wdir,a.n_ch,a.n_bins,a.n_orb,a.e_bin,a.noatten,a.skipwrite)