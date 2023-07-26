__author__ = "Andrew DiFabbio"
__email__ = "avd38@cornell.edu"

import numpy as np
import os
from matplotlib import pyplot as plt
import exp_params

def sum_over_results(path,num_ch,num_bins,verbose=False):
    fl_ct=np.zeros((num_ch,1))
    coin=np.zeros((num_ch*num_ch,num_bins))
    acoin=np.zeros((num_ch*num_ch,num_bins))
    
    for l in os.listdir():
        if '.txt' in l: continue

        if 'fluor_count' in l:
            fl_ct+=np.genfromtxt(os.path.join(path,l),delimiter=',')
        elif 'coincidence' in l:
            coin+=np.genfromtxt(os.path.join(path,l),delimiter=',')
        elif 'accidentals' in l:
            acoin+=np.genfromtxt(os.path.join(path,l),delimiter=',')
    
    if not verbose:
        coin=np.sum(coin.reshape((num_ch,num_ch,num_bins),axis=1))
        acoin=np.sum(acoin.reshape((num_ch,num_ch,num_bins),axis=1))
    
    return fl_ct,coin,acoin

def get_ang_corr(num_ch):
    ang_corr=np.zeros((num_ch,1))
    for r in range(num_ch):
        for ch in range(num_ch):
            if r==ch: continue
            ang_corr[r]+=exp_params.CHANNEL_ANGULAR_COEFF[ch]*exp_params.CHANNEL_SOLID_ANGLE[ch]

    return np.power(ang_corr,-1)

def combine_summed_results(data,num_ch,num_orb):
    ang_corr=get_ang_corr(num_ch)
    fl_ct,coin,acoin=data

    excesses=ang_corr*(coin-acoin/num_orb)/fl_ct
    variances=ang_corr**2*((coin+acoin/num_orb**2)/fl_ct**2 + (coin-acoin/num_orb)**2/fl_ct**3)
    
    norm_vars=np.power(np.sum(np.power(variances,-1),axis=0),-1)
    
    return norm_vars*np.sum(excesses/variances,axis=0),np.sqrt(norm_vars)

def combine_folder(path,num_ch,num_bins,num_orb):
    return combine_summed_results(sum_over_results(path,num_ch,num_bins),num_ch,num_orb)

def combine_data(wdir,num_ch,num_bins,num_orb,write=True):
    result=np.zeros((5,num_bins))
    for r,t in enumerate(['40nm/','80nm/','160nm/','320nm/','empty/']):
        result[r,:]+=combine_folder(os.path.join(wdir,t),num_ch,num_bins,num_orb)
    
    if write: 
        np.savetxt(os.path.join(wdir,'combined_results.csv'),result,delimiter=',')
    else:
        return result
    
