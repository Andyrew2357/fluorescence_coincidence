__author__ = "Andrew DiFabbio"
__email__ = "avd38@cornell.edu"

import argparse
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import exp_params as exp
from scipy.stats import chi2

np.seterr(divide='ignore', invalid='ignore')

# COMBINING REDUCED DATA----------------------------------------------------------------------------------------
def sum_over_results(path,num_ch,num_bins,verbose=False):
    """Sum the coincidences, accidentals, and fluorescence counts for a thickness within a 
    folder, returning the result in the format <fluorescence counts>,<coincidences>,
    <accidentals>. 
    
    If verbose, these results are returned with respect to every permuation of fluorescence
    and scattered detector. Otherwise they are simply returned according to fluorescence
    detector."""

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
    """Calculate the factor for angular dependence in IAB as it relates to the detectors. This
    is essentially an integral over the solid angle of the angular coefficient."""

    ang_corr=np.zeros((num_ch,1))
    for r in range(num_ch):
        for ch in range(num_ch):
            if r==ch: continue
            ang_corr[r]+=exp.CHANNEL_ANGULAR_COEFF[ch]*exp.CHANNEL_SOLID_ANGLE[ch]

    return np.power(ang_corr,-1)

def combine_summed_results(data,num_ch,num_orb,e_bin,n_bins,atten=True,thick=None):
    """Combine the already summed counts by applying angular and attenuation corrections and 
    handling error propagation as necessary. Returns <normalized excess>,<normalized error>.
    
    If atten, attenuation correcion is applied, and a thickness must be given in the form
    'data/40nm/', for example."""

    if atten and thick == None: raise Exception('combine_summed_results: attenuation correction needs thickness.')
    ang_corr=get_ang_corr(num_ch)
    fl_ct,coin,acoin=data

    excesses=ang_corr*(coin-acoin/num_orb)/fl_ct
    variances=ang_corr**2*((coin+acoin/num_orb**2)/fl_ct**2 + (coin-acoin/num_orb)**2/fl_ct**3)
    if atten: excesses,variances=correct_atten(excesses,variances,e_bin,n_bins,thick)

    norm_vars=np.power(np.sum(np.power(variances,-1),axis=0),-1)
    
    return (norm_vars*np.sum(excesses/variances,axis=0))/(e_bin*exp.XMAP_ENERGY_UNIT), \
        np.sqrt(norm_vars)/(e_bin*exp.XMAP_ENERGY_UNIT)

def combine_folder(path,num_ch,num_bins,num_orb,e_bin,atten=True,thick=None):
    """Read and combine the data for a given thickness."""

    if atten and thick == None: raise Exception('combine_folder: attenuation correction needs thickness.')
    return combine_summed_results(sum_over_results(path,num_ch,num_bins),num_ch,num_orb,e_bin,num_bins,atten,thick)

def combine_data(wdir,num_ch,num_bins,num_orb,e_bin,atten=True,write=True):
    """Fully combine the reduced data. If atten, attenuation correction is applied. If write, data
    will be written to CSV files afterward, otherwise the result is returned. The results consist
    of normalized excess and corresponding errors with rows representing thickness in the order 40nm,
    80nm, 160nm, 320nm, empty, and columns representing energy bin."""

    result_excesses=np.zeros((5,num_bins))
    result_errors=np.zeros((5,num_bins))

    for r,t in enumerate(['data/40nm/','data/80nm/','data/160nm/','data/320nm/','data/empty/']):
        exc,err=combine_folder(os.path.join(wdir,t),num_ch,num_bins,num_orb,e_bin,atten,t)
        result_excesses[r,:]+=exc
        result_errors[r:]+=err

    if write: 
        np.savetxt(os.path.join(wdir,'combined_result_excesses.csv'),result_excesses,delimiter=',')
        np.savetxt(os.path.join(wdir,'combined_result_errors.csv'),result_errors,delimiter=',')
    else:
        return result_excesses,result_errors
   
# PLOTTING COMBINED RESULTS AND PREDICTIONS---------------------------------------------------------------------
def k(t):
    return exp.K_COEFF*(t/exp.D)**0.75

def W_secondaries(t,Es):
    return 2*k(t)*exp.Z*(exp.E0-exp.EB-Es)/(4*np.pi*Es)

def W_LET(Es):
    return 2*exp.ALPHA*((exp.E0-exp.EB)/exp.MC2)/(4*np.pi**2*Es)

def plot_thickness(ax,exc,err,Emin,Emax,e_bin,t):
    """Plot the combined results for a given thickness and energy range along with the predicted rates
    for that thickness. Blue for secondaries, red for LET, and green for secondaries+LET."""

    start,stop=int(Emin/(e_bin*exp.XMAP_ENERGY_UNIT)),int(Emax/(e_bin*exp.XMAP_ENERGY_UNIT)+0.5)
    Erange=np.arange((start+0.5)*(e_bin*exp.XMAP_ENERGY_UNIT), \
                     (stop+0.5)*(e_bin*exp.XMAP_ENERGY_UNIT),e_bin*exp.XMAP_ENERGY_UNIT)
    Erange_=np.linspace((start+0.5)*(e_bin*exp.XMAP_ENERGY_UNIT), \
                      (stop+0.5)*(e_bin*exp.XMAP_ENERGY_UNIT),1000)
    # plot predictions
    ax.plot(Erange_,0*Erange_,color='k')
    ax.plot(Erange_,W_secondaries(t,Erange_),color='b')
    ax.plot(Erange_,W_LET(Erange_),color='r')
    ax.plot(Erange_,W_LET(Erange_)+W_secondaries(t,Erange_),color='g')

    # plot excesses
    ax.errorbar(Erange,exc[start:stop],yerr=err[start:stop],ls='none')
    ax.scatter(Erange,exc[start:stop])
    ax.set_title(f'{t}nm')
    ax.set_xlabel('Energy (keV)')
    ax.set_ylabel('Normalized Excess Rate')

def plot(wdir,Emin,Emax,e_bin,save=False):
    """Plot the combined results and predictions for 40nm,80nm,160nm,320nm thicknesses in a given energy
    range."""

    excess=np.genfromtxt(os.path.join(wdir,'combined_result_excesses.csv'),delimiter=',')
    errors=np.genfromtxt(os.path.join(wdir,'combined_result_errors.csv'),delimiter=',')

    fig,((ax0,ax1),(ax2,ax3))=plt.subplots(2,2)
    plot_thickness(ax0,excess[0,:],errors[0,:],Emin,Emax,e_bin,40)
    plot_thickness(ax1,excess[1,:],errors[1,:],Emin,Emax,e_bin,80)
    plot_thickness(ax2,excess[2,:],errors[2,:],Emin,Emax,e_bin,160)
    plot_thickness(ax3,excess[3,:],errors[3,:],Emin,Emax,e_bin,320)
    plt.tight_layout()

    if save:
        if not os.path.isdir(os.path.join(wdir,'figures/')): os.mkdir(os.path.join(wdir,'figures/'))
        plt.savefig(os.path.join(os.path.join(wdir,'figures/'), \
                                 f'normalized_excess_plots_(Emin={Emin},Emax={Emax}).png'))
    else:
        plt.show()

# CALCULATE AND REPORT CHI SQUARE VALUES------------------------------------------------------------------------
def calc_chisq(obs,err,pred,rnd=True):
    """Calculate the chi square and probability given observed results, errors, and predicted results.
    If rnd, round the returned values accordingly."""

    chi_sq=np.sum((obs-pred)**2/err**2)
    nu=obs.shape
    p=(1-chi2.cdf(chi_sq,nu))[0]
    if rnd: return round(chi_sq,3),round(p,5)
    return chi_sq,p

def excess_chisq(wdir,Emin,Emax,e_bin,save=False):
    """Compute the chi square and probability between observed and predicted results of the combined data
    given an energy range. Print the results as a table to the command line and write them to a TXT"""
    

    excess=np.genfromtxt(os.path.join(wdir,'combined_result_excesses.csv'),delimiter=',')
    errors=np.genfromtxt(os.path.join(wdir,'combined_result_errors.csv'),delimiter=',')
    start,stop=int(Emin/(e_bin*exp.XMAP_ENERGY_UNIT)),int(Emax/(e_bin*exp.XMAP_ENERGY_UNIT)+0.5)
    Erange=np.arange((start+0.5)*(e_bin*exp.XMAP_ENERGY_UNIT), \
                     (stop+0.5)*(e_bin*exp.XMAP_ENERGY_UNIT),e_bin*exp.XMAP_ENERGY_UNIT)
    
    table=[('','df','SEC chi2','SEC p','LET chi2','LET p','TOT chi2','TOT p')]
    df=(Erange.shape)[0]
    for r,t in enumerate([40,80,160,320]):
        SECchi2,SECp=calc_chisq(excess[r,start:stop],errors[r,start:stop],W_secondaries(t,Erange))
        LETchi2,LETp=calc_chisq(excess[r,start:stop],errors[r,start:stop],W_LET(Erange))
        TOTchi2,TOTp=calc_chisq(excess[r,start:stop],errors[r,start:stop],W_LET(Erange)+W_secondaries(t,Erange))
        table.append((str(s) for s in (f'{t}nm',df,SECchi2,SECp,LETchi2,LETp,TOTchi2,TOTp)))

    if save:
        if not os.path.isdir(os.path.join(wdir,'chisq/')): os.mkdir(os.path.join(wdir,'chisq/'))
        with open(os.path.join(os.path.join(wdir,'chisq/'),f'chisq_table_(Emin={Emin},Emax={Emax}).txt'),'w+') as f:
            for row in table: f.write('| {:>4} | {:>2} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} |\n'.format(*row))
    for row in table: print('| {:>6} | {:>2} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} |'.format(*row))

# CALCULATE AND PRESENT CORRECTIONS FOR ATTENUATION-------------------------------------------------------------
def get_atten(Erange,n_ch=4,verbose=False):
    """Compute the attentuation from all sources for a given energy range. If verbose, the result includes the
    raw transmittance for every source of attenuation in the format <He>,<kapton>,<1-elem detector QE>,<4-elem
    detector QE>,<320nm>,<160nm>,<80nm>,<40nm>. 
    Otherwise, it returns transmittance results for each mode (based on the detector channel that saw the 
    fluorescence event) and for each target thickness relative to the transmittance at fluorescence energy.
    These are returned in the format <average mode transmittance>,(<320nm>,<160nm>,<80nm>,<40nm>).
    
    This function, along with other functions related to attenuation correction, will likely need to be modified
    if any of the experimental conditions change. This is far less generalizable than everything else."""

    atten_raw=pd.read_csv(exp.ATTENUATION_DOC_PATH)
    He_T=np.interp(Erange,atten_raw['He_E'].values,atten_raw['He_T'].values)
    Kap_T=np.interp(Erange,atten_raw['Kap_E'].values,atten_raw['Kap_T'].values)
    D1_T=np.interp(Erange,atten_raw['D_E'].values,atten_raw['D1_T'].values)
    D4_T=np.interp(Erange,atten_raw['D_E'].values,atten_raw['D4_T'].values)
    M320_T=np.interp(Erange,atten_raw['M_E'].values,atten_raw['M320_T'].values)
    M160_T=np.interp(Erange,atten_raw['M_E'].values,atten_raw['M160_T'].values)
    M80_T=np.interp(Erange,atten_raw['M_E'].values,atten_raw['M80_T'].values)
    M40_T=np.interp(Erange,atten_raw['M_E'].values,atten_raw['M40_T'].values)

    if verbose: return He_T,Kap_T,D1_T,D4_T,M320_T,M160_T,M80_T,M40_T 
    
    D1_Tot=He_T*Kap_T*D1_T
    D4_Tot=He_T*Kap_T*D4_T
    Mode_T=np.zeros((n_ch,len(Erange)))
    
    for r in range(n_ch): 
        S=0
        for ch in range(n_ch):
            if r==ch: continue
            S+=exp.CHANNEL_SOLID_ANGLE[ch]*exp.CHANNEL_ANGULAR_COEFF[ch]
            if ch == 0:
                Mode_T[r,:]+=exp.CHANNEL_SOLID_ANGLE[ch]*exp.CHANNEL_ANGULAR_COEFF[ch]*D1_Tot
            else:
                Mode_T[r,:]+=exp.CHANNEL_SOLID_ANGLE[ch]*exp.CHANNEL_ANGULAR_COEFF[ch]*D4_Tot
        
        Mode_T[r,:]/=S
        if r == 0:
            Mode_T[r,:]/=np.interp(exp.E_FLUOR,Erange,D1_Tot)
        else:
            Mode_T[r,:]/=np.interp(exp.E_FLUOR,Erange,D4_Tot)

    M320_T/=np.interp(exp.E_FLUOR,Erange,M320_T)
    M160_T/=np.interp(exp.E_FLUOR,Erange,M160_T)
    M80_T/=np.interp(exp.E_FLUOR,Erange,M80_T)
    M40_T/=np.interp(exp.E_FLUOR,Erange,M40_T)

    return Mode_T,(M320_T,M160_T,M80_T,M40_T)

def correct_atten(exc,err,e_bin,n_bins,t):
    """Similarly to how corrections are applied for the angular component, this function corrects the excesses
    and energies for attenuation."""

    Mode_T,M_T=get_atten(exp.XMAP_ENERGY_UNIT*e_bin*np.arange(n_bins),4)
    th={'data/40nm/':3,'data/80nm/':2,'data/160nm/':1,'data/320nm/':0}

    if t == 'data/empty/': return exc/Mode_T,err/Mode_T
    return exc/(Mode_T*M_T[th[t]]),err/(Mode_T*M_T[th[t]])

def plot_atten(Emin,Emax,n_ch=4,save=False,wdir=None):
    """Plot the transmittance for all sources of attenuation over a given energy range. If save, save the result
    as a figure."""

    if save and wdir == None: raise Exception('Instructed to save with no wdir provided.')

    Erange=np.linspace(Emin,Emax,1000)
    He_T,Kap_T,D1_T,D4_T,M320_T,M160_T,M80_T,M40_T=get_atten(Erange,n_ch,True)
    
    plt.plot(Erange,M40_T,label='Material Transmittance 40nm')
    plt.plot(Erange,M80_T,label='Material Transmittance 80nm')
    plt.plot(Erange,M160_T,label='Material Transmittance 160nm')
    plt.plot(Erange,M320_T,label='Material Transmittance 320nm')
    plt.plot(Erange,He_T,label='Helium Transmittance')
    plt.plot(Erange,Kap_T,label='Kapton Transmittance')
    plt.plot(Erange,D1_T,label='1-Elem Detector QE')
    plt.plot(Erange,D4_T,label='4-Eleme Detector QE')
    
    plt.legend(loc='lower right')
    plt.title('Sources of Attenuation')
    plt.xlabel('Energy (keV)')
    plt.ylabel('Transmittance')
    plt.tight_layout()

    if save:
        if not os.path.isdir(os.path.join(wdir,'figures/')): os.mkdir(os.path.join(wdir,'figures/'))
        plt.savefig(os.path.join(os.path.join(wdir,'figures/'), \
                                 f'atten_sources_(Emin={Emin},Emax={Emax}).png'))
    else:
        plt.show()

# HANDLING COMMAND LINE ARGUMENTS-------------------------------------------------------------------------------
if __name__=='__main__':
    # Parsing command line arguments
    p = argparse.ArgumentParser()
    sp=p.add_subparsers(dest='command')

    # combine command
    """USAGE: python <PATH TO fl_coin>/fl_coin/calc_exc.py combine <PATH TO WORKING DIRECTORY> <ENERGY BIN> \\
    <NUMBER OF BINS> <NUMBER OF CHANNELS> <NUMBER OF ORBITS> <--skipwrite> <--noatten>"""

    combine_p=sp.add_parser('combine',help='combine reduced data')
    combine_p.add_argument(help='working directory',dest='wdir') 
    combine_p.add_argument(help='length of energy bin (xMAP units)',type=int,dest='e_bin')
    combine_p.add_argument(help='number of bins in final histogram',type=int,dest='n_bins') 
    combine_p.add_argument(help='number of detector channels',type=int,dest='n_ch')
    combine_p.add_argument(help='number of orbits over which to average accidentals',type=int,dest='n_orb')
    combine_p.add_argument('--skipwrite',help='do not write the result to a CSV',action='store_false')
    combine_p.add_argument('--noatten',help='do not correct for attenuation',action='store_false')

    # plot command
    """USAGE: python <PATH TO fl_coin>/fl_coin/calc_exc.py plot <PATH TO WORKING DIRECTORY> <ENERGY BIN> \\
    <LOWER ENERGY BOUND> <UPPER ENERGY BOUND> <--save>"""

    plot_p=sp.add_parser('plot',help='plot excesses command')
    plot_p.add_argument(help='working directory',dest='wdir')
    plot_p.add_argument(help='length of energy bin (xMAP units)',type=int,dest='e_bin')
    plot_p.add_argument(help='lower bound of scattered energies to plot (keV)',type=float,dest='Emin')
    plot_p.add_argument(help='upper bound of scattered energies to plot (keV)',type=float,dest='Emax')
    plot_p.add_argument('--save',help='save the plot generated',action='store_true')

    # chisq command
    """USAGE: python <PATH TO fl_coin>/fl_coin/calc_exc.py chisq <PATH TO WORKING DIRECTORY> <ENERGY BIN> \\
    <LOWER ENERGY BOUND> <UPPER ENERGY BOUND> <--save>"""

    chisq_p=sp.add_parser('chisq',help='calculate chi square results')
    chisq_p.add_argument(help='working directory',dest='wdir')
    chisq_p.add_argument(help='length of energy bin (xMAP units)',type=int,dest='e_bin')
    chisq_p.add_argument(help='lower bound of scattered energies to plot (keV)',type=float,dest='Emin')
    chisq_p.add_argument(help='upper bound of scattered energies to plot (keV)',type=float,dest='Emax')
    chisq_p.add_argument('--save',help='save the plot generated',action='store_true')

    # pltatten command
    """USAGE: python <PATH TO fl_coin>/fl_coin/calc_exc.py pltatten <LOWER ENERGY BOUND> <UPPER ENERGY BOUND> \\
    <--wdir PATH TO WORKING DIRECTORY> <--save>"""

    pltatten_p=sp.add_parser('pltatten',help='plot sources of attenuation')
    pltatten_p.add_argument(help='lower bound of scattered energies to plot (keV)',type=float,dest='Emin')
    pltatten_p.add_argument(help='upper bound of scattered energies to plot (keV)',type=float,dest='Emax')
    pltatten_p.add_argument('--wdir',help='working directory for saving the plot',default=None)
    pltatten_p.add_argument('--save',help='save the plot generated',action='store_true')

    a=p.parse_args()
    if a.command=='plot':
        plot(a.wdir,a.Emin,a.Emax,a.e_bin,a.save)
    elif a.command=='combine':
        combine_data(a.wdir,a.n_ch,a.n_bins,a.n_orb,a.e_bin,a.noatten,a.skipwrite)
    elif a.command=='chisq':
        excess_chisq(a.wdir,a.Emin,a.Emax,a.e_bin,a.save)
    elif a.command=='pltatten':
        plot_atten(a.Emin,a.Emax,4,a.save,a.wdir)
