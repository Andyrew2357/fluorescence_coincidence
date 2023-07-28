__author__ = "Andrew DiFabbio"
__email__ = "avd38@cornell.edu"

import argparse
import numpy as np
import os
from matplotlib import pyplot as plt
import exp_params as exp
from scipy.stats import chi2

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
            ang_corr[r]+=exp.CHANNEL_ANGULAR_COEFF[ch]*exp.CHANNEL_SOLID_ANGLE[ch]

    return np.power(ang_corr,-1)

def combine_summed_results(data,num_ch,num_orb,e_bin):
    ang_corr=get_ang_corr(num_ch)
    fl_ct,coin,acoin=data

    excesses=ang_corr*(coin-acoin/num_orb)/fl_ct
    variances=ang_corr**2*((coin+acoin/num_orb**2)/fl_ct**2 + (coin-acoin/num_orb)**2/fl_ct**3)
    
    norm_vars=np.power(np.sum(np.power(variances,-1),axis=0),-1)
    
    return (norm_vars*np.sum(excesses/variances,axis=0))/(e_bin*exp.XMAP_ENERGY_UNIT), \
        np.sqrt(norm_vars)/(e_bin*exp.XMAP_ENERGY_UNIT)

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
    
def k(t):
    return exp.K_COEFF*(t/exp.D)**0.75

def W_secondaries(t,Es):
    return 2*k(t)*exp.Z*(exp.E0-exp.EB-Es)/(4*np.pi*Es)

def W_LET(Es):
    return 2*exp.ALPHA*((exp.E0-exp.EB)/exp.MC2)/(4*np.pi**2*Es)

def plot_thickness(ax,exc,err,Emin,Emax,e_bin,t):
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

def calc_chisq(obs,err,pred,rnd=True):
    chi_sq=np.sum((obs-pred)**2/err**2)
    nu=obs.shape
    p=(1-chi2.cdf(chi_sq,nu))[0]
    if rnd: return round(chi_sq,3),round(p,5)
    return chi_sq,p

def excess_chisq(wdir,Emin,Emax,e_bin,save=False):
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

if __name__=='__main__':
    # Parsing command line arguments
    p = argparse.ArgumentParser()
    sp=p.add_subparsers(dest='command')

    # plot command
    plot_p=sp.add_parser('plot',help='plot excesses command')
    plot_p.add_argument(help='working directory',dest='wdir')
    plot_p.add_argument(help='length of energy bin (xMAP units)',type=int,dest='e_bin')
    plot_p.add_argument(help='lower bound of scattered energies to plot (keV)',type=float,dest='Emin')
    plot_p.add_argument(help='upper bound of scattered energies to plot (keV)',type=float,dest='Emax')
    plot_p.add_argument('--save_plt',help='save the plot generated',action='store_true')

    # combine command
    combine_p=sp.add_parser('combine',help='combine reduced data')
    combine_p.add_argument(help='working directory',dest='wdir') 
    combine_p.add_argument(help='length of energy bin (xMAP units)',type=int,dest='e_bin')
    combine_p.add_argument(help='number of bins in final histogram',type=int,dest='n_bins') 
    combine_p.add_argument(help='number of detector channels',type=int,dest='n_ch')
    combine_p.add_argument(help='number of orbits over which to average accidentals',type=int,dest='n_orb')
    combine_p.add_argument('--skipwrite',help='do not write the result to a CSV',action='store_false')
    combine_p.add_argument('--noatten',help='do not correct for attenuation',action='store_false')

    # chisq command
    chisq_p=sp.add_parser('chisq',help='calculate chi square results')
    chisq_p.add_argument(help='working directory',dest='wdir')
    chisq_p.add_argument(help='length of energy bin (xMAP units)',type=int,dest='e_bin')
    chisq_p.add_argument(help='lower bound of scattered energies to plot (keV)',type=float,dest='Emin')
    chisq_p.add_argument(help='upper bound of scattered energies to plot (keV)',type=float,dest='Emax')
    chisq_p.add_argument('--save_chi2',help='save the plot generated',action='store_true')

    a=p.parse_args()
    if a.command=='plot':
        plot(a.wdir,a.Emin,a.Emax,a.e_bin,a.save_plt)
    elif a.command=='combine':
        combine_data(a.wdir,a.n_ch,a.n_bins,a.n_orb,a.e_bin,a.noatten,a.skipwrite)
    elif a.command=='chisq':
        excess_chisq(a.wdir,a.Emin,a.Emax,a.e_bin,a.save_chi2)
