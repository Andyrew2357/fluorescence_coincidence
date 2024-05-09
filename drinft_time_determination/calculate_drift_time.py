"""Empirically determine the variation in detection time for all detectors using
data from a time structured source. This is a quick and dirty implementation."""

__author__="Andrew DiFabbio"
__email__="avd38@cornell.edu"

import numpy as np
from joblib import Parallel, delayed
import argparse
import os
from scipy.io import netcdf_file
from scipy import integrate
import multiprocessing

# Parameters
XMAP_TIME_UNIT=20. # 1 xMAP time unit (ns)
ORBITAL_PERIOD=2563.2 # orbital period of the synchrotron (ns)
DIRNAME = os.path.dirname(__file__)
BEAM_STRUCT_PATH=os.path.join(DIRNAME, 'beam_time_struct_9_26_23.csv')
fl_l,fl_h=285,370
# fl_l,fl_h=0,3000

n_off=int(ORBITAL_PERIOD/XMAP_TIME_UNIT)
b_struct=np.genfromtxt(BEAM_STRUCT_PATH,delimiter=',').T

def read(path,simulated=False):
        if simulated: return np.load(path)
        """Read in a netcdf (.nc) file from disk and return the data contained
        in the form of a numpy structured array."""
        
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

def channel_sig(path):
    print(path)
    evts=read(path)
    res=np.zeros((n_ch,n_off))

    sig=set()
    for t,E,ch in evts:
        if fl_l<=E<=fl_h: sig.add((t,ch))

    for t,E,ch in evts:
        if E<fl_l or E>fl_h: continue
        for t_off in range(n_off):
             if (t+int(t_off+0.5*n_off),ch) in sig: res[ch,t_off]+=1
    
    return res 

# BASED ON THE ONE PARAMETER MODEL ------------------------------------------------------
def f(tau,T):
    vt=np.arange(T)

    # Convolve the beam time structure with variation in detection time.
    # This is what a detector should effectively see.
    conv=np.convolve(vt,b_struct,mode='same')

    # Now take an autocorrelation of the result
    out = np.array([integrate.simpson(conv*np.roll(conv,int(t))) for t in tau])
    return out/integrate.simpson(out) 

def recover(sig):
    N,=b_struct.shape
    tau=np.arange(int(0.5*N),int(1.5*N))
    
    bestT = 0
    bestCost = 1e9
    for t in range(1, N):
        cost = integrate.simpson((sig - f(tau, t))**2)
        if cost < bestCost: 
            bestT = t
            bestCost = cost

    return bestT/np.sqrt(18)

# BASED ON THE ONE PARAMETER MODEL ------------------------------------------------------

def find_vt(save=True):
    paths=[os.path.join(ddir,p) for p in os.listdir(ddir)]
    paths=[s for s in paths if os.path.getsize(s)>10000]
    to_reduce=Parallel(n_jobs=multiprocessing.cpu_count())(delayed(channel_sig)(p) for p in paths)
    res=np.zeros((n_ch,n_off))
    for r in to_reduce: res+=r

    if save: np.savetxt(os.path.join(wdir,'channel_sig.txt'),res,delimiter=',')

    signal=np.zeros((n_ch,n_off))
    for r in range(n_ch): signal[r,:]+=res[r,:]/integrate.simpson(res[r,:])

    with open(os.path.join(wdir,'find_vt_log.txt'),'w') as fl:
        fl.write('find_vt Log File \n')

        for ch in range(n_ch):
            vartime=recover(signal[ch,:].copy())
            print(f'Recovered Variation in Detection Time for Channel {ch}: {vartime}')
            fl.write(f'Recovered Variation in Detection Time for Channel {ch}: {vartime}\n')

if __name__=='__main__':

    p=argparse.ArgumentParser()

    p.add_argument('ddir')
    p.add_argument('wdir')
    p.add_argument('n_ch',type=int)

    a=p.parse_args()
    ddir,wdir,n_ch=a.ddir,a.wdir,a.n_ch

    find_vt()



# python calculate_drift_time.py F:\FranckGroup\IAB_2018_Raw_Data\franck-2376-1\pDbp17p2cm\list C:\Users\avdif\Documents\FranckGroup\calculate_drift_time_test_2_5_24 4
