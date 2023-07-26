import numpy as np
from collections import defaultdict
from scipy.io import netcdf_file

class Processor:
    
    def __init__(self,t_factor,e_factor,t_orb,fl_l,fl_h,n_bins,n_ch,n_orb,sim=False):
        self.t_factor=t_factor
        self.e_factor=e_factor
        self.t_orb=t_orb
        self.fl_l=fl_l
        self.fl_h=fl_h
        self.n_bins=n_bins
        self.n_ch=n_ch
        self.n_orb=n_orb
        self.sim=sim
    
    def process_file(self,path):
        """process the relevant file for coincidences and accidentals and add this info to
        the reduced data array."""
        reduced_coin = np.zeros((self.n_ch*self.n_ch,self.n_bins),dtype=int)
        reduced_acoin = np.zeros((self.n_ch*self.n_ch,self.n_bins),dtype=int)
        fluor_count = np.zeros((self.n_ch),dtype=int)
        events = self.read(path,self.sim)
        
        # create a dictionary for events binned in time and energy for given channels
        scattered = defaultdict(list)
        for e in events:
            scattered[(e['channel'],int(e['time']*self.t_factor))].append(int(e['E']*self.e_factor))

        # loop through events keeping track of the sum of coincident and acoincident
        # counts at each energy level. The sum is sufficient for poisson stats.
        
        for e in events:
            if e['E'] < self.fl_l or e['E'] > self.fl_h: continue
            
            fluor_count[e['channel']]+=1
            t_in = int(e['time']*self.t_factor)
            
            for sc in range(self.n_ch):
                if e['channel'] == sc: continue

                for E in scattered[(sc,t_in)]: reduced_coin[self.n_ch*e['channel']+sc,E]+=1
                for i in range(1,self.n_orb+1):
                    t_off = int((e['time']-i*self.t_orb)*self.t_factor)
                    for E in scattered[(sc,t_off)]: reduced_acoin[self.n_ch*e['channel']+sc,E]+=1
        
        return (reduced_coin,reduced_acoin,fluor_count)
    
    def read(self,path,simulated=False):
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