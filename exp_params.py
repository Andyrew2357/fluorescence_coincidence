"""Experimental Parameters"""

XMAP_ENERGY_UNIT = 0.02586 # 1 xMAP energy unit (keV)
XMAP_TIME_UNIT = 20. # 1 xMAP time unit (ns)
ORBITAL_PERIOD = 2563.2 # orbital period of the synchrotron (ns)
CHANNEL_SOLID_ANGLE = (0.11, 0.0935, 0.0935, 0.0935) # solid angles of detector channels
CHANNEL_ANGULAR_COEFF = (0.7, 0.6, 0.6, 0.6) # angular factors of detector channels
CHANNEL_DETECTOR = ('90EX', 'ME4', 'ME4', 'ME4')
THICKNESS = ['40nm', '80nm', '160nm', '320nm', 'empty']

import os
mypath = os.path.dirname(os.path.realpath(__file__))
ATTENUATION_PATH = os.path.join(mypath, 'atten_data') # path to a folder describing the effect of attenuation

"""Prediction Parameters"""

K_COEFF = 1.4e-6 
D = 4600
EB = 8.979
E0 = 46
Z = 29
MC2 = 511
ALPHA = 1/137
