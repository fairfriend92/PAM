import numpy as np

# Constants

mu      = 0.        # Chemical potential
Gamma   = 8.e-3     # Damping parameter
e_d     = 0.        # Energy of localized electrons
e_p     = -1.       # Energy of conduction electrons
t       = 0.5       # Hopping amplitude 
V       = 0.9       # d-p electrons hybridization
U       = 2.0       # Coulomb interaction     
E       = 0.5       # Electric field

# Input

dw          = 1.e-2                         # Frequency step
minW        = -5.                           # Smallest frequency
maxW        = 5.                            # Highest frequency
wArr        = np.arange(minW, maxW, dw)     # Frequencies
N_w         = len(wArr)                     # Number of frequencies

Sig_URArr   = U**2/(4*wArr)                 # Initial Coulomb self-energy, retarded component


de      = 2.*t/256.                     # Lattice energy step  
eArr    = np.arange(-2*t, 2*t, de)      # Lattice energy