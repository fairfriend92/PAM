import numpy as np

''' Constants '''

mu      = 0.        # Chemical potential
Gamma   = 8.e-3     # Damping parameter
e_d     = 0.        # Energy of localized electrons
e_p     = -1.       # Energy of conduction electrons
t       = 0.5       # Hopping amplitude 
D       = 2 * t     # Half-bandwidth
V       = 0.9       # d-p electrons hybridization
U       = 2.0       # Coulomb interaction     
E       = 0.5       # Electric field
L       = 500       # Length of the semi-infinite chain
beta    = 64        # Inverse of temperatures 
numThr  = 4         # Max number of threads

''' Input '''

# Frequencies
dw          = 1.e-2                         # Frequency step
minW        = -5.                           # Smallest frequency
maxW        = 5.                            # Highest frequency
wArr        = np.arange(minW, maxW, dw)     # Frequencies
N_w         = len(wArr)                     # Number of frequencies

# Self-energies
Sig_U_RArr  = U**2/(4*wArr)                     # Initial Coulomb self-energy, retarded component
Sig_U_K     = 0.                                # Initial Coulomb self-energy, Keldysh component
Sig_B_R     = -1.j*Gamma                        # Bath self-energy, retarded component
Sig_B_KArr  = -1.j*np.tanh(beta*wArr/2.)*Gamma  # Bath self-energy, Keldysh component

# Energies
de      = 2.*t/256.                                 # Lattice energy step  
eArr    = np.arange(-2*t, 2*t, de)                  # Lattice energy
N_e     = len(eArr)                                 # Len of energy array
dosArr  = 2*np.sqrt(D**2 - eArr**2)/(np.pi * D**2)  # Bethe lattice DOS