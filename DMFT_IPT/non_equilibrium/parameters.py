import numpy as np

# Constants

mu      = 0.        # Chemical potential
Gamma   = 8.e-3     # Damping parameter
e_d     = 0.        # Energy of localized electrons
e_p     = -1.       # Energy of conduction electrons
t       = 0.5       # Hopping amplitude 
V       = 0.9       # d-p electrons hybridization

# Input

dw      = 1.e-2                         # Frequency step
minW    = -5.                           # Smallest frequency
maxW    = 5.                            # Highest frequency
wArr    = np.arange(minW, maxW, dw)     # Frequencies