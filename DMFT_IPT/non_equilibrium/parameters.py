import numpy as np

''' Constants '''

mu      = 0.5                                       # Chemical potential
Gamma   = 8.e-3                                     # Damping parameter
#Gamma   = 3.e-1
e_d     = 0.                                        # Energy of localized electrons
e_p     = -1.                                       # Energy of conduction electrons
t       = 0.5                                       # Hopping amplitude 
D       = 2.*t                                      # Half-bandwidth
V       = 0.9                                       # d-p electrons hybridization
E       = 0.0                                       # Electric field
L       = 500                                       # Length of the semi-infinite chain
error   = 1.e-3                                     # Convergence criterium

''' Input '''

# Coulomb interaction

dU      = 1.                                        # Coulomb interaction step
minU    = 0.                                        # Lowest Coulomb interaction
maxU    = 2.                                        # Highest Coulomb interaction
UArr    = np.arange(minU, maxU, dU)                 # Array of Us
UArr    = [0.]
N_U     = len(UArr)                                 # Number of Us

# Temperatures

dBeta       = 8.                                    # Inverse of temp. step
minBeta     = 64.                                   # Lowest inverse of temp.
maxBeta     = 72.                                   # Highest inverse of temp.
betaArr     = np.arange(minBeta, maxBeta, dBeta)    # Array of betas
betaArr     = [64.]
N_beta      = len(betaArr)                          # Number of betas

# Frequencies
dw          = 1.e-2                                 # Frequency step
minW        = -5.                                   # lowest frequency
maxW        = 5.                                    # Highest frequency
wArr        = np.arange(minW, maxW, dw)             # Frequencies
N_w         = len(wArr)                             # Number of frequencies

# Self-energies
Sig_U_RArr  = minU**2/(4.*wArr)                     # Initial Coulomb self-energy, retarded component
Sig_U_KArr  = 0.*np.zeros(N_w)                      # Initial Coulomb self-energy, Keldysh component
Sig_B_RArr  = -1.j*Gamma*np.ones(N_w)               # Bath self-energy, retarded component
Sig_B_KArr  = -1.j*np.tanh(minBeta*wArr/2.)*Gamma   # Bath self-energy, Keldysh component

# Momentum
dk      = np.pi/256.                                # Momenta separation
kArr    = np.arange(-np.pi, np.pi, dk)              # Momenta array
N_k     = len(kArr)                                 # Number of momentum vectors
e_kArr  = -2.*t*np.cos(kArr)                        # Lattice energy

# Broadcast arrays to the appropriate matrix shapes
wMtrx       = np.transpose(wArr*np.ones([N_k, N_w]))
e_kMtrx     = e_kArr*np.ones([N_w, N_k])
