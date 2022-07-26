from parameters import *

# Compute the inverse of the RHS or the LHS semi-infinite chain
def get_FInv(G_00Inv,   
             sign)                  
    # Last element of the chain
    FInv = G_00Inv + sign*L*E
    
    # Continued fraction
    for x in np.arange(sign*(L-1.), 0.):
        FInv = G_00Inv + x*E - t**2/FInv
    
# Inverse of the central site dd Green function
G_00Inv = 1.j*wArr + mu - eArr*np.ones(N_w) + 1.j*Gamma + - V**2/(1.j*wArr + mu - e_d - Sig_URArr)

# Inverse of the semi-infinite chains
F_rhsInv = get_FInv(G_00Inv, +1)
F_lhsInv = get_FInv(G_00Inv, -1)

# Inverse of the dd Green function: G(w, e)_dd^-1
Gwe_ddInv = G_00Inv - t**2(1./F_rhsInv + 1./F_lhsInv)