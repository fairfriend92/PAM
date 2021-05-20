'''
 this one is the code for IHM  HF calculation, here t2 =0
'''

import HF_PAM_bethe
import numpy as np
import matplotlib.pyplot as plt

D = 1.0
Ec = 0.0
Ef = 0.0
V = 0.5


nA_up = 0.0
nA_dn = 0.5
nB_up = 0.0
nB_dn = 0.5

n_loop = 100
mix = 0.5
N_old = [nA_up,nA_dn,nB_up,nB_dn]
U=2.0
f = open('PAM_HF_U%sV%sEf%s.dat'%(U,V,U/2),'w')
f.write('# Ec,nA_up, nA_dn, nB_up, nB_dn, ntotal, it\n')

#print >> f, 
nB=0.5
for Ec in [0.0,0.25,0.75,1.0,1.25,1.5]:
	
	Ef = -U/2.0
	HF = HF_PAM_bethe.HF(D, Ec, Ef,  V)

	iteration = range(n_loop)
	nB_old = nB
	for it in iteration:
		if it !=0: nB_old = nB
		if it%10.0 == 0.0: print(nB_old)
		nA_up,nA_dn,nB_up,nB_dn = HF.solve(U,nB)
		nB_new = nB_up+nB_dn
		nB = mix*nB_new + (1-mix)*nB_old
		
	f.write("%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n"%(Ec,nA_up,nA_dn,nB_up,nB_dn, nA_up+nA_dn+nB_up+nB_dn,it))	
	print('Ec,nA_up,nA_dn,nB_up,nB_dn,ntotal,it', Ec,nA_up,nA_dn,nB_up,nB_dn, nA_up+nA_dn+nB_up+nB_dn,it)
	#print >>f, U,nA_up, nA_dn, nB_up, nB_dn, nA_up-nA_dn, nB_up-nB_dn, 0.5*abs(nA_up-nA_dn - nB_up+nB_dn), 0.5*abs(nA_up-nA_dn + nB_up-nB_dn), nA_up + nA_dn + nB_up + nB_dn
	#plt.plot(HF.energy,HF.rho)
	np.savetxt('dosU%sV%sEc%sEf%s.dat'%(U,V,Ec,Ef), np.transpose([HF.energy,HF.lambda_up,HF.lambda_up]))
    
    # We plot the energy bands
	plt.plot(HF.energy,HF.lambda_up,'--',label = 'lambda_up')
	plt.plot(HF.energy,HF.lambda_dn,'--',label = 'lambda_dn')
    
	plt.legend()
	plt.text(-0.5,-0.75,'U=%s V=%s Ec=%s Ef=%s'%(U,V,Ec,Ef))
	plt.savefig('lamda_up_dn_U%sV%sEc%sEf%s.png'%(U,V,Ec,Ef))
	#plt.show()
	

#plt.plot(HF.energy,HF.rho)
#plt.plot(HF.energy,HF.L_up,'--*',label = 'lambda_up')
#plt.plot(HF.energy,HF.L_dn,'--^',label = 'lambda_dn')
#plt.plot(HF.energy,HF.CA_up,'--*',label = 'lambda_up')
#plt.plot(HF.energy,HF.CB_dn,'--^',label = 'lambda_dn')

#plt.legend()
#plt.text(-0.5,-0.75,'D = %s U= %s,Delta = %s'%(U,D,delta))
#plt.savefig('lamda_up_dn%s.eps'%U)
#plt.show()

#nA_up,nA_dn,nB_up,nB_dn (0.058077476354961577, 0.058077476354961577, 6.0656051697190437e-05, 6.0656051697190437e-05)


