import numpy as np


class HF:
	def __init__(self, D = 1,Ec = 0.5,Ef = 0.5, V=0.0):
		self.D = D # half-bandwidth 
		self.Ec = Ec # conducting lattice electron energy
		self.Ef = Ef # localize electrons electron energy
		self.V  = V # c-f electrons hybridization term
		print("Ec=%s,Ef=%s,V=%s"%(self.Ec, self.Ef, self.V))
		# if rho is passed as external variable the energy,rho,length,de will be changed
        
        # The range of energy values to be explored is generated - energy is e_k
		self.de = 0.001
		self.energy = np.arange(-self.D + 0.001*self.de,self.D -0.001*self.de,self.de)
		self.length = len(self.energy)
        
        # This is the density of states for the Bethe lattice with NN hopping
		self.rho = np.zeros(self.length)
		for i in range(self.length):
			self.rho[i] = (2* np.sqrt(self.D**2 - self.energy[i]**2)/np.pi )
        
        # Find index i_ of energy value equal to 0
		for i in range(self.length):
			#print i	,self.length
			if self.energy[i] > -self.de /2 and self.energy[i] < self.de /2:
				i_ = i
                
		#print self.energy[i]
		#self.energy.pop(i)
		#self.rho.pop(i)
		self.rho = np.delete(self.rho,i_)
		self.energy = np.delete(self.energy,i_)
		self.length = self.length -1
		print("zero frequency is discarded--------------------------------------------------------")
			
		
		self.lambda_up = np.zeros(self.length)
		self.lambda_dn = np.zeros(self.length)
		self.CA_up = np.zeros(self.length)  #Here A is for conducting lattice(c) and B is for localised electrons(f)
		self.CA_dn = np.zeros(self.length)
		self.CB_up = np.zeros(self.length)
		self.CB_dn = np.zeros(self.length)
		self.A_up = 0.
		self.A_dn = 0.
		self.B_up = 0.
		self.B_dn = 0.
		self.L_up = np.zeros(self.length)
		self.L_dn = np.zeros(self.length)
		
	def solve(self,U,nB):
		self.U = U

		self.B = self.Ef + self.U*nB

		#calculating the enrgy of the IHM                
		for i in range(self.length):
            #EIGEN_VALUES - these are the dispersion relations for the energy bands
			self.lambda_up[i] = 0.5*(  (self.Ec -self.energy[i] + self.B) + np.sqrt( (self.Ec -self.energy[i] - self.B)**2 + 4*self.V**2) )
			self.lambda_dn[i] = 0.5*(  (self.Ec -self.energy[i] + self.B) - np.sqrt( (self.Ec -self.energy[i] - self.B)**2 + 4*self.V**2) )	
			# note that t =0.5
            
            #EIGEN_VECTORS_COORDS
            # 2 eigenvectors for eigenvalues up and down, each eigenvector has 2 coordinates
            # but the first coordinate is just 1 (and is omitted here)
			self.L_up[i] = (self.B - self.lambda_up[i])/self.V
			self.L_dn[i] = (self.B - self.lambda_dn[i])/self.V
			
            # First coordinate (the one we had assumed to be one) of both eigenvectors
			self.CB_up[i] = 1.0/(1.0 + self.L_up[i]**2 )
			self.CB_dn[i] = 1.0/( 1.0 + self.L_dn[i]**2)
            
            # Second coordinate of both eigenvectors
            # the coordinate are raised to the power of 2 to get the 
            # probabilities
			self.CA_up[i] = self.L_up[i]**2/( 1.0 + self.L_up[i]**2)
			self.CA_dn[i] = self.L_dn[i]**2/( 1.0 + self.L_dn[i]**2)
        
        # These are the occupations numbers
		nA_up = 0.0
		nA_dn = 0.0
		nB_up = 0.0
		nB_dn = 0.0
        
		for i in range(self.length):
			if self.lambda_up[i] < 0.0:
				nA_up = nA_up + self.CA_dn[i] * self.rho[i]
				nB_up = nB_up + self.CB_dn[i] * self.rho[i]
				
			if self.lambda_dn[i] < 0.0:
				nA_dn = nA_dn + self.CA_up[i] * self.rho[i]
				nB_dn = nB_dn + self.CB_up[i] * self.rho[i]
			
				 
		return 	nA_up*self.de, nA_dn*self.de, nB_up*self.de, nB_dn*self.de	
	def normalise(self,):
		norm = 0.0
		for i in range(self.length): 
			if i == 0 or i == self.length: norm = norm + self.rho[i]*self.de
			elif i %2 == 0 : norm = norm + 2*self.rho[i]*self.de
			else: norm = norm + 4*self.rho[i]*self.de
		return norm/3.0
					 			 			
						
			
		
		
		
		
		
		
		
				
		
		
		
