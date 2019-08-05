from LP_CTO import LP_CTO
import numpy
from copy import deepcopy

def BRLP_CTO(reward,template_probability_distribution,E_min):
	beta_low=0
	beta_high=1
	beta_mid=(beta_low+beta_high)/2
	E_bar=0
	for i in range(len(reward)):
		E_bar+=reward[i]*template_probability_distribution[i]
#	print(beta_mid)
	temp=LP_CTO(reward,beta_mid,template_probability_distribution)
	E=temp[0]
	alpha=temp[1]
#	print(E,alpha)
	if E_bar<=E_min:
		while(abs(E-E_min)>0.01):
			if(E>E_min):
				beta_low=deepcopy(beta_mid)
			else:
				beta_high=deepcopy(beta_mid)
			beta_mid=(beta_low+beta_high)/2
			temp=LP_CTO(reward,beta_mid,template_probability_distribution)
			E=temp[0]
			alpha=temp[1]
#			print(beta_mid)
#			print(E,alpha)
	return numpy.random.choice(range(1,len(alpha)+1),p=alpha)
