from pulp import *

def LP_CTO(reward,beta,sample):
	length=len(reward)
	problem=LpProblem('LP-CTO-STEP',LpMaximize)
	probability=[]
	for i in range(length):
		probability.append(LpVariable('p'+str(i),0,1.0))
	problem+=lpDot(reward,probability)
	for i in range(length):
		problem+= probability[i]>=beta*sample[i]
	problem+= lpSum(probability)==1
	status=problem.solve()
#	print(LpStatus[status])
	ans=[]
	final_reward=0
	for i in range(length):
		ans.append(value(probability[i]))
		final_reward+=reward[i]*value(probability[i])
	return (final_reward,ans)

#EXAMPLE

#reward=[1,0,0,0,0,0,0,0,0,0]
#beta=0.2
#sample=[0.8,0.2,0,0,0,0,0,0,0,0]
#ans=LP_CTO(reward,beta,sample)
#print(ans)

# Try Changing values for beta
