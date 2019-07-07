import numpy as np
from sklearn.cluster import KMeans
def read_file(filename):
	with open(filename,'r') as f:
		l=f.readlines()
		lines=[]
		for i in l:
			lines.append(i.strip().split(','))
	headings=lines.pop(0)
	trader=[]
	timestamp=[]
	features=[]
	severity=[]
	for i in lines:
		trader.append(i[0])
		timestamp.append(i[1])
		features.append(i[2:-1])
		severity.append(float(i[-1]))
	return (trader,timestamp,features,severity)

def do_cluster(features,no_traders):
	features=np.array(features)
	kmeans=KMeans(n_clusters=no_traders,random_state=0).fit(features)
	return list(kmeans.labels_)

def main(no_analysts,threshold_k):
	(trader,timestamp,features,severity)=read_file('feature_vector.csv')
	traders=list(set(trader))
	traders.sort()
	labels=do_cluster(features,no_analysts)
#	match={}
#	for i in range(len(trader)):
#		for j in range(len(traders)):
#			match[trader[i]+','+str(j)]=0
#	for i in range(len(trader)):
#		match[trader[i]+','+str(labels[i])]+=1
#	a1=list(match.keys())
#	a1.sort(key=lambda i:match[i],reverse=True)
#	#print(a1)
#	map_cluster_trader={}
#	vis=[]
#	for i in a1:
#		x,y=i.split(',')
#		y=int(y)
#		if y not in map_cluster_trader and x not in vis:
#			map_cluster_trader[y]=x
#			vis.append(x)
#	print(map_cluster_trader)
#	allocation={}
#	for i in traders:
#		allocation[i]=[]
#	for i in range(len(labels)):
#		allocation[map_cluster_trader[labels[i]]].append(i)
#	print(allocation)
#	return allocation
	map_analyst_labels={}
	for i in range(no_analysts):
		map_analyst_labels[i]=[]
	for i in range(len(labels)):
		map_analyst_labels[labels[i]].append(i)
	for i in map_analyst_labels:
		map_analyst_labels[i].sort(key=lambda i:severity[i],reverse=True)
		map_analyst_labels[i]=map_analyst_labels[i][:threshold_k]
	return map_analyst_labels
print(main(3,5))
