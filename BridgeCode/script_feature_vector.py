import numpy as np
from sklearn.cluster import KMeans
def read_file(filename):
	with open(filename,'r') as f:
		l=f.readlines()
		lines=[]
		for i in l:
			lines.append(i.strip().split(','))
	headings=lines.pop(0)
	#print(headings)
	trader=[]
	timestamp=[]
	features=[]
	severity=[]
	labels=[]
	for i in lines:
		trader.append(i[0])
		timestamp.append(i[1])
		features.append(i[2:-2])
		severity.append(float(i[-2]))
		labels.append(float(i[-1]))
	return (trader,timestamp,features,severity,labels)

def do_cluster(features,no_traders,cluster_centers):
	features=np.array(features)
	if not len(cluster_centers):
		kmeans=KMeans(n_clusters=no_traders,random_state=0).fit(features)
		print("Random Cluster Center")
	else:
		kmeans=KMeans(n_clusters=no_traders,init=cluster_centers).fit(features)
	return kmeans

def main(no_analysts,threshold_k,cluster_centers=[]):
	(trader,timestamp,features,severity,labels_gt)=read_file('features_rbf/feature_vector_0.csv')
#	print(labels_gt)
	traders=list(set(trader))
	traders.sort()
	kmeans=do_cluster(features,no_analysts,cluster_centers)
	labels=kmeans.labels_
	map_analyst_labels={}
	for i in range(no_analysts):
		map_analyst_labels[i]=[]
	for i in range(len(labels)):
		map_analyst_labels[labels[i]].append(i)
	for i in map_analyst_labels:
		map_analyst_labels[i].sort(key=lambda i:severity[i],reverse=True)
		map_analyst_labels[i]=map_analyst_labels[i][:threshold_k]
	for i in map_analyst_labels:
		x=[]
		for j in map_analyst_labels[i]:
			x.append((trader[j],labels_gt[j]))
		map_analyst_labels[i]=x
	return map_analyst_labels,kmeans.cluster_centers_

def get_positives_negatives(allocation):
	alloc_pos_neg={}
	for i in allocation:
		alloc_pos_neg[i]={'pos':0,'neg':0}
		for j in allocation[i]:
			if j[1]>0:
				alloc_pos_neg[i]['pos']+=1
			else:
				alloc_pos_neg[i]['neg']+=1
	return alloc_pos_neg
allocation,cluster_center=main(3,5)
print(allocation)
alloc_pos_neg=get_positives_negatives(allocation)
print(alloc_pos_neg)
#print(cluster_center)
allocation,cluster_center=main(3,5,cluster_center)
print(allocation)
alloc_pos_neg=get_positives_negatives(allocation)
print(alloc_pos_neg)
#print(cluster_center)
