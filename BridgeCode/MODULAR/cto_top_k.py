from file_read import *
from kmeans_clustering import *
from distance import *

def cto(no_analysts,threshold_k,index,option_distance,option_clustering,cluster_centers=[]):
	(trader,timestamp,features,severity,labels_gt)=read_file('features_rbf/feature_vector_'+str(index)+'.csv')
	features=update_features(features,option_distance)
	traders=list(set(trader))
	traders.sort()
	kmeans=do_cluster(features,no_analysts,cluster_centers,severity,trader,option_clustering)
	labels=kmeans.labels_
	map_analyst_labels={}
	cluster_size = []
	for i in range(no_analysts):
		map_analyst_labels[i]=[]
	for i in range(len(labels)):
		map_analyst_labels[labels[i]].append(i)
	for i in range(no_analysts):
		cluster_size.append(len(map_analyst_labels[i]))
	print(cluster_size)
	for i in map_analyst_labels:
		map_analyst_labels[i].sort(key=lambda i:severity[i],reverse=False)
		map_analyst_labels[i]=map_analyst_labels[i][:threshold_k]
	for i in map_analyst_labels:
		x=[]
		for j in map_analyst_labels[i]:
			x.append((trader[j],labels_gt[j]))
		map_analyst_labels[i]=x
	return map_analyst_labels,kmeans.cluster_centers_,cluster_size
