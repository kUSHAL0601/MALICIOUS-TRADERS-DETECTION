from file_read import *
from kmeans_clustering import *
from distance import *

from random import shuffle

def pick_random(array,size):
	array_idx=[i for i in range(len(array))]
	shuffle(array_idx)
	array_idx=array_idx[:size]
	array_ans=[]
	for i in array_idx:
		array_ans.append(array[i])
	return array_ans

def cto(no_analysts,threshold_k,index,option_distance,option_clustering,cluster_centers=[]):
	threshold_k1=int(0.7*threshold_k)
	threshold_k2=threshold_k - threshold_k1
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
		chosen=map_analyst_labels[i][:threshold_k1]
		chosen+=pick_random(map_analyst_labels[i][threshold_k1:],threshold_k2)
		map_analyst_labels[i]=chosen
	for i in map_analyst_labels:
		x=[]
		for j in map_analyst_labels[i]:
			x.append((trader[j],labels_gt[j]))
		map_analyst_labels[i]=x
	return map_analyst_labels,kmeans.cluster_centers_,cluster_size