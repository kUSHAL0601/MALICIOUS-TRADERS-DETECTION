from file_read import *
from kmeans_clustering import *
from distance import *

eligibility={}
frequency={}

def get_top(arr_cluster,vis,trader,severity,idx):
	max_eligibility=1e9
	topTuple=0
	for i in arr_cluster:
		if i not in vis:
			new_val=(eligibility[idx][trader[i]]+frequency[idx][trader[i]]+1)*severity[i]
			if new_val<max_eligibility:
				max_eligibility=new_val
				topTuple=i
	eligibility[idx][trader[topTuple]]=0.25*max_eligibility
	for i in set(trader):
		if i != trader[topTuple]:
			eligibility[idx][i]*=0.9
	return topTuple

def update_frequency(labels_idx,trader,idx):
	for i in labels_idx:
		frequency[idx][trader[i]]+=1


def cto(no_analysts,threshold_k,index,option_distance,option_clustering,cluster_centers=[]):
	for i in range(no_analysts):
		if i not in eligibility:
			eligibility[i]={}
			frequency[i]={}
	(trader,timestamp,features,severity,labels_gt)=read_file('features_rbf/feature_vector_'+str(index)+'.csv')
	features=update_features(features,option_distance)
	traders=list(set(trader))
	for j in range(no_analysts):
		for i in traders:
			if i not in eligibility[j]:
				eligibility[j][i]=0.0
				frequency[j][i]=0
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
	inc_trades=[]
	for i in map_analyst_labels:
		map_analyst_labels[i].sort(key=lambda i:severity[i],reverse=False)
		top_labels=[]
		vis=set()
		for _ in range(threshold_k):
			x=get_top(map_analyst_labels[i],vis,trader,severity,i)
			vis.add(x)
			top_labels.append(x)
		update_frequency(map_analyst_labels[i],trader,i)
		map_analyst_labels[i]=top_labels
		inc_trades+=top_labels
		#map_analyst_labels[i]=map_analyst_labels[i][:threshold_k]
	for i in map_analyst_labels:
		x=[]
		for j in map_analyst_labels[i]:
			x.append((trader[j],labels_gt[j]))
		map_analyst_labels[i]=x

	return map_analyst_labels,kmeans.cluster_centers_,cluster_size
