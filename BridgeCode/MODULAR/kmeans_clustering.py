import numpy as np
from sklearn.cluster import KMeans

def get_weight(severity,traders,option):
	'''
	Get weights to be used for clustering

	Returns weights based on severity values

	Keyword arguments:
	severity --> Severity values of the features
	traders --> list of traders doing particular trade
	option --> Basic(Kmeans), Weighted(kmeans with severity as weights), NormalizedT(kmeans with normalized[based on trader] severity as weights)
	'''
	if option=='Basic':
		return None
	elif option=='Weighted':
		return severity
	elif option=='NormalizedT':
		sum_d={}
		for i in set(traders):
			sum_d[i]=0
		for i in len(traders):
			sum_d[traders[i]]+=severity[i]
		for i in len(severity):
			severity[i]/=sum_d[traders[i]]
		return severity
	else:
		return None

def do_cluster(features,no_traders,cluster_centers,severity,traders,option="Basic"):
	'''
	Do Kmeans Clustering

	Returns fitted features based on given options

	Keyword arguments:
	features --> The features we want to fit
	cluster_centers --> Cluster center the Kmeans has to start with. Can be taken from previous iteration
	severity --> Severity values of the features
	traders --> list of traders doing particular trade
	option --> Basic(Kmeans), Weighted(kmeans with severity as weights), NormalizedT(kmeans with normalized[based on trader] severity as weights)
	'''
	weights=get_weight(severity,traders,option)
	features=np.array(features)
	if not len(cluster_centers):
		kmeans=KMeans(n_clusters=no_traders,random_state=0).fit(features,sample_weight=weights)
		print("Random Cluster Center")
	else:
		kmeans=KMeans(n_clusters=no_traders,init=cluster_centers).fit(features,sample_weight=weights)
	return kmeans