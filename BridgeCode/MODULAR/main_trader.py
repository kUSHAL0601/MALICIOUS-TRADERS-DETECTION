import os
from file_read import *
import numpy as np
from sklearn.cluster import KMeans


files = list(os.listdir("feature_rbf_per_secs/"))
files.sort()
#print(files)
d={}
def do_update(f,cluster_center=[]):
	(trader,timestamp,features,severity,labels_gt)=read_file("feature_rbf_per_secs/"+f)
	features=np.array(features)
	traders=list(set(trader))
	for i in traders:
		if i not in d:
			d[i]={}
			d[i]['severity']=0
			d[i]['features']=None
	for i in range(len(trader)):
		d[trader[i]]['severity']+=severity[i]
		try:
			d[trader[i]]['features']+=features[i]
		except:
			d[trader[i]]['features']=features[i]
	cumulative_features=[]
	cumulative_severity=[]
	for i in d:
		cumulative_features.append(d[i]['features'])
		cumulative_severity.append(d[i]['severity'])
	if not len(cluster_center):
		kmeans=KMeans(n_clusters=1,random_state=0).fit(cumulative_features,sample_weight=cumulative_severity)
		print("Random Cluster Center")
	else:
		kmeans=KMeans(n_clusters=1,init=cluster_center).fit(cumulative_features,sample_weight=cumulative_severity)
	return kmeans.cluster_centers_
cluster_center=[]
for f in range(len(files)):
	cluster_center=do_update(files[f],cluster_center)
	if f%5==4:
		res=[]
		for i in d:
			dist=0
			for j in range(len(list(d[i]['features']))):
				dist+=(float(d[i]['features'][j])-float(cluster_center[0][j]))**2
			res.append((d[i]['severity'],i,dist**0.5))
		res.sort(reverse=True)
		print(res)
