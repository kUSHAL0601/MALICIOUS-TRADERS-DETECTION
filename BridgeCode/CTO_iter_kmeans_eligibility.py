import numpy as np
from sklearn.cluster import KMeans
import os,os.path

eligibility={}

def update_eligibility(trader,severity,discount_f):
	traders=list(set(trader))
	sum_t={}
	for i in traders:
		sum_t[i]=0
		if i not in eligibility:
			eligibility[i]=0
	norm_severity={}
	for i in range(len(trader)):
		sum_t[trader[i]]+=severity[i]
		norm_severity[i]=severity[i]
	for i in range(len(trader)):
		norm_severity[i]/=sum_t[trader[i]]
	return norm_severity
	

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
## K-means clustering
def do_cluster(features,no_traders,cluster_centers):
	features=np.array(features)
	if not len(cluster_centers):
		kmeans=KMeans(n_clusters=no_traders,random_state=0).fit(features)
		print("Random Cluster Center")
	else:
		kmeans=KMeans(n_clusters=no_traders,init=cluster_centers).fit(features)
	return kmeans
## Code for cto function
def cto(no_analysts,threshold_k,i,cluster_centers=[]):
	(trader,timestamp,features,severity,labels_gt)=read_file('features_rbf/feature_vector_'+str(i)+'.csv')
#	print(labels_gt)
	traders=list(set(trader))
	traders.sort()
	#norm_severity=update_eligibility(trader,severity,)
	kmeans=do_cluster(features,no_analysts,cluster_centers)
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


sum = 0
sum_cluster = []
dir_len = len(os.listdir("features_rbf/"))
## Each for loop iteration executes K-means algorithm based on cluser centres from previous iterations except first where random clusters are initiated.
for i in range(dir_len):
	k  = 10
	sum_cstr = 0 
	if i == 0:
		allocation,cluster_center,cluster_size=cto(5,k,i)
		print(allocation)
		alloc_pos_neg=get_positives_negatives(allocation)
		for key,item in alloc_pos_neg.items():
			print("accuracy based on cluster",item['neg']/k)#/cluster_size[key])
			sum += item['neg']
			sum_cstr +=item['neg']
		sum_cluster.append(sum_cstr)	
			
	else:
		allocation,cluster_center,cluster_size=cto(5,k,i,cluster_center)
		print(allocation)
		alloc_pos_neg=get_positives_negatives(allocation)
		for key,item in alloc_pos_neg.items():
			print("accuracy based on cluster",item['neg']/k)#/cluster_size[key])
			sum += item['neg']
			sum_cstr +=item['neg']
		sum_cluster.append(sum_cstr)

print("Anomalies detected in each iteration : ",sum_cluster)
print("Total anomalies detected:", sum)
print("Percentage of all anomalies detected: ", (sum*100)/69)
