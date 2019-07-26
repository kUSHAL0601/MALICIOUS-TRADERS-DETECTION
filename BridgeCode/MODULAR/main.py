import os

option_distance="Euclidean"
options_distance={1:"Euclidean",2:"Mahalanobis"}
option_clustering="Basic"
options_clustering={1:"Basic",2:"Weighted",3:"NormalizedT"}
option_algo="Eligibility"
options_algo={1:"Top_K",2:"Eligibility",3:"Randomization"}


print("Distance Measure")
print(1,"Euclidean")
print(2,"Mahalanobis")
z=input("Select Distance Measure:")
try:
	option_distance=options_distance[int(z)]
except:
	pass
print()
print("Clustering Algos")
print(1,"Basic(Kmeans)")
print(2,"Weighted(Kmeans with severity as weights)")
print(3,"NormalizedT(Kmeans with normalized[based on trader] severity as weights)")
z=input("Select Clustering Algo:")
try:
	option_clustering=options_clustering[int(z)]
except:
	pass
print()
print("CTO Algos")
print(1,"Top K")
print(2,"Eligibility")
print(3,"Randomization")
z=input("Select CTO Algo:")
try:
	option_algo=options_algo[int(z)]
except:
	pass

if option_algo=="Eligibility":
	from cto_eligibility import *
elif option_algo=="Top_K":
	from cto_top_k import *
elif option_algo=="Randomization":
	from cto_randomization import *

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


cur_sum = 0
sum_cluster = []
dir_len = len(os.listdir("features_rbf/"))
## Each for loop iteration executes K-means algorithm based on cluser centres from previous iterations except first where random clusters are initiated.
for i in range(dir_len):
	k  = 10
	sum_cstr = 0 
	if i == 0:
		allocation,cluster_center,cluster_size=cto(5,k,i,option_distance,option_clustering)
		print(allocation)
		alloc_pos_neg=get_positives_negatives(allocation)
		for key,item in alloc_pos_neg.items():
			print("accuracy based on cluster",item['neg']/k)#/cluster_size[key])
			cur_sum += item['neg']
			sum_cstr +=item['neg']
		sum_cluster.append(sum_cstr)	
	else:
		allocation,cluster_center,cluster_size=cto(5,k,i,option_distance,option_clustering,cluster_center)
		print(allocation)
		alloc_pos_neg=get_positives_negatives(allocation)
		for key,item in alloc_pos_neg.items():
			print("accuracy based on cluster",item['neg']/k)#/cluster_size[key])
			cur_sum += item['neg']
			sum_cstr +=item['neg']
		sum_cluster.append(sum_cstr)

print("Anomalies detected in each iteration : ",sum_cluster)
print("Total anomalies detected:", cur_sum)
print("Percentage of all anomalies detected: ", (cur_sum*100)/69)
