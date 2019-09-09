import os
import sys
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
	from cto_eligibility_v1 import *
elif option_algo=="Top_K":
	from cto_top_k import *
elif option_algo=="Randomization":
	from cto_randomization import *

print()
print()
print()
print('Chosen configuration : ',"Distance:",option_distance,"Clustering:",option_clustering,"CTO:",option_algo)
print()
print()

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
# sum_cluster = []

file_dir = "features_rbf_batch"
n_analyst = 2
no_of_obs_trades = 2
sev_threshold = 0.5
iter_ = 0
if len(sys.argv)>1:
	no_of_obs_trades = int(sys.argv[1])
	n_analyst = int(sys.argv[2])
	sev_threshold = float(sys.argv[3])
	iter_ = int(sys.argv[4])
	# sev_threshold = 0.5
dir_len = len(os.listdir(file_dir))

sum_detect = 0
sum_cluster = []
# dir_len = len(os.listdir("features_linear/"))
pos_neg_clstr = []
elements_per_cluster = []
negatives_per_cluster = []
negatives_all = 0
## Each for loop iteration executes K-means algorithm based on cluser centres from previous iterations except first where random clusters are initiated.
for i in range(dir_len-iter_):
	# k  = 20
	# k  = 10
	sum_cstr = 0 
	sum_pos_neg = 0
	# sum_cstr = 0 
	if i == 0:
		print('Initializing cluster centers randomly')
		allocation,cluster_center,cluster_size=cto(n_analyst,no_of_obs_trades,i,option_distance,option_clustering,sev_threshold,file_dir)
		print(allocation)
		# alloc_pos_neg=get_positives_negatives(allocation)
		# for key,item in alloc_pos_neg.items():
		# 	print("accuracy based on cluster",item['neg']/no_of_obs_trades)#/cluster_size[key])
		# 	# print("accuracy based on cluster",item['neg']/k)#/cluster_size[key])
		# 	sum_detect += item['neg']
		# 	sum_cstr +=item['neg']
		# 	sum_pos_neg += item['neg'] + item['pos']
		# 	elements_per_cluster.append(item['neg'] + item['pos'])
		# 	negatives_per_cluster.append(item['neg'])
		# negatives_all += neg_labels
		# pos_neg_clstr.append(sum_pos_neg)
		# sum_cluster.append(sum_cstr)	
	else:
		print('Initializing cluster centers based on previous iteration')
		allocation,cluster_center,cluster_size=cto(n_analyst,no_of_obs_trades,i,option_distance,option_clustering,sev_threshold,file_dir,cluster_center)
		print(allocation)
		# alloc_pos_neg=get_positives_negatives(allocation)
		# for key,item in alloc_pos_neg.items():
		# 	print("accuracy based on cluster",item['neg']/no_of_obs_trades)#/cluster_size[key])
		# 	sum_detect += item['neg']
		# 	sum_cstr +=item['neg']
		# 	sum_pos_neg += item['neg'] + item['pos']
		# 	elements_per_cluster.append(item['neg'] + item['pos'])
		# 	negatives_per_cluster.append(item['neg'])
		# negatives_all += neg_labels
		# pos_neg_clstr.append(sum_pos_neg)
		# sum_cluster.append(sum_cstr)

# print("Correct Anomalies detected in each iteration : ",sum_cluster)
# print("All anomalies detected in each iteration",pos_neg_clstr)
# print("Size of each cluster detections",elements_per_cluster)
# print("malicious trades in each cluster",negatives_per_cluster)
# print("Total number of potential detections",sum(elements_per_cluster))
# negatives_per_cluster,elements_per_cluster = np.asarray(negatives_per_cluster),np.asarray(elements_per_cluster)
# print("accuracy per cluster",np.divide(negatives_per_cluster,elements_per_cluster))
# print("Total anomalies detected:", sum_detect)
# print("Percentage of all anomalies detected: ", (sum_detect*100)/negatives_all)
# print("Percentage of correct among all anomalies detected: ", (sum_detect*100)/sum(elements_per_cluster))
# print("Number of anomalies in dataset",negatives_all)
