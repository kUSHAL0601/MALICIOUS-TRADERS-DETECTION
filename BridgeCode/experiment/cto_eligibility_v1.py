from file_read import *
from kmeans_clustering import *
from distance import *

eligibility={}
frequency={}
rank={}

def get_top(arr_cluster,vis,trader,severity):
	max_eligibility=1e9
	topTuple=0
	for i in arr_cluster:
		if i not in vis:
			new_val=(eligibility[trader[i]]+frequency[trader[i]]+1)*severity[i]
			if new_val<max_eligibility:
				max_eligibility=new_val
				topTuple=i
	eligibility[trader[topTuple]]=0.25*max_eligibility
	for i in set(trader):
		if i != trader[topTuple]:
			eligibility[i]*=0.9
	# frequency[trader[topTuple]]+=1
	return topTuple

def update_frequency(trader):
	# return
	for i in trader:
		frequency[i]+=1

#_non_normalized
# _non_norm_min
def cto(no_analysts,threshold_k,index,option_distance,option_clustering,severity_threshold,file_dir,cluster_centers=[]):
	(trader,timestamp,features,severity,labels_gt)=read_file(file_dir+'/feature_vector_'+str(index)+'.csv')
	features=update_features(features,option_distance)
	labels_gt_arr = np.asarray(labels_gt)
	# print("pos",len(labels_gt_arr[labels_gt_arr>0]))
	# print("neg",len(labels_gt_arr[labels_gt_arr<0]))
	pos_labels_gt = labels_gt.count(1)
	neg_labels_gt = labels_gt.count(-1)
	traders=list(set(trader))
	for i in traders:
		if i not in eligibility:
			eligibility[i]=0.0
			frequency[i]=0
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
	#print(cluster_size)
	inc_trades=[]
	for i in map_analyst_labels:
		map_analyst_labels[i].sort(key=lambda i:severity[i],reverse=True)
		top_labels=[]
		vis=set()
		for _ in range(threshold_k):
			x=get_top(map_analyst_labels[i],vis,trader,severity)
			print(trader[x],end=' ')
			vis.add(x)
			top_labels.append(x)
		map_analyst_labels[i]=top_labels
		inc_trades+=top_labels
		#map_analyst_labels[i]=map_analyst_labels[i][:threshold_k]
	for i in map_analyst_labels:
		x=[]
		for j in map_analyst_labels[i]:
			if (severity[j]<severity_threshold):
				x.append((trader[j],labels_gt[j],round(severity[j],2)))
		map_analyst_labels[i]=x

	# print(eligibility)
	eligibility_arr=sorted([[eligibility[i],frequency[i],i] for i in eligibility])
	for i in range(len(eligibility_arr)):
		try:
			rank[eligibility_arr[i][2]]+=i
		except:
			rank[eligibility_arr[i][2]]=i

	print("\n\n\n",sorted(eligibility_arr),sorted(list(rank.keys()),key=lambda i: rank[i]),"\n\n\n")
	update_frequency(trader)
	return map_analyst_labels,kmeans.cluster_centers_,cluster_size,neg_labels_gt