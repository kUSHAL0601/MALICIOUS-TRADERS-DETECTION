from file_read import *
from kmeans_clustering import *
from distance import *

eligibility={}
frequency={}
rank={}

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

#_non_normalized
# _non_norm_min
def cto(no_analysts,threshold_k,index,option_distance,option_clustering,severity_threshold,file_dir,cluster_centers=[]):
	for i in range(no_analysts):
		if i not in eligibility:
			eligibility[i]={}
			frequency[i]={}
	(trader,timestamp,features,severity)=read_file(file_dir+'/feature_'+str(index)+'.csv')
	features=update_features(features,option_distance)
	# labels_gt_arr = np.asarray(labels_gt)
	# print("pos",len(labels_gt_arr[labels_gt_arr>0]))
	# print("neg",len(labels_gt_arr[labels_gt_arr<0]))
	# pos_labels_gt = labels_gt.count(1)
	# neg_labels_gt = labels_gt.count(-1)
	traders=list(set(trader))
	for j in range(no_analysts):
		for i in traders:
			if i not in eligibility[j]:
				eligibility[j][i]=0.0
				frequency[j][i]=0
	traders.sort()
	kmeans=do_cluster(features,no_analysts,cluster_centers,severity,trader,option_clustering)
	print('Eligibility based allocation')
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
			x=get_top(map_analyst_labels[i],vis,trader,severity,i)
			# print(trader[x],end=' ')
			vis.add(x)
			top_labels.append(x)
		update_frequency(map_analyst_labels[i],trader,i)
		map_analyst_labels[i]=list(set(top_labels))
		inc_trades+=list(set(top_labels))
		#map_analyst_labels[i]=map_analyst_labels[i][:threshold_k]
	for i in map_analyst_labels:
		x=[]
		for j in map_analyst_labels[i]:
			if (severity[j]<severity_threshold):
				x.append((trader[j],round(severity[j],2)))
		map_analyst_labels[i]=x
	for j in eligibility:
		eligibility_arr=sorted([[eligibility[j][i],frequency[j][i],i] for i in eligibility[j]])
		for i in range(len(eligibility_arr)):
			try:
				rank[eligibility_arr[i][2]]+=i
			except:
				rank[eligibility_arr[i][2]]=i
	
	print("\n\n\n",sorted(list(rank.keys()),key=lambda i: rank[i]),"\n\n\n")
	return map_analyst_labels,kmeans.cluster_centers_,cluster_size
