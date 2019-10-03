from file_read import *
from kmeans_clustering import *
from distance import *
#_non_normalized
def cto(no_analysts,threshold_k,index,option_distance,option_clustering,severity_threshold,file_dir,cluster_centers=[]):
	(trader,timestamp,features,severity,labels_gt)=read_file(file_dir+'feature_'+str(index)+'.csv')
	features=update_features(features,option_distance)
	labels_gt_arr = np.asarray(labels_gt)
	print("pos",len(labels_gt_arr[labels_gt_arr>0]))
	print("neg",len(labels_gt_arr[labels_gt_arr<0]))
	pos_labels_gt = labels_gt.count(1)
	neg_labels_gt = labels_gt.count(-1)
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
			if (severity[j]<severity_threshold):
				x.append((trader[j],labels_gt[j],severity[j]))
		map_analyst_labels[i]=x
	return map_analyst_labels,kmeans.cluster_centers_,cluster_size,neg_labels_gt
