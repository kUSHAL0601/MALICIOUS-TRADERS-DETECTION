import matplotlib.pyplot as plt
from statistics import mean,stdev,median
import csv
import math
import numpy as np
# from pyod.models.ocsvm import OCSVM
# from pyod.models.pca import PCA 
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.svm import OneClassSVM as oc_svm 



set_folder_per_batch = 'features_rbf_batch'
set_threshold = 1.47256
batch_size = 100
## create a folder named features_rbf_per_sec if generate_per_sec = True 
generate_per_sec_data = False

if generate_per_sec_data:
    set_folder_per_sec = 'features_rbf_per_sec/feature_'

# print(malicious_keys)
print("Loading classifier ..")
filename = 'classifier.sav'
clf1 = pickle.load(open(filename, 'rb')) 

traders=[]
trader_list = []
print("Reading data ..")
with open('message.csv', 'r') as f:
    reader = list(csv.reader(f))
    trader_timestamp_dict={}
    for row in range(1,len(reader)):
        entry=reader[row]
        time_stamp=entry[1][:-7]
        entry_type=entry[2]
        order_id=entry[3]
        price=float(entry[4])
        volume=float(entry[5])
        direction=entry[6]
        trader_id=entry[7]
        stock_id=entry[8]
        order_level=entry[9]
        matched_order_trader_id=entry[10]
        match_price=entry[11]
        match_volume=entry[12]
        match_timestamp=entry[13]
        traders.append(trader_id)

        # print(time_stamp,direction,trader_id)
        if (time_stamp,trader_id) not in trader_timestamp_dict:
            trader_timestamp_dict[(time_stamp,trader_id)]={}
            trader_timestamp_dict[(time_stamp,trader_id)]['buying']={}
            trader_timestamp_dict[(time_stamp,trader_id)]['buying']['price']=[]
            trader_timestamp_dict[(time_stamp,trader_id)]['buying']['volume']=[]
            trader_timestamp_dict[(time_stamp,trader_id)]['selling']={}
            trader_timestamp_dict[(time_stamp,trader_id)]['selling']['price']=[]
            trader_timestamp_dict[(time_stamp,trader_id)]['selling']['volume']=[]

        if int(direction)==1 and int(entry_type) == 1:
            trader_timestamp_dict[(time_stamp,trader_id)]['buying']['price'].append(price)
            trader_timestamp_dict[(time_stamp,trader_id)]['buying']['volume'].append(volume)
            trader_list.append([trader_id,price,volume])
        elif int(direction)==-1 and int(entry_type) == 1:
            trader_timestamp_dict[(time_stamp,trader_id)]['selling']['price'].append(price)
            trader_timestamp_dict[(time_stamp,trader_id)]['selling']['volume'].append(volume)
            trader_list.append([trader_id,price*-1,volume])    
    # print(trader_timestamp_dict)
# traders=list(set(traders))
# print(traders)
keys=list(trader_timestamp_dict.keys())
keys.sort()
# trader_arr = trader_list
# clf = PCA()
user_order=[]



## Standardize data




print("Preparing Data ...")
# trader_list = [v for v in trader_timestamp_dict.values()]
# for key, value in trader_timestamp_dict.iteritems():
#     temp = [key,value]
#     trader_list.append(temp)
trader_arr = np.asarray(trader_list)
# print(len(traders))
# print(len(set(trader_arr[:,0])))
# print(len(keys))
# malicious_complete_data = np.zeros((len(malicious_keys),16))
# normal_complete_data = np.zeros((len(traders)-len(malicious_keys),16))
malicious_complete_data = []
normal_complete_data = []

malicious_labels = []
normal_labels = []
all_labels = []
data_a1_buy_stddev=[]
malicious_data_a1_buy_stddev=[]

a1_sum_buy=[]
a1_sum_sell=[]
a1_timestamps=[]
a1_min_buy=[]
a1_max_buy=[]
a1_min_sell=[]
a1_max_sell=[]
a1_mean_sell=[]
a1_mean_buy=[]
a1_median_sell=[]
a1_median_buy=[]
a1_timestamps_buy=[]
a1_timestamps_sell=[]
a1_timestamps_buy_stddev=[]
a1_timestamps_sell_stddev=[]
a1_buy_stddev = []
a1_sell_stddev = []

malicious_a1_sum_buy=[]
malicious_a1_sum_sell=[]
malicious_a1_timestamps=[]
malicious_a1_min_buy=[]
malicious_a1_max_buy=[]
malicious_a1_min_sell=[]
malicious_a1_max_sell=[]
malicious_a1_mean_sell=[]
malicious_a1_mean_buy=[]
malicious_a1_median_sell=[]
malicious_a1_median_buy=[]
malicious_a1_timestamps_buy=[]
malicious_a1_timestamps_sell=[]
malicious_a1_timestamps_buy_stddev=[]
malicious_a1_timestamps_sell_stddev=[]
malicious_a1_buy_stddev = []
malicious_a1_sell_stddev = []

a1_vol_sum_buy=[]
a1_vol_sum_sell=[]
a1_vol_timestamps=[]
a1_vol_min_buy=[]
a1_vol_max_buy=[]
a1_vol_min_sell=[]
a1_vol_max_sell=[]
a1_vol_mean_sell=[]
a1_vol_mean_buy=[]
a1_vol_median_sell=[]
a1_vol_median_buy=[]
a1_vol_timestamps_buy=[]
a1_vol_timestamps_sell=[]
a1_vol_stddev_buy=[]
a1_vol_timestamps_buy_stddev=[]
a1_vol_timestamps_sell_stddev=[]
a1_vol_buy_stddev = []
a1_vol_sell_stddev = []

malicious_a1_vol_sum_buy=[]
malicious_a1_vol_sum_sell=[]
malicious_a1_vol_timestamps=[]
malicious_a1_vol_min_buy=[]
malicious_a1_vol_max_buy=[]
malicious_a1_vol_min_sell=[]
malicious_a1_vol_max_sell=[]
malicious_a1_vol_mean_sell=[]
malicious_a1_vol_mean_buy=[]
malicious_a1_vol_median_sell=[]
malicious_a1_vol_median_buy=[]
malicious_a1_vol_timestamps_buy=[]
malicious_a1_vol_timestamps_sell=[]
malicious_a1_vol_stddev_buy=[]
malicious_a1_vol_timestamps_buy_stddev=[]
malicious_a1_vol_timestamps_sell_stddev=[]
malicious_a1_vol_buy_stddev = []
malicious_a1_vol_sell_stddev = []
non_malicious = []
mal_traders = []
non_malicious_timestamps = []
malicious_timestamps = []
for i in keys:
    # print(i)
# if i not in malicious_keys:
    non_malicious_timestamps.append(i[0])
    non_malicious.append(i[1])
    a1_sum_buy = sum(trader_timestamp_dict[i]['buying']['price'])
    a1_sum_sell = sum(trader_timestamp_dict[i]['selling']['price'])

    if len((trader_timestamp_dict[i]['buying']['price']))>0:
        # a1_timestamps_buy.append(i[0])
        a1_mean_buy = mean(trader_timestamp_dict[i]['buying']['price'])
        a1_median_buy = median(trader_timestamp_dict[i]['buying']['price'])
        a1_min_buy = min(trader_timestamp_dict[i]['buying']['price'])
        a1_max_buy = max(trader_timestamp_dict[i]['buying']['price'])
    elif len((trader_timestamp_dict[i]['buying']['price']))==0:
        # a1_timestamps_buy.append(i[0])
        a1_mean_buy = 0
        a1_median_buy = 0
        a1_min_buy = 0
        a1_max_buy = 0
    if  len((trader_timestamp_dict[i]['selling']['price']))>0:
        # a1_timestamps_sell.append(i[0])
        a1_mean_sell = mean(trader_timestamp_dict[i]['selling']['price'])
        a1_median_sell = median(trader_timestamp_dict[i]['selling']['price'])
        a1_min_sell = min(trader_timestamp_dict[i]['selling']['price'])
        a1_max_sell = max(trader_timestamp_dict[i]['selling']['price'])  
    elif  len((trader_timestamp_dict[i]['selling']['price']))==0:
        # a1_timestamps_sell.append(i[0])
        a1_mean_sell = 0
        a1_median_sell = 0
        a1_min_sell = 0
        a1_max_sell = 0  
    if  len((trader_timestamp_dict[i]['buying']['price']))>1:
        # a1_timestamps_buy_stddev.append(i[0])
        a1_buy_stddev = stdev(trader_timestamp_dict[i]['buying']['price'])
    elif len((trader_timestamp_dict[i]['buying']['price']))>=0:
        a1_buy_stddev = 0
    if  len((trader_timestamp_dict[i]['selling']['price']))>1:
        # a1_timestamps_sell_stddev.append(i[0])
        a1_sell_stddev = stdev(trader_timestamp_dict[i]['selling']['price'])
    elif len((trader_timestamp_dict[i]['buying']['price']))>=0:
        a1_sell_stddev = 0

    a1_vol_sum_buy = sum(trader_timestamp_dict[i]['buying']['volume'])
    a1_vol_sum_sell = sum(trader_timestamp_dict[i]['selling']['volume'])
    if  len((trader_timestamp_dict[i]['buying']['volume']))>0:
        # a1_vol_timestamps_buy.append(i[0])
        a1_vol_mean_buy = mean(trader_timestamp_dict[i]['buying']['volume'])
        a1_vol_median_buy = median(trader_timestamp_dict[i]['buying']['volume'])
        a1_vol_min_buy = min(trader_timestamp_dict[i]['buying']['volume'])
        a1_vol_max_buy = max(trader_timestamp_dict[i]['buying']['volume'])
    elif  len((trader_timestamp_dict[i]['buying']['volume']))==0:
        # a1_vol_timestamps_buy.append(i[0])
        a1_vol_mean_buy = 0
        a1_vol_median_buy = 0
        a1_vol_min_buy = 0
        a1_vol_max_buy = 0
            
    if  len((trader_timestamp_dict[i]['selling']['volume']))>0:
        # a1_vol_timestamps_sell.append(i[0])
        a1_vol_mean_sell = mean(trader_timestamp_dict[i]['selling']['volume'])
        a1_vol_median_sell = median(trader_timestamp_dict[i]['selling']['volume'])
        a1_vol_min_sell = min(trader_timestamp_dict[i]['selling']['volume'])
        a1_vol_max_sell = max(trader_timestamp_dict[i]['selling']['volume'])  
    elif  len((trader_timestamp_dict[i]['selling']['volume']))==0:
        # a1_vol_timestamps_sell.append(i[0])
        a1_vol_mean_sell = 0
        a1_vol_median_sell = 0
        a1_vol_min_sell = 0
        a1_vol_max_sell = 0 
    if  len((trader_timestamp_dict[i]['buying']['volume']))>1:
        # a1_vol_timestamps_buy_stddev.append(i[0])
        a1_vol_buy_stddev = stdev(trader_timestamp_dict[i]['buying']['volume'])
    elif  len((trader_timestamp_dict[i]['buying']['volume']))>=0:
        # a1_vol_timestamps_buy_stddev.append(i[0])
        a1_vol_buy_stddev = 0
    if  len((trader_timestamp_dict[i]['selling']['volume']))>1:
        # a1_vol_timestamps_sell_stddev.append(i[0])
        a1_vol_sell_stddev = stdev(trader_timestamp_dict[i]['selling']['volume'])
    if  len((trader_timestamp_dict[i]['selling']['volume']))>=0:
        # a1_vol_timestamps_sell_stddev.append(i[0])
        a1_vol_sell_stddev = 0
    normal_complete_data.append([a1_sum_buy,a1_mean_buy,a1_median_buy,a1_min_buy,a1_max_buy,a1_vol_sum_buy,a1_vol_mean_buy,a1_vol_median_buy,a1_vol_min_buy,a1_vol_max_buy, a1_sum_sell,a1_mean_sell,a1_median_sell,a1_min_sell,a1_max_sell,a1_vol_sum_sell,a1_vol_mean_sell,a1_vol_median_sell,a1_vol_min_sell,a1_vol_max_sell])
    normal_labels.append(1)
    # all_labels.append(0)

normal_complete_data_arr = np.asarray(normal_complete_data)
# malicious_complete_data.pop(0)
# malicious_complete_data_arr = np.asarray(malicious_complete_data)            
# print(normal_complete_data_arr.shape)
# print(malicious_keys)
mal_trader = {}

# print(len(malicious_complete_data))
# print(mal_trader)
# feat_select = np.zeros(normal_complete_data_arr.shape[1])
# for value in feature_selected:
#     np.delete(normal_complete_data_arr,feature_selected[value],axis =1)
#     np.delete(malicious_complete_data_arr,feature_selected[value],axis =1)

## new features difference of buy, mean, sell,mean 
(shape1,shape2) = normal_complete_data_arr.shape[0],normal_complete_data_arr.shape[1]
diff_normal = np.zeros((shape1,shape2))
# diff_normal = 0
for i in range(1,len(normal_complete_data_arr)):
    diff_normal[i] = normal_complete_data_arr[i] - normal_complete_data_arr[i-1] 

normal_complete_data_arr = np.hstack((normal_complete_data_arr,diff_normal))
test_all_data = normal_complete_data_arr

filename = 'train_mean_std.txt'
train_normal_arr_params = pickle.load(open(filename, 'rb')) 
train_normal_arr_mean = train_normal_arr_params[0]
train_normal_arr_std = train_normal_arr_params[1]


test_all_data_stnd = test_all_data - train_normal_arr_mean
# negatives_mat = negatives_mat - negatives_mat.mean(axis =0)
# all_data_mat = all_data_mat - all_data_mat.mean(axis=0)

test_all_data_stnd = test_all_data_stnd/(train_normal_arr_std+1)
## Generate labels


eta = 1
def sigmoid(x):
  # return 1 / (1 + math.exp(-1*eta*x))
  return 1 / (1 + math.exp(eta*x))
import csv
## OCSVM
# clf1 = OCSVM(kernel = 'rbf',gamma = 1,nu = 0.4)

#Usable
# print(predicted)
# print(clf1.score_samples(test_all_data_stnd))


# score1 = clf1.score_samples(train_normal_arr_stnd)
score = clf1.score_samples(test_all_data_stnd)
# print(score1.reshape(len(score1),1).shape)
# score = np.vstack((score1.reshape(len(score1),1),score2.reshape(len(score2),1)))

# score = clf1.score_samples(test_all_data_stnd)
severity = []
for val in score:
    # print (np.round((1.8 - val)*10**4,2))
    severity.append(sigmoid(np.round((set_threshold - val),5)))
severity_cto = []
# print(max(severity))
# from collections import Counter
traders_mal = []
# for i in range(len(pred_indices_mal)):
#     traders_mal.append(test_keys[pred_indices_mal[i]])
#     severity_cto.append(severity[pred_indices_mal[i]])
# result = Counter(traders_mal)
# [item for items, c in Counter(traders_mal).most_common() 
                                      # for item in [items] * c]
# print(severity)
# severity_cto = np.asarray(severity_cto)
# severity_cto = severity_cto/max(severity_cto)
severity = np.asarray(severity)
# print(severity)

timestamp_all_list = non_malicious_timestamps 
traders_stack = np.asarray(non_malicious).reshape(len(non_malicious),1)
timestamp_stack = np.asarray(non_malicious_timestamps).reshape(len(non_malicious_timestamps),1)
all_trader_timestamp = np.hstack((traders_stack,timestamp_stack))
# all_trader_tstmp_labels = np.hstack((all_trader_timestamp,np.asarray(all_labels).reshape(len(all_labels),1)))

# print(all_feature_vector.shape)

feature_write = np.hstack((all_trader_timestamp,test_all_data_stnd,severity.reshape(len(severity),1)))

# print("all",all_labels.count(1))
# print("mal",all_labels.count(-1))


head = 'trader', 'timestamp', 'a1_sum_buy','a1_mean_buy','a1_median_buy','a1_min_buy','a1_max_buy','a1_vol_sum_buy','a1_vol_mean_buy','a1_vol_median_buy','a1_vol_min_buy','a1_vol_max_buy',' a1_sum_sell','a1_mean_sell','a1_median_sell','a1_min_sell','a1_max_sell','a1_vol_sum_sell','a1_vol_mean_sell','a1_vol_median_sell','a1_vol_min_sell','a1_vol_max_sell','diff_a1_sum_buy','diff_a1_mean_buy','diff_a1_median_buy','diff_a1_min_buy','diff_a1_max_buy','diff_a1_vol_sum_buy','diff_a1_vol_mean_buy','diff_a1_vol_median_buy','diff_a1_vol_min_buy','diff_a1_vol_max_buy',' diff_a1_sum_sell','diff_a1_mean_sell','diff_a1_median_sell','diff_a1_min_sell','diff_a1_max_sell','diff_a1_vol_sum_sell','diff_a1_vol_mean_sell','diff_a1_vol_median_sell','diff_a1_vol_min_sell','diff_a1_vol_max_sell','severity','labels'
# print(len(head))
# print(feature_write.dtype)
feature_write.view('U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32').sort(order=['f1'],axis=0)

all_feature_dict = {}
# timestamp_stack = timestamp_stack.tolist()
# print(timestamp_stack[0])
if generate_per_sec_data :
    timestamp_set_list = list(set(timestamp_all_list))
    timestamp_set_list.sort()
    for key in timestamp_set_list:
        all_feature_dict[(key)] = {}
    print("Preparing batches per sec")
    for key in timestamp_set_list:
        # print(key)
        count = 0
        for i in range(len(feature_write)):
            
            if (key == feature_write[i][1]):
                # print("key, feat_1",key,feature_write[i][1])
                count +=1
                all_feature_dict[(key)][count] = feature_write[i]
    #usable
    for val in timestamp_set_list: 

        with open(set_folder_per_sec+val+'.csv','w') as f:
            
            csv_writer = csv.writer(f)
            csv_writer.writerow(['trader', 'timestamp', 'a1_sum_buy','a1_mean_buy','a1_median_buy','a1_min_buy','a1_max_buy','a1_vol_sum_buy','a1_vol_mean_buy','a1_vol_median_buy','a1_vol_min_buy','a1_vol_max_buy',' a1_sum_sell','a1_mean_sell','a1_median_sell','a1_min_sell','a1_max_sell','a1_vol_sum_sell','a1_vol_mean_sell','a1_vol_median_sell','a1_vol_min_sell','a1_vol_max_sell','diff_a1_sum_buy','diff_a1_mean_buy','diff_a1_median_buy','diff_a1_min_buy','diff_a1_max_buy','diff_a1_vol_sum_buy','diff_a1_vol_mean_buy','diff_a1_vol_median_buy','diff_a1_vol_min_buy','diff_a1_vol_max_buy',' diff_a1_sum_sell','diff_a1_mean_sell','diff_a1_median_sell','diff_a1_min_sell','diff_a1_max_sell','diff_a1_vol_sum_sell','diff_a1_vol_mean_sell','diff_a1_vol_median_sell','diff_a1_vol_min_sell','diff_a1_vol_max_sell','severity','labels'])
            for keys,item in all_feature_dict[(val)].items():
                csv_writer.writerow(item)


# print(all_feature_dict[(timestamp_set_list[3])])
# feature_write = np.vstack((head,feature_write))
# print(feature_write[0:2])
# np.sort(feature_write,order = 'timestamp')
# with open('feature_vector_rbf_all.csv','w') as f:
#     csv_writer = csv.writer(f)
#     csv_writer.writerow(['trader', 'timestamp', 'a1_sum_buy','a1_mean_buy','a1_median_buy','a1_min_buy','a1_max_buy','a1_vol_sum_buy','a1_vol_mean_buy','a1_vol_median_buy','a1_vol_min_buy','a1_vol_max_buy',' a1_sum_sell','a1_mean_sell','a1_median_sell','a1_min_sell','a1_max_sell','a1_vol_sum_sell','a1_vol_mean_sell','a1_vol_median_sell','a1_vol_min_sell','a1_vol_max_sell','diff_a1_sum_buy','diff_a1_mean_buy','diff_a1_median_buy','diff_a1_min_buy','diff_a1_max_buy','diff_a1_vol_sum_buy','diff_a1_vol_mean_buy','diff_a1_vol_median_buy','diff_a1_vol_min_buy','diff_a1_vol_max_buy',' diff_a1_sum_sell','diff_a1_mean_sell','diff_a1_median_sell','diff_a1_min_sell','diff_a1_max_sell','diff_a1_vol_sum_sell','diff_a1_vol_mean_sell','diff_a1_vol_median_sell','diff_a1_vol_min_sell','diff_a1_vol_max_sell','severity','labels'])
#     csv_writer.writerows(feature_write)

#usable
print("Preparing batches of size"+str(batch_size)+" ...")
for i in range(int(len(feature_write)/batch_size)+1):
    with open(set_folder_per_batch+"/feature_"+str(i)+'.csv','w',) as f:
    # with open('your_file.txt', 'w') as f:
        if i != int(len(feature_write)/batch_size): 
            csv_writer = csv.writer(f)
            # f.write(str(count))
            # csv_writer.writerow([str(len(cto_input))])
            # csv_writer.writerow(['trader', 'timestamp', 'a1_sum_buy','a1_mean_buy','a1_median_buy','a1_min_buy','a1_max_buy','a1_vol_sum_buy','a1_vol_mean_buy','a1_vol_median_buy','a1_vol_min_buy','a1_vol_max_buy',' a1_sum_sell','a1_mean_sell','a1_median_sell','a1_min_sell','a1_max_sell','a1_vol_sum_sell','a1_vol_mean_sell','a1_vol_median_sell','a1_vol_min_sell','a1_vol_max_sell','diff_a1_sum_buy','diff_a1_mean_buy','diff_a1_median_buy','diff_a1_min_buy','diff_a1_max_buy','diff_a1_vol_sum_buy','diff_a1_vol_mean_buy','diff_a1_vol_median_buy','diff_a1_vol_min_buy','diff_a1_vol_max_buy',' diff_a1_sum_sell','diff_a1_mean_sell','diff_a1_median_sell','diff_a1_min_sell','diff_a1_max_sell','diff_a1_vol_sum_sell','diff_a1_vol_mean_sell','diff_a1_vol_median_sell','diff_a1_vol_min_sell','diff_a1_vol_max_sell','severity','labels'])
            csv_writer.writerow(['trader', 'timestamp', 'a1_sum_buy','a1_min_buy','a1_max_buy',' a1_sum_sell','a1_min_sell','a1_max_sell','diff_a1_sum_buy','diff_a1_min_buy','diff_a1_max_buy',' diff_a1_sum_sell','diff_a1_min_sell','diff_a1_max_sell','severity','labels'])
            csv_writer.writerows(feature_write[batch_size*i:batch_size*(i+1)])
        elif i == int(len(feature_write)/batch_size):
            csv_writer = csv.writer(f)
            # f.write(str(count))
            # csv_writer.writerow([str(len(cto_input))])
            csv_writer.writerow(['trader', 'timestamp', 'a1_sum_buy','a1_min_buy','a1_max_buy',' a1_sum_sell','a1_min_sell','a1_max_sell','diff_a1_sum_buy','diff_a1_min_buy','diff_a1_max_buy',' diff_a1_sum_sell','diff_a1_min_sell','diff_a1_max_sell','severity','labels'])
            csv_writer.writerows(feature_write[batch_size*i:len(feature_write)])

# feature_write_per_min = {}


# with open('feature_vector_rbf_min.csv','w',) as f:
#     csv_writer = csv.writer(f)
#         # f.write(str(count))
#         # csv_writer.writerow([str(len(cto_input))])
#     csv_writer.writerow(['trader', 'timestamp', 'a1_sum_buy','a1_mean_buy','a1_median_buy','a1_min_buy','a1_max_buy','a1_vol_sum_buy','a1_vol_mean_buy','a1_vol_median_buy','a1_vol_min_buy','a1_vol_max_buy',' a1_sum_sell','a1_mean_sell','a1_median_sell','a1_min_sell','a1_max_sell','a1_vol_sum_sell','a1_vol_mean_sell','a1_vol_median_sell','a1_vol_min_sell','a1_vol_max_sell','diff_a1_sum_buy','diff_a1_mean_buy','diff_a1_median_buy','diff_a1_min_buy','diff_a1_max_buy','diff_a1_vol_sum_buy','diff_a1_vol_mean_buy','diff_a1_vol_median_buy','diff_a1_vol_min_buy','diff_a1_vol_max_buy',' diff_a1_sum_sell','diff_a1_mean_sell','diff_a1_median_sell','diff_a1_min_sell','diff_a1_max_sell','diff_a1_vol_sum_sell','diff_a1_vol_mean_sell','diff_a1_vol_median_sell','diff_a1_vol_min_sell','diff_a1_vol_max_sell','severity','labels'])
#     csv_writer.writerows(feature_write)
# with open('myfile.txt', 'w') as f:
    # csv_writer = csv.writer(f)
    # csv_writer.writerows(cto_input)
# with open('feature_vector.csv','r') as f:
#     reader = list(csv.reader(f))
#     reader = np.asarray(reader)
#     np.sort(reader,order = 'timestamp')
#     print(reader[2])
# print(predicted)
# print(test_labels)
# for index in indices:
#     print(malicious_keys[int(index-0.2*len(normal_complete_data_arr))])
# print(len(test_all_data_stnd))
# print(len(indices)/len(test_all_data_stnd))

# print(max(roc))
# fig1,ax1 = plt.subplots(1,1)
# print(len(roc))
# ax1.scatter(np.arange(len(recall_list)),recall_list)
# # ax2.scatter(np.arange(len(clf1.decision_scores_)),clf1.decision_scores_)
# plt.show()
# print(indices_positive/len(test_labels))