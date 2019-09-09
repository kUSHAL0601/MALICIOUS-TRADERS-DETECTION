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

# set_folder = 'features_rbf_per_sec/feature_'

def print_accuracy(train_arr,test_arr,trader_id):
    if len(train_arr)==0 or len(test_arr)==0:
        return
    for i in range(len(train_arr)):
        l1=len(train_arr[i])
        l2=len(test_arr[i])
        if l1==0 or l2==0:
            continue
        train_data=np.array([train_arr[i]]).T
        test_data=np.array([test_arr[i]]).T
        # clf=OCSVM(kernel ='rbf',gamma = 0.5)
        print(len(train_arr))
        clf = PCA(n_components =15)
        clf.fit(train_arr)
        y_pred=clf.predict(train_arr)
        print("TRAINING ACCURACY for TRADER",trader_id,":",100 - (sum(y_pred)*100/l1))
        y_pred=clf.predict(test_data)
        print("TESTING ACCURACY: ",sum(y_pred)*100/l2)

malicious_keys=[]
with open('attack.csv', 'r') as f1:
    reader1 = list(csv.reader(f1))
    reader1.pop(0)
    for i in reader1:
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
                if order_id==i[1]:
                    malicious_keys.append((time_stamp,trader_id))


# print(malicious_keys)


def cumulative_sum(arr):
    cum_sum=[]
    cur_sum=0
    for i in arr:
        cur_sum+=i
        cum_sum.append(cur_sum)
    return cum_sum

def moving_average(arr,window_size):
    l=len(arr)
    ans=[]
    for i in range(l - window_size + 1):
        ans.append(mean(arr[i:i+window_size]))
    return ans


def draw(x,y,strx,stry,tit):
        return
        plt.plot(x,y)
        plt.title(tit)
        plt.xlabel(strx)
        plt.ylabel(stry)
        plt.xticks(rotation=90)
        #plt.savefig(tit.replace(' ','') + '.png')
        # plt.show()

def draw2(y,x,strx,stry,tit):
        return
        plt.plot(x,y)
        plt.title(tit)
        plt.xlabel(strx)
        plt.ylabel(stry)
        plt.xticks(rotation=90)
        #plt.savefig(tit.replace(' ','') + '.png')

def draw_hist(x,strx,stry,tit):
        return
        plt.title(tit)
        plt.xlabel(strx)
        plt.ylabel(stry)
        plt.xticks(rotation=90)
        plt.hist(x,bins=20)
        #plt.savefig(tit.replace(' ','') + '_hist.png')

traders=[]
trader_list = []
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
data_a1_cumulative_buy_stddev=[]
data_a1_cumulative_mean_buy=[]
data_a1_cumulative_mean_sell=[]
data_a1_cumulative_sell_stddev=[]
data_a1_cumulative_sum_buy=[]
data_a1_cumulative_sum_sell=[]
data_a1_cumulative_vol_buy_stddev=[]
data_a1_cumulative_vol_mean_buy=[]
data_a1_cumulative_vol_mean_sell=[]
data_a1_cumulative_vol_sell_stddev=[]
data_a1_cumulative_vol_sum_buy=[]
data_a1_cumulative_vol_sum_sell=[]
data_a1_max_buy=[]
data_a1_max_sell=[]
data_a1_mean_buy=[]
data_a1_mean_sell=[]
data_a1_median_buy=[]
data_a1_median_sell=[]
data_a1_min_buy=[]
data_a1_min_sell=[]
data_a1_mov_avg_buy_stddev=[]
data_a1_mov_avg_cumulative_buy_stddev=[]
data_a1_mov_avg_cumulative_max_buy=[]
data_a1_mov_avg_cumulative_max_sell=[]
data_a1_mov_avg_cumulative_mean_buy=[]
data_a1_mov_avg_cumulative_mean_sell=[]
data_a1_mov_avg_cumulative_median_buy=[]
data_a1_mov_avg_cumulative_median_sell=[]
data_a1_mov_avg_cumulative_min_buy=[]
data_a1_mov_avg_cumulative_vol_min_sell=[]
data_a1_mov_avg_cumulative_sell_stddev=[]
data_a1_mov_avg_cumulative_vol_sum_buy=[]
data_a1_mov_avg_cumulative_vol_sum_sell=[]
data_a1_mov_avg_max_buy=[]
data_a1_mov_avg_max_sell=[]
data_a1_mov_avg_mean_buy=[]
data_a1_mov_avg_mean_sell=[]
data_a1_mov_avg_median_buy=[]
data_a1_mov_avg_median_sell=[]
data_a1_mov_avg_min_buy=[]
data_a1_mov_avg_min_sell=[]
data_a1_mov_avg_sell_stddev=[]
data_a1_mov_avg_sum_buy=[]
data_a1_mov_avg_sum_sell=[]
data_a1_mov_avg_vol_buy_stddev=[]
data_a1_mov_avg_vol_max_buy=[]
data_a1_mov_avg_vol_max_sell=[]
data_a1_mov_avg_vol_mean_buy=[]
data_a1_mov_avg_vol_mean_sell=[]
data_a1_mov_avg_cumulative_median_buy=[]
data_a1_mov_avg_cumulative_vol_median_sell=[]
data_a1_mov_avg_cumulative_vol_min_buy=[]
data_sum_buy=[]
data_sum_sell=[]
data_a1_vol_sum_buy=[]
data_a1_sell_stddev=[]
data_a1_vol_buy_stddev=[]
data_a1_vol_max_buy=[]
data_a1_vol_max_sell=[]
data_a1_vol_sum_sell=[]
data_a1_vol_sum_buy=[]
data_a1_vol_sell_stddev=[]
data_a1_vol_min_sell=[]
data_a1_vol_min_buy=[]
data_a1_vol_median_sell=[]
data_a1_vol_median_buy=[]
data_a1_vol_mean_sell=[]
data_a1_vol_mean_buy=[]
data_a1_vol_max_sell=[]
data_a1_vol_max_buy=[]
data_a1_mov_avg_vol_sum_sell=[]
data_a1_mov_avg_vol_sum_buy=[]
data_a1_mov_avg_vol_sell_stddev=[]
data_a1_mov_avg_vol_min_sell=[]
data_a1_mov_avg_vol_min_buy=[]
data_a1_mov_avg_vol_min_sell=[]
data_a1_mov_avg_vol_median_buy=[]
data_a1_mov_avg_vol_median_sell=[]
data_a1_mov_avg_cumulative_sum_buy=[]
data_a1_mov_avg_cumulative_sum_sell=[]
data_a1_mov_avg_cumulative_min_sell=[]
data_a1_mov_avg_cumulative_vol_max_buy=[]
data_a1_mov_avg_cumulative_vol_max_sell=[]
data_a1_mov_avg_cumulative_vol_mean_sell=[]
data_a1_mov_avg_cumulative_vol_mean_buy=[]
data_a1_mov_avg_cumulative_vol_median_buy=[]
data_a1_mov_avg_cumulative_vol_buy_stddev=[]
data_a1_mov_avg_cumulative_vol_sell_stddev=[]

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
    if i not in malicious_keys:
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
    ## if i in malicious keys
    elif i in malicious_keys:
        # malicious_a1_timestamps = i[0]
        malicious_timestamps.append(i[0])
        malicious_a1_sum_buy = sum(trader_timestamp_dict[i]['buying']['price'])
        malicious_a1_sum_sell = sum(trader_timestamp_dict[i]['selling']['price'])
    
        if len((trader_timestamp_dict[i]['buying']['price']))>0:
            # a1_timestamps_buy.append(i[0])
            malicious_a1_mean_buy = mean(trader_timestamp_dict[i]['buying']['price'])
            malicious_a1_median_buy = median(trader_timestamp_dict[i]['buying']['price'])
            malicious_a1_min_buy = min(trader_timestamp_dict[i]['buying']['price'])
            malicious_a1_max_buy = max(trader_timestamp_dict[i]['buying']['price'])
        elif len((trader_timestamp_dict[i]['buying']['price']))==0:
            # a1_timestamps_buy.append(i[0])
            malicious_a1_mean_buy = 0
            malicious_a1_median_buy = 0
            malicious_a1_min_buy = 0
            malicious_a1_max_buy = 0
        if  len((trader_timestamp_dict[i]['selling']['price']))>0:
            # a1_timestamps_sell.append(i[0])
            malicious_a1_mean_sell = mean(trader_timestamp_dict[i]['selling']['price'])
            malicious_a1_median_sell = median(trader_timestamp_dict[i]['selling']['price'])
            malicious_a1_min_sell = min(trader_timestamp_dict[i]['selling']['price'])
            malicious_a1_max_sell = max(trader_timestamp_dict[i]['selling']['price'])  
        elif  len((trader_timestamp_dict[i]['selling']['price']))==0:
            # a1_timestamps_sell.append(i[0])
            malicious_a1_mean_sell = 0
            malicious_a1_median_sell =0
            malicious_a1_min_sell = 0
            malicious_a1_max_sell = 0  
        if  len((trader_timestamp_dict[i]['buying']['price']))>1:
            # a1_timestamps_buy_stddev.append(i[0])
            malicious_a1_buy_stddev =stdev(trader_timestamp_dict[i]['buying']['price'])
        elif len((trader_timestamp_dict[i]['buying']['price']))>=0:
            malicious_a1_buy_stddev = 0
        if  len((trader_timestamp_dict[i]['selling']['price']))>1:
            # a1_timestamps_sell_stddev.append(i[0])
            malicious_a1_sell_stddev = stdev(trader_timestamp_dict[i]['selling']['price'])
        elif len((trader_timestamp_dict[i]['buying']['price']))>=0:
            malicious_a1_sell_stddev = 0

        malicious_a1_vol_sum_buy = sum(trader_timestamp_dict[i]['buying']['volume'])
        malicious_a1_vol_sum_sell = sum(trader_timestamp_dict[i]['selling']['volume'])
        if  len((trader_timestamp_dict[i]['buying']['volume']))>0:
            # a1_vol_timestamps_buy.append(i[0])
            malicious_a1_vol_mean_buy = mean(trader_timestamp_dict[i]['buying']['volume'])
            malicious_a1_vol_median_buy = median(trader_timestamp_dict[i]['buying']['volume'])
            malicious_a1_vol_min_buy = min(trader_timestamp_dict[i]['buying']['volume'])
            malicious_a1_vol_max_buy = max(trader_timestamp_dict[i]['buying']['volume'])
        elif  len((trader_timestamp_dict[i]['buying']['volume']))==0:
            # a1_vol_timestamps_buy.append(i[0])
            malicious_a1_vol_mean_buy = 0
            malicious_a1_vol_median_buy = 0
            malicious_a1_vol_min_buy = 0
            malicious_a1_vol_max_buy = 0
                
        if  len((trader_timestamp_dict[i]['selling']['volume']))>0:
            # a1_vol_timestamps_sell.append(i[0])
            malicious_a1_vol_mean_sell = mean(trader_timestamp_dict[i]['selling']['volume'])
            malicious_a1_vol_median_sell = median(trader_timestamp_dict[i]['selling']['volume'])
            malicious_a1_vol_min_sell = min(trader_timestamp_dict[i]['selling']['volume'])
            malicious_a1_vol_max_sell = max(trader_timestamp_dict[i]['selling']['volume'])  
        elif  len((trader_timestamp_dict[i]['selling']['volume']))==0:
            # a1_vol_timestamps_sell.append(i[0])
            malicious_a1_vol_mean_sell = 0
            malicious_a1_vol_median_sell = 0
            malicious_a1_vol_min_sell = 0
            malicious_a1_vol_max_sell = 0
        if  len((trader_timestamp_dict[i]['buying']['volume']))>1:
            # a1_vol_timestamps_buy_stddev.append(i[0])
            malicious_a1_vol_buy_stddev = stdev(trader_timestamp_dict[i]['buying']['volume'])
        elif  len((trader_timestamp_dict[i]['buying']['volume']))>=0:
            # a1_vol_timestamps_buy_stddev.append(i[0])
            malicious_a1_vol_buy_stddev = 0
        if  len((trader_timestamp_dict[i]['selling']['volume']))>1:
            # a1_vol_timestamps_sell_stddev.append(i[0])
            malicious_a1_vol_sell_stddev = stdev(trader_timestamp_dict[i]['selling']['volume'])
        if  len((trader_timestamp_dict[i]['selling']['volume']))>=0:
            # a1_vol_timestamps_sell_stddev.append(i[0])
            malicious_a1_vol_sell_stddev = 0
        # a1_sum_buy,a1_mean_buy,a1_median_buy,a1_min_buy,a1_max_buy,a1_buy_stddev,a1_vol_sum_buy,a1_vol_mean_buy,a1_vol_median_buy,a1_vol_min_buy,a1_vol_max_buy, a1_vol_buy_stddev, a1_sum_sell,a1_mean_sell,a1_median_sell,a1_min_sell,a1_max_sell,a1_sell_stddev,a1_vol_sum_sell,a1_vol_mean_sell,a1_vol_median_sell,a1_vol_min_sell,a1_vol_max_sell,a1_vol_sell_stddev

        malicious_complete_data.append([a1_sum_buy,a1_mean_buy,a1_median_buy,a1_min_buy,a1_max_buy,a1_vol_sum_buy,a1_vol_mean_buy,a1_vol_median_buy,a1_vol_min_buy,a1_vol_max_buy, a1_sum_sell,a1_mean_sell,a1_median_sell,a1_min_sell,a1_max_sell,a1_vol_sum_sell,a1_vol_mean_sell,a1_vol_median_sell,a1_vol_min_sell,a1_vol_max_sell])
        malicious_labels.append(-1)
        mal_traders.append(i[1])
        # all_labels.append(1)
normal_complete_data_arr = np.asarray(normal_complete_data)
# malicious_complete_data.pop(0)
malicious_complete_data_arr = np.asarray(malicious_complete_data)            
# print(normal_complete_data_arr.shape)
print(malicious_keys)
mal_trader = {}

for k,c in malicious_keys:
    mal_trader[c] = 0
for k,c in malicious_keys:
    mal_trader[c] += 1

print(len(malicious_complete_data))
print(mal_trader)
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


(shape1,shape2) = malicious_complete_data_arr.shape[0],malicious_complete_data_arr.shape[1]
diff_malicious = np.zeros((shape1,shape2))
for i in range(1,len(malicious_complete_data_arr)):
    diff_malicious[i] = malicious_complete_data_arr[i] - malicious_complete_data_arr[i-1] 

malicious_complete_data_arr = np.hstack((malicious_complete_data_arr,diff_malicious))


file2 = open('non_malicious_timestamps','wb')
pickle.dump(non_malicious_timestamps,file2)

file2 = open('malicious_timestamps','wb')
pickle.dump(malicious_timestamps,file2)

file2 = open('non_malicious','wb')
pickle.dump(non_malicious,file2)

file2 = open('malicious','wb')
pickle.dump(mal_traders,file2)

file2 = open('data_normal','wb')
pickle.dump(normal_complete_data_arr,file2)

file3 = open('data_mal','wb')
pickle.dump(malicious_complete_data_arr,file3)

