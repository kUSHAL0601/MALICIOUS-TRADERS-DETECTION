import matplotlib.pyplot as plt
from statistics import mean,stdev,median
import csv
import math
import numpy as np
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA 
from sklearn.metrics import roc_auc_score
from sklearn.svm import OneClassSVM as oc_svm 

set_folder = 'features_rbf_per_sec/feature_'

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
clf = PCA()
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


all_data = np.vstack((normal_complete_data_arr,malicious_complete_data_arr))

for i in range(len(all_data)):
    if i < len(normal_complete_data_arr):
        all_labels.append(1)
    else:
        all_labels.append(-1)
# print(all_labels.count(1))        
## train_Data
# print(0.8*len(normal_complete_data_arr))
train_normal_arr = normal_complete_data_arr[0:int(0.8*len(normal_complete_data_arr))]
# train_mal_arr = 
test_normal_arr =  normal_complete_data_arr[int(0.8*len(normal_complete_data_arr)):len(normal_complete_data_arr)]

test_all_data = np.vstack((test_normal_arr,malicious_complete_data_arr))



train_normal_arr_stnd_2 = train_normal_arr - train_normal_arr.mean(axis = 0)
# negatives_mat = negatives_mat - negatives_mat.mean(axis =0)
# all_data_mat = all_data_mat - all_data_mat.mean(axis=0)

train_normal_arr_stnd = train_normal_arr_stnd_2/(train_normal_arr_stnd_2.std(axis = 0)+1)

# test_all_data_stnd = test_all_data - test_all_data.mean(axis = 0)
test_all_data_stnd = test_all_data - train_normal_arr.mean(axis = 0)
# negatives_mat = negatives_mat - negatives_mat.mean(axis =0)
# all_data_mat = all_data_mat - all_data_mat.mean(axis=0)

test_all_data_stnd = test_all_data_stnd/(train_normal_arr_stnd_2.std(axis = 0)+1)
## Generate labels

all_feature_vector = np.vstack((train_normal_arr_stnd,test_all_data_stnd))
# normal_complete_data_arr = all_data[0:len(normal_complete_data_arr)]
# test_labels = all_labels[int(0.8)*len(normal_complete_data_arr):len(all_labels)]
train_labels = all_labels[0:int(0.8*len(normal_complete_data_arr))] 
test_labels = all_labels[int(0.8*len(normal_complete_data_arr)):len(all_labels)]




# print("test",test_labels.count(-1))
# clf1 = PCA(n_components = 15,n_selected_components = 3)
# clf1.fit(train_normal_arr_stnd)
# predicted = clf1.predict(test_all_data_stnd)
# accuracy = 0
# recall = 0
# print(clf1.components_)
# for i in range(len(predicted)):
#     if predicted[i] == test_labels[i] and test_labels[i] == 1:
#         recall +=1
#     if predicted[i] == all_labels[i]:
#         accuracy +=1
# print("PCA Accuracy",accuracy/len(train_normal_arr_stnd))
# print("PCA Recall",recall/len(malicious_complete_data_arr))
# print(clf1.singular_values_)
test_keys = non_malicious[int(0.8*len(normal_complete_data_arr)):len(all_labels)] + mal_traders
eta = 1
def sigmoid(x):
  # return 1 / (1 + math.exp(-1*eta*x))
  return 1 / (1 + math.exp(eta*x))
import csv
## OCSVM
# clf1 = OCSVM(kernel = 'rbf',gamma = 1,nu = 0.4)

#Usable
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1,1.2,0.8,2,0.1,0.2,0.3,0.5,2,3,4],'nu':  [0.4,0.03,0.2,0.1,0.05,0.06,0.07,0.08]}]
# tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1],'nu':  [0.4]}]
scores = ['precision','recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(oc_svm(), tuned_parameters, cv=2,
                       scoring='%s_macro' % score)
    clf.fit(train_normal_arr_stnd,train_labels)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = test_labels, clf.predict(test_all_data_stnd)
    print(classification_report(y_true, y_pred))
    print()

clf1 = oc_svm(kernel = 'rbf',nu= 0.03,gamma = 0.1)
set_threshold = 0.9705
# scores = cross_val_score(estimator=clf1, X=train_normal_arr_stnd, cv=5)
clf1.fit(train_normal_arr_stnd)
roc = []
recall_list = []

# clf1.threshold_ = i
predicted1 = clf1.predict(test_all_data_stnd)


# print(predicted)
print(clf1.score_samples(test_all_data_stnd))

# print(test_labels)
# for k in np.arange(-0.01,0.01,0.0005):
# print("for",k)
predicted = []
# try:    
for val in range(len(test_all_data_stnd)):
    # print(clf1.score_samples(test_all_data_stnd[val].reshape(1,-1)))
    if clf1.score_samples(test_all_data_stnd[val].reshape(1,-1)) > set_threshold:
        predicted.append(1)
    else:
        predicted.append(-1)
roc = []
accuracy = 0
recall = 0
indices = []
indices_positive = 0
pred_indices_mal = []
for i in range(len(predicted)):
    if predicted[i] == test_labels[i]:
        accuracy +=1
    if predicted[i] == test_labels[i] and test_labels[i] == -1:
        recall +=1
        indices.append(i)
    if predicted[i] == -1:
        pred_indices_mal.append(i)
        indices_positive +=1
precision = recall/indices_positive
print("OCSVM Precision",100*precision)
print("OCSVM accuracy",100*accuracy/len(test_labels))
print("OCSVM recall",100*recall/len(malicious_complete_data_arr))
print("f measure", ((precision*recall)/(precision+recall)))
# print(len(clf1.support_vectors_))
# print(clf1.score_samples(test_all_data_stnd))
# print(len(clf1.decision_scores_[clf1.decision_scores_>clf1.threshold_])) 
roc.append(roc_auc_score(predicted,test_labels))
recall_list.append(recall)

print("roc",roc_auc_score(predicted,test_labels))
# except:
#     roc.append(0)
#     recall_list.append(0)
#     pass
from sklearn.metrics import roc_curve, auc
# for i in range(n_classes):
# test_labels[test_labels==-1] = 0
# test_labels[test_labels==1] = -1
# test_labels[test_labels==0] = 1
# predicted[predicted==-1] = 0
# predicted[predicted==1] = -1
# predicted[predicted==0] = 1
fpr, tpr, _ = roc_curve(test_labels,predicted)
roc_auc = auc(fpr, tpr)
print(fpr,tpr,_)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate / Recall(here)')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
# plt.show()

score1 = clf1.score_samples(train_normal_arr_stnd)
score2 = clf1.score_samples(test_all_data_stnd)
print(score1.reshape(len(score1),1).shape)
score = np.vstack((score1.reshape(len(score1),1),score2.reshape(len(score2),1)))

# score = clf1.score_samples(test_all_data_stnd)
severity = []
for val in score:
    # print (np.round((1.8 - val)*10**4,2))
    severity.append(sigmoid(np.round((set_threshold - val),5)))
severity_cto = []
print(max(severity))
from collections import Counter
traders_mal = []
for i in range(len(pred_indices_mal)):
    traders_mal.append(test_keys[pred_indices_mal[i]])
    severity_cto.append(severity[pred_indices_mal[i]])
# result = Counter(traders_mal)
# [item for items, c in Counter(traders_mal).most_common() 
                                      # for item in [items] * c]
# print(severity)
severity_cto = np.asarray(severity_cto)
# severity_cto = severity_cto/max(severity_cto)
severity = np.asarray(severity)
# print(severity)
# severity = severity/max(severity)
# print()
# print(len(test_all_data_stnd))
# print(len(indices)/len(test_all_data_stnd))
# print(len(pred_indices_mal))
# print(indices_positive)
cto_input = []
for i in range(len(traders_mal)):
    cto_input.append([traders_mal[i],'layer',severity_cto[i]])
cto_sort = cto_input.sort()
# print(cto_input)
# cto_input_arr = np.asarray(cto_input)
# print(cto_input_arr.shape)
# indexes = np.arange(1,len(cto_input_arr)+1)
# indexes = np.asarray(indexes)
# indexes  = np.matrix(indexes)
# indexes =  indexes.T
# print(indexes.T.shape)
# cto_write = np.hstack((indexes.T,cto_input_arr))
# cto_write = cto_write.tolist()
# count = 1
timestamp_all_list = non_malicious_timestamps + malicious_timestamps
traders_stack = np.vstack((np.asarray(non_malicious).reshape(len(non_malicious),1),np.asarray(mal_traders).reshape(len(mal_traders),1)))
timestamp_stack = np.vstack((np.asarray(non_malicious_timestamps).reshape(len(non_malicious_timestamps),1),np.asarray(malicious_timestamps).reshape(len(malicious_timestamps),1)))
all_trader_timestamp = np.hstack((traders_stack,timestamp_stack))
# all_trader_tstmp_labels = np.hstack((all_trader_timestamp,np.asarray(all_labels).reshape(len(all_labels),1)))

# print(all_feature_vector.shape)
# print(np.asarray(severity).reshape(len(severity),1).shape)
feature_write = np.hstack((all_trader_timestamp,all_feature_vector,severity.reshape(len(severity),1),np.asarray(all_labels).reshape(len(all_labels),1)))

print("all",all_labels.count(1))
print("mal",all_labels.count(-1))


head = 'trader', 'timestamp', 'a1_sum_buy','a1_mean_buy','a1_median_buy','a1_min_buy','a1_max_buy','a1_vol_sum_buy','a1_vol_mean_buy','a1_vol_median_buy','a1_vol_min_buy','a1_vol_max_buy',' a1_sum_sell','a1_mean_sell','a1_median_sell','a1_min_sell','a1_max_sell','a1_vol_sum_sell','a1_vol_mean_sell','a1_vol_median_sell','a1_vol_min_sell','a1_vol_max_sell','diff_a1_sum_buy','diff_a1_mean_buy','diff_a1_median_buy','diff_a1_min_buy','diff_a1_max_buy','diff_a1_vol_sum_buy','diff_a1_vol_mean_buy','diff_a1_vol_median_buy','diff_a1_vol_min_buy','diff_a1_vol_max_buy',' diff_a1_sum_sell','diff_a1_mean_sell','diff_a1_median_sell','diff_a1_min_sell','diff_a1_max_sell','diff_a1_vol_sum_sell','diff_a1_vol_mean_sell','diff_a1_vol_median_sell','diff_a1_vol_min_sell','diff_a1_vol_max_sell','severity','labels'
print(len(head))
print(feature_write.dtype)
feature_write.view('U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32,U32').sort(order=['f1'],axis=0)

all_feature_dict = {}
# timestamp_stack = timestamp_stack.tolist()
# print(timestamp_stack[0])
timestamp_set_list = list(set(timestamp_all_list))
timestamp_set_list.sort()
for key in timestamp_set_list:
    all_feature_dict[(key)] = {}

for key in timestamp_set_list:
    # print(key)
    count = 0
    for i in range(len(feature_write)):
        
        if (key == feature_write[i][1]):
            # print("key, feat_1",key,feature_write[i][1])
            count +=1
            all_feature_dict[(key)][count] = feature_write[i]
for val in timestamp_set_list: 

    with open(set_folder+val+'.csv','w') as f:
        
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
for i in range(int(len(feature_write)/100)+1):
    with open('features_rbf_non_normalized_mult_3/feature_vector_'+str(i)+'.csv','w',) as f:
    # with open('your_file.txt', 'w') as f:
        if i != int(len(feature_write)/100): 
            csv_writer = csv.writer(f)
            # f.write(str(count))
            # csv_writer.writerow([str(len(cto_input))])
            # csv_writer.writerow(['trader', 'timestamp', 'a1_sum_buy','a1_mean_buy','a1_median_buy','a1_min_buy','a1_max_buy','a1_vol_sum_buy','a1_vol_mean_buy','a1_vol_median_buy','a1_vol_min_buy','a1_vol_max_buy',' a1_sum_sell','a1_mean_sell','a1_median_sell','a1_min_sell','a1_max_sell','a1_vol_sum_sell','a1_vol_mean_sell','a1_vol_median_sell','a1_vol_min_sell','a1_vol_max_sell','diff_a1_sum_buy','diff_a1_mean_buy','diff_a1_median_buy','diff_a1_min_buy','diff_a1_max_buy','diff_a1_vol_sum_buy','diff_a1_vol_mean_buy','diff_a1_vol_median_buy','diff_a1_vol_min_buy','diff_a1_vol_max_buy',' diff_a1_sum_sell','diff_a1_mean_sell','diff_a1_median_sell','diff_a1_min_sell','diff_a1_max_sell','diff_a1_vol_sum_sell','diff_a1_vol_mean_sell','diff_a1_vol_median_sell','diff_a1_vol_min_sell','diff_a1_vol_max_sell','severity','labels'])
            csv_writer.writerow(['trader', 'timestamp', 'a1_sum_buy','a1_min_buy','a1_max_buy',' a1_sum_sell','a1_min_sell','a1_max_sell','diff_a1_sum_buy','diff_a1_min_buy','diff_a1_max_buy',' diff_a1_sum_sell','diff_a1_min_sell','diff_a1_max_sell','severity','labels'])
            csv_writer.writerows(feature_write[100*i:100*(i+1)])
        elif i == int(len(feature_write)/100):
            csv_writer = csv.writer(f)
            # f.write(str(count))
            # csv_writer.writerow([str(len(cto_input))])
            csv_writer.writerow(['trader', 'timestamp', 'a1_sum_buy','a1_min_buy','a1_max_buy',' a1_sum_sell','a1_min_sell','a1_max_sell','diff_a1_sum_buy','diff_a1_min_buy','diff_a1_max_buy',' diff_a1_sum_sell','diff_a1_min_sell','diff_a1_max_sell','severity','labels'])
            csv_writer.writerows(feature_write[100*i:len(feature_write)])

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
