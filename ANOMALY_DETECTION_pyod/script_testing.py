import matplotlib.pyplot as plt
from statistics import mean,stdev,median
import csv

import numpy as np
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.auto_encoder import AutoEncoder

imp_features=[]

def print_accuracy(test_arr,train_arr,trader_id,feature,timestamps):
    xyz=0
    for i in malicious_keys:
        if i[1]==trader_id:
            xyz+=1
    if len(train_arr)==0:
        return
    for i in range(len(train_arr)):
        # print(len(train_arr[i]))
        l1=len(train_arr[i])
        l2=len(test_arr[i])
        if l1==0:
            continue
        train_data=np.array([train_arr[i]]).T
        test_data=np.array([test_arr[i]]).T
        clf=OCSVM()
        clf.fit(train_data)
        y_pred=clf.predict(train_data)
        print("FEATURE:",feature,"TRAINING ACCURACY for TRADER",trader_id,":",100 - (sum(y_pred)*100/l1))
        if l2==0:
            continue
        if not xyz:
            return
        mal=[]
        count=0
        for i in range(len(y_pred)):
            if y_pred[i]==1:
                if (timestamps[i],trader_id) in malicious_keys:
                    count+=1
        if count:
            imp_features.append(feature)
        print("FEATURE:",feature,"TESTING ACCURACY for TRADER",trader_id,":",count*100/xyz)

def print_accuracy1(train_arr,trader_id,feature,timestamps):
    xyz=0
    for i in malicious_keys:
        if i[1]==trader_id:
            xyz+=1
    if len(train_arr)==0:
        return
    for i in range(len(train_arr)):
        l1=len(train_arr[i])
        if l1==0:
            continue
        train_data=np.array(train_arr[i]).T
        # print(train_data.shape)
        # test_data=np.array([test_arr[i]]).T
        clf=OCSVM()
        clf.fit(train_data)
        y_pred=clf.predict(train_data)
        print("OCSVM","FEATURE:",feature,"TRAINING ACCURACY for TRADER",trader_id,":",100 - (sum(y_pred)*100/len(y_pred)))

        clf=PCA()
        clf.fit(train_data)
        y_pred=clf.predict(train_data)
        print("PCA","FEATURE:",feature,"TRAINING ACCURACY for TRADER",trader_id,":",100 - (sum(y_pred)*100/len(y_pred)))

        if not xyz:
            return
        clf=OCSVM()
        clf.fit(train_data)
        y_pred=clf.predict(train_data)
        mal=[]
        count=0
        for i in range(len(y_pred)):
            if y_pred[i]==1:
                if (timestamps[i],trader_id) in malicious_keys:
                    count+=1
        print("OCSVM","FEATURES:",feature,"TESTING ACCURACY for TRADER",trader_id,":",count*100/xyz)
        if count:
            imp_features.append(feature)

        clf=PCA()
        clf.fit(train_data)
        y_pred=clf.predict(train_data)
        mal=[]
        count=0
        for i in range(len(y_pred)):
            if y_pred[i]==1:
                if (timestamps[i],trader_id) in malicious_keys:
                    count+=1
        print("PCA","FEATURES:",feature,"TESTING ACCURACY for TRADER",trader_id,":",count*100/xyz)
        if count:
            imp_features.append(feature)



malicious_keys=[]
malicious_traders=[]
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
                    malicious_traders.append(trader_id)

# malicious_traders=list(set(malicious_traders))
# print("MALICIOUS TRADERS")
# for i in malicious_traders:
#     print(i)
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

def draw2(y,x,strx,stry,tit):
        return

def draw_hist(x,strx,stry,tit):
        return

traders=[]

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
        elif int(direction)==-1 and int(entry_type) == 1:
            trader_timestamp_dict[(time_stamp,trader_id)]['selling']['price'].append(price)
            trader_timestamp_dict[(time_stamp,trader_id)]['selling']['volume'].append(volume)
    # print(trader_timestamp_dict)
traders=list(set(traders))
# print(traders)
keys=list(trader_timestamp_dict.keys())
keys.sort()

user_order=[]

for j in traders:
    print("\n\n",j,"\n\n")
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
    malicious_a1_min_buy = []
    malicious_a1_max_buy = []
    malicious_a1_min_sell = []
    malicious_a1_max_sell = []
    malicious_a1_mean_sell = []
    malicious_a1_mean_buy = []
    malicious_a1_median_sell = []
    malicious_a1_median_buy = []
    malicious_a1_stddev_sell = []
    malicious_a1_stddev_buy = []

    malicious_a1_sum_buy_vol=[]
    malicious_a1_sum_sell_vol=[]
    malicious_a1_min_buy_vol = []
    malicious_a1_max_buy_vol = []
    malicious_a1_min_sell_vol = []
    malicious_a1_max_sell_vol = []
    malicious_a1_mean_sell_vol = []
    malicious_a1_mean_buy_vol = []
    malicious_a1_median_sell_vol = []
    malicious_a1_median_buy_vol = []
    malicious_a1_stddev_sell_vol = []
    malicious_a1_stddev_buy_vol = []

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

    malicious_a1_cumulative_sum_buy=[]
    malicious_a1_cumulative_sum_sell=[]
    malicious_a1_cumulative_min_buy=[]
    malicious_a1_cumulative_min_sell=[]
    malicious_a1_cumulative_max_buy=[]
    malicious_a1_cumulative_max_sell=[]
    malicious_a1_cumulative_median_buy=[]
    malicious_a1_cumulative_median_sell=[]
    malicious_a1_cumulative_buy_stddev=[]
    malicious_a1_cumulative_sell_stddev=[]

    malicious_a1_cumulative_vol_sum_buy=[]
    malicious_a1_cumulative_vol_sum_sell=[]
    malicious_a1_cumulative_vol_min_buy=[]
    malicious_a1_cumulative_vol_min_sell=[]
    malicious_a1_cumulative_vol_max_buy=[]
    malicious_a1_cumulative_vol_max_sell=[]
    malicious_a1_cumulative_vol_median_buy=[]
    malicious_a1_cumulative_vol_median_sell=[]
    malicious_a1_cumulative_vol_buy_stddev=[]
    malicious_a1_cumulative_vol_sell_stddev=[]

    for i in keys:
        # print(i)
        if i[1]==j:
        # if True:
            # user_order.append(i)
            a1_timestamps.append(i[0])
            a1_sum_buy.append(sum(trader_timestamp_dict[i]['buying']['price']))
            a1_sum_sell.append(sum(trader_timestamp_dict[i]['selling']['price']))
            if i in malicious_keys:
                malicious_a1_sum_buy.append(sum(trader_timestamp_dict[i]['buying']['price']))
                malicious_a1_sum_sell.append(sum(trader_timestamp_dict[i]['selling']['price']))
            if  len((trader_timestamp_dict[i]['buying']['price']))>0:
                a1_timestamps_buy.append(i[0])
                a1_mean_buy.append(mean(trader_timestamp_dict[i]['buying']['price']))
                a1_median_buy.append(median(trader_timestamp_dict[i]['buying']['price']))
                a1_min_buy.append(min(trader_timestamp_dict[i]['buying']['price']))
                a1_max_buy.append(max(trader_timestamp_dict[i]['buying']['price']))
                if i in malicious_keys:
                    malicious_a1_mean_buy.append(mean(trader_timestamp_dict[i]['buying']['price']))
                    malicious_a1_median_buy.append(median(trader_timestamp_dict[i]['buying']['price']))
                    malicious_a1_min_buy.append(min(trader_timestamp_dict[i]['buying']['price']))
                    malicious_a1_max_buy.append(max(trader_timestamp_dict[i]['buying']['price']))
            if  len((trader_timestamp_dict[i]['selling']['price']))>0:
                a1_timestamps_sell.append(i[0])
                a1_mean_sell.append(mean(trader_timestamp_dict[i]['selling']['price']))
                a1_median_sell.append(median(trader_timestamp_dict[i]['selling']['price']))
                a1_min_sell.append(min(trader_timestamp_dict[i]['selling']['price']))
                a1_max_sell.append(max(trader_timestamp_dict[i]['selling']['price']))
                if i in malicious_keys:
                    malicious_a1_mean_sell.append(mean(trader_timestamp_dict[i]['selling']['price']))
                    malicious_a1_median_sell.append(median(trader_timestamp_dict[i]['selling']['price']))
                    malicious_a1_min_sell.append(min(trader_timestamp_dict[i]['selling']['price']))
                    malicious_a1_max_sell.append(max(trader_timestamp_dict[i]['selling']['price']))  
            if  len((trader_timestamp_dict[i]['buying']['price']))>1:
                a1_timestamps_buy_stddev.append(i[0])
                a1_buy_stddev.append(stdev(trader_timestamp_dict[i]['buying']['price']))
                if i in malicious_keys:
                    malicious_a1_stddev_buy.append(stdev(trader_timestamp_dict[i]['buying']['price']))
            if  len((trader_timestamp_dict[i]['selling']['price']))>1:
                a1_timestamps_sell_stddev.append(i[0])
                a1_sell_stddev.append(stdev(trader_timestamp_dict[i]['selling']['price']))
                if i in malicious_keys:
                    malicious_a1_stddev_sell.append(stdev(trader_timestamp_dict[i]['selling']['price']))


#  Volume computations
            a1_vol_sum_buy.append(sum(trader_timestamp_dict[i]['buying']['volume']))
            a1_vol_sum_sell.append(sum(trader_timestamp_dict[i]['selling']['volume']))
            if i in malicious_keys:
                malicious_a1_sum_buy_vol.append(sum(trader_timestamp_dict[i]['buying']['volume']))
                malicious_a1_sum_sell_vol.append(sum(trader_timestamp_dict[i]['selling']['volume']))
            if  len((trader_timestamp_dict[i]['buying']['volume']))>0:
                a1_vol_timestamps_buy.append(i[0])
                a1_vol_mean_buy.append(mean(trader_timestamp_dict[i]['buying']['volume']))
                a1_vol_median_buy.append(median(trader_timestamp_dict[i]['buying']['volume']))
                a1_vol_min_buy.append(min(trader_timestamp_dict[i]['buying']['volume']))
                a1_vol_max_buy.append(max(trader_timestamp_dict[i]['buying']['volume']))
                if i in malicious_keys:
                    malicious_a1_mean_buy_vol.append(mean(trader_timestamp_dict[i]['buying']['volume']))
                    malicious_a1_median_buy_vol.append(median(trader_timestamp_dict[i]['buying']['volume']))
                    malicious_a1_min_buy_vol.append(min(trader_timestamp_dict[i]['buying']['volume']))
                    malicious_a1_max_buy_vol.append(max(trader_timestamp_dict[i]['buying']['volume']))
            if  len((trader_timestamp_dict[i]['selling']['volume']))>0:
                a1_vol_timestamps_sell.append(i[0])
                a1_vol_mean_sell.append(mean(trader_timestamp_dict[i]['selling']['volume']))
                a1_vol_median_sell.append(median(trader_timestamp_dict[i]['selling']['volume']))
                a1_vol_min_sell.append(min(trader_timestamp_dict[i]['selling']['volume']))
                a1_vol_max_sell.append(max(trader_timestamp_dict[i]['selling']['volume']))
                if i in malicious_keys:
                    malicious_a1_mean_sell_vol.append(mean(trader_timestamp_dict[i]['selling']['volume']))
                    malicious_a1_median_sell_vol.append(median(trader_timestamp_dict[i]['selling']['volume']))
                    malicious_a1_min_sell_vol.append(min(trader_timestamp_dict[i]['selling']['volume']))
                    malicious_a1_max_sell_vol.append(max(trader_timestamp_dict[i]['selling']['volume']))  
            if  len((trader_timestamp_dict[i]['buying']['volume']))>1:
                a1_vol_timestamps_buy_stddev.append(i[0])
                a1_vol_buy_stddev.append(stdev(trader_timestamp_dict[i]['buying']['volume']))
                if i in malicious_keys:
                    malicious_a1_stddev_buy_vol.append(stdev(trader_timestamp_dict[i]['buying']['volume']))
            if  len((trader_timestamp_dict[i]['selling']['volume']))>1:
                a1_vol_timestamps_sell_stddev.append(i[0])
                a1_vol_sell_stddev.append(stdev(trader_timestamp_dict[i]['selling']['volume']))
                if i in malicious_keys:
                    malicious_a1_stddev_sell_vol.append(stdev(trader_timestamp_dict[i]['selling']['volume']))
    a1_cumulative_sum_buy = cumulative_sum(a1_sum_buy)
    a1_cumulative_sum_sell = cumulative_sum(a1_sum_sell)
    a1_cumulative_mean_sell = cumulative_sum(a1_mean_sell)
    a1_cumulative_mean_buy = cumulative_sum(a1_mean_buy)
    a1_cumulative_median_sell = cumulative_sum(a1_median_sell)
    a1_cumulative_median_buy = cumulative_sum(a1_median_buy)
    a1_cumulative_buy_stddev = cumulative_sum(a1_buy_stddev)
    a1_cumulative_sell_stddev = cumulative_sum(a1_sell_stddev)
    a1_cumulative_min_buy = cumulative_sum(a1_min_buy)
    a1_cumulative_min_sell = cumulative_sum(a1_min_sell)
    a1_cumulative_max_buy = cumulative_sum(a1_max_buy)
    a1_cumulative_max_sell = cumulative_sum(a1_max_sell)

    a1_cumulative_vol_sum_buy = cumulative_sum(a1_vol_sum_buy)
    a1_cumulative_vol_sum_sell = cumulative_sum(a1_vol_sum_sell)
    a1_cumulative_vol_mean_sell = cumulative_sum(a1_vol_mean_sell)
    a1_cumulative_vol_mean_buy = cumulative_sum(a1_vol_mean_buy)
    a1_cumulative_vol_median_sell = cumulative_sum(a1_vol_median_sell)
    a1_cumulative_vol_median_buy = cumulative_sum(a1_vol_median_buy)
    a1_cumulative_vol_buy_stddev = cumulative_sum(a1_vol_buy_stddev)
    a1_cumulative_vol_sell_stddev = cumulative_sum(a1_vol_sell_stddev)
    a1_cumulative_vol_min_buy = cumulative_sum(a1_vol_min_buy)
    a1_cumulative_vol_min_sell = cumulative_sum(a1_vol_min_sell)
    a1_cumulative_vol_max_buy = cumulative_sum(a1_vol_max_buy)
    a1_cumulative_vol_max_sell = cumulative_sum(a1_vol_max_sell)


    for z in malicious_keys:
        if z[1]==j:
            t_stamp=z[0]
            try:
                idx=a1_timestamps.index(t_stamp)
                if idx>=0:
                    malicious_a1_cumulative_sum_buy.append(a1_cumulative_sum_buy[idx])
                    malicious_a1_cumulative_sum_sell.append(a1_cumulative_sum_sell[idx])
            except:
                pass
            try:
                idx=a1_timestamps_buy.index(t_stamp)
                if idx>=0:
                    malicious_a1_cumulative_mean_buy.append(a1_cumulative_mean_buy[idx])
                    malicious_a1_cumulative_median_buy.append(a1_cumulative_median_buy[idx])
                    malicious_a1_cumulative_min_buy.append(a1_cumulative_min_buy[idx])
                    malicious_a1_cumulative_max_buy.append(a1_cumulative_max_buy[idx])
            except:
                pass
            try:
                idx=a1_timestamps_sell.index(t_stamp)
                if idx>=0:
                    malicious_a1_cumulative_mean_sell.append(a1_cumulative_mean_buy[idx])
                    malicious_a1_cumulative_median_sell.append(a1_cumulative_median_buy[idx])
                    malicious_a1_cumulative_min_sell.append(a1_cumulative_min_sell[idx])
                    malicious_a1_cumulative_max_sell.append(a1_cumulative_max_sell[idx])
            except:
                pass
            try:
                idx=a1_timestamps_buy_stddev.index(t_stamp)
                if idx>=0:
                    malicious_a1_cumulative_buy_stddev.append(a1_cumulative_buy_stddev[idx])
            except:
                pass
            try:
                idx=a1_timestamps_sell_stddev.index(t_stamp)
                if idx>=0:
                    malicious_a1_cumulative_sell_stddev.append(a1_cumulative_sell_stddev[idx])
            except:
                pass

            try:
                idx=a1_vol_timestamps.index(t_stamp)
                if idx>=0:
                    malicious_a1_cumulative_vol_sum_buy.append(a1_cumulative_vol_sum_buy[idx])
                    malicious_a1_cumulative_vol_sum_sell.append(a1_cumulative_vol_sum_sell[idx])
            except:
                pass
            try:
                idx=a1_vol_timestamps_buy.index(t_stamp)
                if idx>=0:
                    malicious_a1_cumulative_vol_mean_buy.append(a1_cumulative_vol_mean_buy[idx])
                    malicious_a1_cumulative_vol_median_buy.append(a1_cumulative_vol_median_buy[idx])
                    malicious_a1_cumulative_vol_min_buy.append(a1_cumulative_vol_min_buy[idx])
                    malicious_a1_cumulative_vol_max_buy.append(a1_cumulative_vol_max_buy[idx])
            except:
                pass
            try:
                idx=a1_vol_timestamps_sell.index(t_stamp)
                if idx>=0:
                    malicious_a1_cumulative_vol_mean_sell.append(a1_cumulative_vol_mean_buy[idx])
                    malicious_a1_cumulative_vol_median_sell.append(a1_cumulative_vol_median_buy[idx])
                    malicious_a1_cumulative_vol_min_sell.append(a1_cumulative_vol_min_sell[idx])
                    malicious_a1_cumulative_vol_max_sell.append(a1_cumulative_vol_max_sell[idx])
            except:
                pass
            try:
                idx=a1_vol_timestamps_buy_stddev.index(t_stamp)
                if idx>=0:
                    malicious_a1_cumulative_vol_buy_stddev.append(a1_cumulative_vol_buy_stddev[idx])
            except:
                pass
            try:
                idx=a1_vol_timestamps_sell_stddev.index(t_stamp)
                if idx>=0:
                    malicious_a1_cumulative_vol_sell_stddev.append(a1_cumulative_vol_sell_stddev[idx])
            except:
                pass

    # GRAPHS

    # 4 Graphs of sum
    draw(a1_timestamps, a1_sum_buy , "time stamps" , "Total Buying capacity of T_3" , "Total Buying capacity of a trader w.r.t. time")
    data_sum_buy.append(a1_sum_buy)
    draw(a1_timestamps, a1_sum_sell , "time stamps" , "Total selling capacity of T_3" , "Total selling capacity of a trader w.r.t. time")
    data_sum_sell.append(a1_sum_sell)
    draw(a1_timestamps, a1_vol_sum_buy , "time stamps" , "Total Buying volume of T_3" , "Total Buying volume of a trader w.r.t. time")
    data_a1_vol_sum_buy.append(a1_vol_sum_buy)
    draw(a1_timestamps, a1_vol_sum_sell , "time stamps" , "Total selling volume of T_3" , "Total selling volume of a trader w.r.t. time")
    data_a1_vol_sum_sell.append(a1_vol_sum_sell)

    # 4 Graphs of std-dev
    draw(a1_timestamps_buy_stddev, a1_buy_stddev , "time stamps" , "Std. deviation of buying price" , "Std. deviation of buying price of a trader w.r.t. time")
    data_a1_buy_stddev.append(a1_buy_stddev)
    if i in malicious_keys:
        malicious_data_a1_buy_stddev.append(a1_buy_stddev)
    draw(a1_timestamps_sell_stddev, a1_sell_stddev , "time stamps" , "Std. deviation of selling price" , "Std. deviation of selling price of a trader w.r.t. time")
    data_a1_sell_stddev.append(a1_sell_stddev)
    draw(a1_vol_timestamps_buy_stddev, a1_vol_buy_stddev , "time stamps" , "Std. deviation of buying volume" , "Std. deviation of buying volume of a trader w.r.t. time")
    data_a1_vol_buy_stddev.append(a1_vol_buy_stddev)
    draw(a1_vol_timestamps_sell_stddev, a1_vol_sell_stddev , "time stamps" , "Std. deviation of selling volume" , "Std. deviation of selling volume of a trader w.r.t. time")
    data_a1_vol_sell_stddev.append(a1_vol_sell_stddev)

    # 12 Graphs of Cumulative {Mean , Standard Deviation , Sum)-----> (Think about - Moving average}
    
    draw(a1_timestamps, a1_cumulative_sum_buy, "time stamps", "Cumulative sum of T_3" , "Cumulative sum of buying prices of a trader w.r.t time stamps")
    data_a1_cumulative_sum_buy.append(a1_cumulative_sum_buy)
    draw(a1_timestamps, a1_cumulative_sum_sell, "time stamps", "Cumulative sum of T_3" , "Cumulative sum of selling prices of a trader w.r.t time stamps")
    data_a1_cumulative_sum_sell.append(a1_cumulative_sum_sell)
    draw(a1_timestamps, a1_cumulative_vol_sum_buy, "time stamps", "Cumulative sum of T_3" , "Cumulative sum of buying volumes of a trader w.r.t time stamps")
    data_a1_cumulative_vol_sum_buy.append(a1_cumulative_vol_sum_buy)

    draw(a1_timestamps_buy, a1_cumulative_mean_buy, "time stamps", "Cumulative mean of T_3" , "Cumulative mean of buying prices of a trader w.r.t time stamps")
    data_a1_cumulative_mean_buy.append(a1_cumulative_mean_buy)
    draw(a1_timestamps, a1_cumulative_vol_sum_sell, "time stamps", "Cumulative sum of T_3" , "Cumulative sum of selling volumes of a trader w.r.t time stamps")
    data_a1_cumulative_vol_sum_sell.append(a1_cumulative_vol_sum_sell)
    draw(a1_timestamps_buy, a1_cumulative_vol_mean_buy, "time stamps", "Cumulative mean of T_3" , "Cumulative mean of buying volume of a trader w.r.t time stamps")
    data_a1_cumulative_vol_mean_buy.append(a1_cumulative_vol_mean_buy)
    draw(a1_timestamps_sell, a1_cumulative_mean_sell, "time stamps", "Cumulative mean of T_3" , "Cumulative mean of selling prices of a trader w.r.t time stamps")
    data_a1_cumulative_mean_sell.append(a1_cumulative_mean_sell)
    draw(a1_timestamps_sell, a1_cumulative_vol_mean_sell, "time stamps", "Cumulative mean of T_3" , "Cumulative mean of selling volume of a trader w.r.t time stamps")
    data_a1_cumulative_vol_mean_sell.append(a1_cumulative_vol_mean_sell)

    draw(a1_timestamps_buy_stddev, a1_cumulative_buy_stddev, "time stamps", "Cumulative std-dev of T_3" , "Cumulative std-dev of buying prices of a trader w.r.t time stamps")
    data_a1_cumulative_buy_stddev.append(a1_cumulative_buy_stddev)
    draw(a1_timestamps_buy_stddev, a1_cumulative_vol_buy_stddev, "time stamps", "Cumulative std-dev of T_3" , "Cumulative std-dev of buying volume of a trader w.r.t time stamps")
    data_a1_cumulative_vol_buy_stddev.append(a1_cumulative_vol_buy_stddev)
    draw(a1_timestamps_sell_stddev, a1_cumulative_sell_stddev, "time stamps", "Cumulative std-dev of T_3" , "Cumulative std-dev of selling prices of a trader w.r.t time stamps")
    data_a1_cumulative_sell_stddev.append(a1_cumulative_sell_stddev)
    draw(a1_timestamps_sell_stddev, a1_cumulative_vol_sell_stddev, "time stamps", "Cumulative std-dev of T_3" , "Cumulative std-dev of selling volume of a trader w.r.t time stamps")
    data_a1_cumulative_vol_sell_stddev.append(a1_cumulative_vol_sell_stddev)

    # HISTOGRAM

    # 4 Graphs of sum
    draw_hist( a1_sum_buy , "time stamps" , "Total Buying capacity of T_3" , "Total Buying capacity of a trader w.r.t. time")
    draw_hist( a1_sum_sell , "time stamps" , "Total selling capacity of T_3" , "Total selling capacity of a trader w.r.t. time")
    draw_hist( a1_vol_sum_buy , "time stamps" , "Total Buying volume of T_3" , "Total Buying volume of a trader w.r.t. time")
    draw_hist( a1_vol_sum_sell , "time stamps" , "Total selling volume of T_3" , "Total selling volume of a trader w.r.t. time")

    # 4 Graphs of std-dev
    draw_hist( a1_buy_stddev , "time stamps" , "Std. deviation of buying price" , "Std. deviation of buying price of a trader w.r.t. time")
    draw_hist( a1_sell_stddev , "time stamps" , "Std. deviation of selling price" , "Std. deviation of selling price of a trader w.r.t. time")
    draw_hist( a1_vol_buy_stddev , "time stamps" , "Std. deviation of buying volume" , "Std. deviation of buying volume of a trader w.r.t. time")
    draw_hist( a1_vol_sell_stddev , "time stamps" , "Std. deviation of selling volume" , "Std. deviation of selling volume of a trader w.r.t. time")

    # 12 Graphs of Cumulative {Mean , Standard Deviation , Sum)-----> (Think about - Moving average}
    
    draw_hist( a1_cumulative_sum_buy, "time stamps", "Cumulative sum of T_3" , "Cumulative sum of buying prices of a trader w.r.t time stamps")
    draw_hist( a1_cumulative_sum_sell, "time stamps", "Cumulative sum of T_3" , "Cumulative sum of selling prices of a trader w.r.t time stamps")
    draw_hist( a1_cumulative_vol_sum_buy, "time stamps", "Cumulative sum of T_3" , "Cumulative sum of buying volumes of a trader w.r.t time stamps")
    draw_hist( a1_cumulative_vol_sum_sell, "time stamps", "Cumulative sum of T_3" , "Cumulative sum of selling volumes of a trader w.r.t time stamps")

    draw_hist( a1_cumulative_mean_buy, "time stamps", "Cumulative mean of T_3" , "Cumulative mean of buying prices of a trader w.r.t time stamps")
    draw_hist( a1_cumulative_vol_mean_buy, "time stamps", "Cumulative mean of T_3" , "Cumulative mean of buying volume of a trader w.r.t time stamps")
    draw_hist( a1_cumulative_mean_sell, "time stamps", "Cumulative mean of T_3" , "Cumulative mean of selling prices of a trader w.r.t time stamps")
    draw_hist( a1_cumulative_vol_mean_sell, "time stamps", "Cumulative mean of T_3" , "Cumulative mean of selling volume of a trader w.r.t time stamps")

    draw_hist( a1_cumulative_buy_stddev, "time stamps", "Cumulative std-dev of T_3" , "Cumulative std-dev of buying prices of a trader w.r.t time stamps")
    draw_hist( a1_cumulative_vol_buy_stddev, "time stamps", "Cumulative std-dev of T_3" , "Cumulative std-dev of buying volume of a trader w.r.t time stamps")
    draw_hist( a1_cumulative_sell_stddev, "time stamps", "Cumulative std-dev of T_3" , "Cumulative std-dev of selling prices of a trader w.r.t time stamps")
    draw_hist( a1_cumulative_vol_sell_stddev, "time stamps", "Cumulative std-dev of T_3" , "Cumulative std-dev of selling volume of a trader w.r.t time stamps")

    # 32 (4x8) graphs of moving averages of { min , max , mean , std dev , median , sum , cumulative (sum & mean) }


    # GRAPHS

    # Plotting Buying Price Statistics
    plt.plot(a1_timestamps_buy,a1_mean_buy)
    data_a1_mean_buy.append(a1_mean_buy)
    plt.plot(a1_timestamps_buy,a1_min_buy)
    data_a1_min_buy.append(a1_min_buy)
    plt.plot(a1_timestamps_buy,a1_max_buy)
    data_a1_max_buy.append(a1_max_buy)
    plt.plot(a1_timestamps_buy,a1_median_buy)
    data_a1_median_buy.append(a1_median_buy)
    plt.title("statistics of buying prices of trader w.r.t time stamps")
    plt.xlabel("time stamps")
    plt.ylabel("price")
    plt.legend(['mean of buying prices', 'minimum buying prices', 'maximum buying prices' , 'median of buying prices'], loc='lower right')
    plt.xticks(rotation=90)
    # #plt.savefig('BuyingPrices.png')
    # plt.show()


    # Plotting selling price statistics
    plt.plot(a1_timestamps_sell,a1_mean_sell)
    data_a1_mean_sell.append(a1_mean_sell)
    plt.plot(a1_timestamps_sell,a1_min_sell)
    data_a1_min_sell.append(a1_min_sell)
    plt.plot(a1_timestamps_sell,a1_max_sell)
    data_a1_max_sell.append(a1_max_sell)
    plt.plot(a1_timestamps_sell,a1_median_sell)
    data_a1_median_sell.append(a1_median_sell)
    plt.title("statistics of selling prices of trader w.r.t time stamps")
    plt.xlabel("time stamps")
    plt.ylabel("price")
    plt.legend(['mean of selling prices', 'minimum buying prices', 'maximum buying prices' , 'median of selling prices'], loc='lower right')
    plt.tight_layout()
    plt.xticks(rotation=90)
    # plt.show()
    # #plt.savefig('SellingPrices.png')

    # Plotting Buying Volume Statistics
    plt.plot(a1_vol_timestamps_buy,a1_vol_mean_buy)
    data_a1_vol_mean_buy.append(a1_vol_mean_buy)
    plt.plot(a1_vol_timestamps_buy,a1_vol_min_buy)
    data_a1_vol_min_buy.append(a1_vol_min_buy)
    plt.plot(a1_vol_timestamps_buy,a1_vol_max_buy)
    data_a1_vol_max_buy.append(a1_vol_max_buy)
    plt.plot(a1_vol_timestamps_buy,a1_vol_median_buy)
    data_a1_vol_median_buy.append(a1_vol_median_buy)
    plt.title("statistics of buying volume of trader w.r.t time stamps")
    plt.xlabel("time stamps")
    plt.ylabel("volume")
    plt.legend(['mean of buying volumes', 'minimum buying volumes', 'maximum buying volumes' , 'median of buying volumes'], loc='lower right')
    plt.xticks(rotation=90)
    # plt.show()
    # #plt.savefig('BuyingVolume.png')


    # Plotting selling Volume statistics
    plt.plot(a1_vol_timestamps_sell,a1_vol_mean_sell)
    data_a1_vol_mean_sell.append(a1_vol_mean_sell)
    plt.plot(a1_vol_timestamps_sell,a1_vol_min_sell)
    data_a1_vol_min_sell.append(a1_vol_min_sell)
    plt.plot(a1_vol_timestamps_sell,a1_vol_max_sell)
    data_a1_vol_max_sell.append(a1_vol_max_sell)
    plt.plot(a1_vol_timestamps_sell,a1_vol_median_sell)
    data_a1_vol_median_sell.append(a1_vol_median_sell)
    plt.title("statistics of selling volume of trader w.r.t time stamps")
    plt.xlabel("time stamps")
    plt.ylabel("volume")
    plt.legend(['mean of selling volumes', 'minimum buying volumes', 'maximum buying volumes' , 'median of selling volumes'], loc='lower right')
    plt.tight_layout()
    plt.xticks(rotation=90)
    # plt.show()
    # #plt.savefig('SellingVolume.png')


    # HISTOGRAMS

    # Plotting Buying Price Statistics
    plt.hist(a1_mean_buy,bins=20)
    plt.title("statistics of buying prices of trader w.r.t time stamps")
    plt.xlabel("time stamps")
    plt.ylabel("price")
    plt.legend(['mean of buying prices', 'minimum buying prices', 'maximum buying prices' , 'median of buying prices'], loc='lower right')
    plt.xticks(rotation=90)
    # #plt.savefig('BuyingPricesMean_hist.png')

    plt.hist(a1_min_buy,bins=20)
    plt.title("statistics of buying prices of trader w.r.t time stamps")
    plt.xlabel("time stamps")
    plt.ylabel("price")
    plt.legend(['mean of buying prices', 'minimum buying prices', 'maximum buying prices' , 'median of buying prices'], loc='lower right')
    plt.xticks(rotation=90)
    # #plt.savefig('BuyingPricesMin_hist.png')

    plt.hist(a1_max_buy,bins=20)
    plt.title("statistics of buying prices of trader w.r.t time stamps")
    plt.xlabel("time stamps")
    plt.ylabel("price")
    plt.legend(['mean of buying prices', 'minimum buying prices', 'maximum buying prices' , 'median of buying prices'], loc='lower right')
    plt.xticks(rotation=90)
    # #plt.savefig('BuyingPricesMax_hist.png')

    plt.hist(a1_median_buy,bins=20)
    plt.title("statistics of buying prices of trader w.r.t time stamps")
    plt.xlabel("time stamps")
    plt.ylabel("price")
    plt.legend(['mean of buying prices', 'minimum buying prices', 'maximum buying prices' , 'median of buying prices'], loc='lower right')
    plt.xticks(rotation=90)
    # #plt.savefig('BuyingPricesMedian_hist.png')

    # Plotting selling price statistics
    plt.hist(a1_mean_sell,bins=20)
    plt.title("statistics of selling prices of trader w.r.t time stamps")
    plt.xlabel("time stamps")
    plt.ylabel("price")
    plt.legend(['mean of selling prices', 'minimum selling prices', 'maximum selling prices' , 'median of selling prices'], loc='lower right')
    plt.xticks(rotation=90)
    # #plt.savefig('SellingPricesMean_hist.png')

    plt.hist(a1_min_sell,bins=20)
    plt.title("statistics of selling prices of trader w.r.t time stamps")
    plt.xlabel("time stamps")
    plt.ylabel("price")
    plt.legend(['mean of selling prices', 'minimum selling prices', 'maximum selling prices' , 'median of selling prices'], loc='lower right')
    plt.xticks(rotation=90)
    # #plt.savefig('SellingPricesMin_hist.png')

    plt.hist(a1_max_sell,bins=20)
    plt.title("statistics of selling prices of trader w.r.t time stamps")
    plt.xlabel("time stamps")
    plt.ylabel("price")
    plt.legend(['mean of selling prices', 'minimum selling prices', 'maximum selling prices' , 'median of selling prices'], loc='lower right')
    plt.xticks(rotation=90)
    # #plt.savefig('SellingPricesMax_hist.png')

    plt.hist(a1_median_sell,bins=20)
    plt.title("statistics of selling prices of trader w.r.t time stamps")
    plt.xlabel("time stamps")
    plt.ylabel("price")
    plt.legend(['mean of selling prices', 'minimum selling prices', 'maximum selling prices' , 'median of selling prices'], loc='lower right')
    plt.xticks(rotation=90)
    # #plt.savefig('SellingPricesMedian_hist.png')

    # Plotting Buying Volume Statistics
    plt.hist(a1_vol_mean_buy,bins=20)
    plt.title("statistics of buying volumes of trader w.r.t time stamps")
    plt.xlabel("time stamps")
    plt.ylabel("volume")
    plt.legend(['mean of buying volumes', 'minimum buying volumes', 'maximum buying volumes' , 'median of buying volumes'], loc='lower right')
    plt.xticks(rotation=90)
    # #plt.savefig('BuyingVolumesMean_hist.png')

    plt.hist(a1_vol_min_buy,bins=20)
    plt.title("statistics of buying volumes of trader w.r.t time stamps")
    plt.xlabel("time stamps")
    plt.ylabel("volume")
    plt.legend(['mean of buying volumes', 'minimum buying volumes', 'maximum buying volumes' , 'median of buying volumes'], loc='lower right')
    plt.xticks(rotation=90)
    # #plt.savefig('BuyingVolumesMin_hist.png')

    plt.hist(a1_vol_max_buy,bins=20)
    plt.title("statistics of buying volumes of trader w.r.t time stamps")
    plt.xlabel("time stamps")
    plt.ylabel("volume")
    plt.legend(['mean of buying volumes', 'minimum buying volumes', 'maximum buying volumes' , 'median of buying volumes'], loc='lower right')
    plt.xticks(rotation=90)
    # #plt.savefig('BuyingVolumesMax_hist.png')

    plt.hist(a1_vol_median_buy,bins=20)
    plt.title("statistics of buying volumes of trader w.r.t time stamps")
    plt.xlabel("time stamps")
    plt.ylabel("volume")
    plt.legend(['mean of buying volumes', 'minimum buying volumes', 'maximum buying volumes' , 'median of buying volumes'], loc='lower right')
    plt.xticks(rotation=90)
    # #plt.savefig('BuyingVolumesMedian_hist.png')


    # Plotting selling Volume statistics
    plt.hist(a1_vol_mean_sell,bins=20)
    plt.title("statistics of selling volumes of trader w.r.t time stamps")
    plt.xlabel("time stamps")
    plt.ylabel("volume")
    plt.legend(['mean of selling volumes', 'minimum selling volumes', 'maximum selling volumes' , 'median of selling volumes'], loc='lower right')
    plt.xticks(rotation=90)
    # #plt.savefig('SellingVolumesMean_hist.png')

    plt.hist(a1_vol_min_sell,bins=20)
    plt.title("statistics of selling volumes of trader w.r.t time stamps")
    plt.xlabel("time stamps")
    plt.ylabel("volume")
    plt.legend(['mean of selling volumes', 'minimum selling volumes', 'maximum selling volumes' , 'median of selling volumes'], loc='lower right')
    plt.xticks(rotation=90)
    # #plt.savefig('SellingVolumesMin_hist.png')

    plt.hist(a1_vol_max_sell,bins=20)
    plt.title("statistics of selling volumes of trader w.r.t time stamps")
    plt.xlabel("time stamps")
    plt.ylabel("volume")
    plt.legend(['mean of selling volumes', 'minimum selling volumes', 'maximum selling volumes' , 'median of selling volumes'], loc='lower right')
    plt.xticks(rotation=90)
    # #plt.savefig('SellingVolumesMax_hist.png')

    plt.hist(a1_vol_median_sell,bins=20)
    plt.title("statistics of selling volumes of trader w.r.t time stamps")
    plt.xlabel("time stamps")
    plt.ylabel("volume")
    plt.legend(['mean of selling volumes', 'minimum selling volumes', 'maximum selling volumes' , 'median of selling volumes'], loc='lower right')
    plt.xticks(rotation=90)
    #plt.savefig('SellingVolumesMedian_hist.png')


    window_size = 5

    a1_mov_avg_sum_buy = moving_average(a1_sum_buy,window_size)

    data_a1_mov_avg_sum_buy.append(a1_mov_avg_sum_buy)
    a1_mov_avg_min_buy = moving_average(a1_min_buy,window_size)
    data_a1_mov_avg_min_buy.append(a1_mov_avg_min_buy)
    a1_mov_avg_max_buy = moving_average(a1_max_buy,window_size)
    data_a1_mov_avg_max_buy.append(a1_mov_avg_max_buy)
    a1_mov_avg_sum_sell = moving_average(a1_sum_sell,window_size)
    data_a1_mov_avg_sum_sell.append(a1_mov_avg_sum_sell)
    a1_mov_avg_min_sell = moving_average(a1_min_sell,window_size)
    data_a1_mov_avg_min_sell.append(a1_mov_avg_min_sell)
    a1_mov_avg_max_sell = moving_average(a1_max_sell,window_size)
    data_a1_mov_avg_max_sell.append(a1_mov_avg_max_sell)
    a1_mov_avg_mean_sell = moving_average(a1_mean_sell,window_size)
    data_a1_mov_avg_mean_sell.append(a1_mov_avg_mean_sell)
    a1_mov_avg_mean_buy = moving_average(a1_mean_buy,window_size)
    data_a1_mov_avg_mean_buy.append(a1_mov_avg_mean_buy)
    a1_mov_avg_median_sell = moving_average(a1_median_sell,window_size)
    data_a1_mov_avg_median_sell.append(a1_mov_avg_median_sell)
    a1_mov_avg_median_buy = moving_average(a1_median_buy,window_size)
    data_a1_mov_avg_median_buy.append(a1_mov_avg_median_buy)
    a1_mov_avg_buy_stddev = moving_average(a1_buy_stddev,window_size)
    data_a1_mov_avg_buy_stddev.append(a1_mov_avg_buy_stddev)
    a1_mov_avg_sell_stddev = moving_average(a1_sell_stddev,window_size)
    data_a1_mov_avg_sell_stddev.append(a1_mov_avg_sell_stddev)
    
    a1_mov_avg_vol_sum_buy = moving_average(a1_vol_sum_buy,window_size)
    data_a1_mov_avg_vol_sum_buy.append(a1_mov_avg_vol_sum_buy)
    a1_mov_avg_vol_min_buy = moving_average(a1_vol_min_buy,window_size)
    data_a1_mov_avg_vol_min_buy.append(a1_mov_avg_vol_min_buy)
    a1_mov_avg_vol_max_buy = moving_average(a1_vol_max_buy,window_size)
    data_a1_mov_avg_vol_max_buy.append(a1_mov_avg_vol_max_buy)
    a1_mov_avg_vol_sum_sell = moving_average(a1_vol_sum_sell,window_size)
    data_a1_mov_avg_vol_sum_sell.append(a1_mov_avg_vol_sum_sell)
    a1_mov_avg_vol_min_sell = moving_average(a1_vol_min_sell,window_size)
    data_a1_mov_avg_vol_min_sell.append(a1_mov_avg_vol_min_sell)
    a1_mov_avg_vol_max_sell = moving_average(a1_vol_max_sell,window_size)
    data_a1_mov_avg_vol_max_sell.append(a1_mov_avg_vol_max_sell)
    a1_mov_avg_vol_mean_sell = moving_average(a1_vol_mean_sell,window_size)
    data_a1_mov_avg_vol_mean_sell.append(a1_mov_avg_vol_mean_sell)
    a1_mov_avg_vol_mean_buy = moving_average(a1_vol_mean_buy,window_size)
    data_a1_mov_avg_vol_mean_buy.append(a1_mov_avg_vol_mean_buy)
    a1_mov_avg_vol_median_sell = moving_average(a1_vol_median_sell,window_size)
    data_a1_mov_avg_vol_median_sell.append(a1_mov_avg_vol_median_sell)
    a1_mov_avg_vol_median_buy = moving_average(a1_vol_median_buy,window_size)
    data_a1_mov_avg_vol_median_buy.append(a1_mov_avg_vol_median_buy)
    a1_mov_avg_vol_buy_stddev = moving_average(a1_vol_buy_stddev,window_size)
    data_a1_mov_avg_vol_buy_stddev.append(a1_mov_avg_vol_buy_stddev)
    a1_mov_avg_vol_sell_stddev = moving_average(a1_vol_sell_stddev,window_size)
    data_a1_mov_avg_vol_sell_stddev.append(a1_mov_avg_vol_sell_stddev)

    a1_mov_avg_cumulative_sum_buy = moving_average(a1_cumulative_sum_buy,window_size)
    data_a1_mov_avg_cumulative_sum_buy.append(a1_mov_avg_cumulative_sum_buy)
    a1_mov_avg_cumulative_min_buy = moving_average(a1_cumulative_min_buy,window_size)
    data_a1_mov_avg_cumulative_min_buy.append(a1_mov_avg_cumulative_min_buy)
    a1_mov_avg_cumulative_max_buy = moving_average(a1_cumulative_max_buy,window_size)
    data_a1_mov_avg_cumulative_max_buy.append(a1_mov_avg_cumulative_max_buy)
    a1_mov_avg_cumulative_sum_sell = moving_average(a1_cumulative_sum_sell,window_size)
    data_a1_mov_avg_cumulative_sum_sell.append(a1_mov_avg_cumulative_sum_sell)
    a1_mov_avg_cumulative_min_sell = moving_average(a1_cumulative_min_sell,window_size)
    data_a1_mov_avg_cumulative_min_sell.append(a1_mov_avg_cumulative_min_sell)
    a1_mov_avg_cumulative_max_sell = moving_average(a1_cumulative_max_sell,window_size)
    data_a1_mov_avg_cumulative_max_sell.append(a1_mov_avg_cumulative_max_sell)
    a1_mov_avg_cumulative_mean_sell = moving_average(a1_cumulative_mean_sell,window_size)
    data_a1_mov_avg_cumulative_mean_sell.append(a1_mov_avg_cumulative_mean_sell)
    a1_mov_avg_cumulative_mean_buy = moving_average(a1_cumulative_mean_buy,window_size)
    data_a1_mov_avg_cumulative_mean_buy.append(a1_mov_avg_cumulative_mean_buy)
    a1_mov_avg_cumulative_median_sell = moving_average(a1_cumulative_median_sell,window_size)
    data_a1_mov_avg_cumulative_median_sell.append(a1_mov_avg_cumulative_median_sell)
    a1_mov_avg_cumulative_median_buy = moving_average(a1_cumulative_median_buy,window_size)
    data_a1_mov_avg_cumulative_median_buy.append(a1_mov_avg_cumulative_median_buy)
    a1_mov_avg_cumulative_buy_stddev = moving_average(a1_cumulative_buy_stddev,window_size)
    data_a1_mov_avg_cumulative_buy_stddev.append(a1_mov_avg_cumulative_buy_stddev)
    a1_mov_avg_cumulative_sell_stddev = moving_average(a1_cumulative_sell_stddev,window_size)
    data_a1_mov_avg_cumulative_sell_stddev.append(a1_mov_avg_cumulative_sell_stddev)

    a1_mov_avg_cumulative_vol_sum_buy = moving_average(a1_cumulative_vol_sum_buy,window_size)
    data_a1_mov_avg_cumulative_vol_sum_buy.append(a1_mov_avg_cumulative_vol_sum_buy)
    a1_mov_avg_cumulative_vol_min_buy = moving_average(a1_cumulative_vol_min_buy,window_size)
    data_a1_mov_avg_cumulative_vol_min_buy.append(a1_mov_avg_cumulative_vol_min_buy)
    a1_mov_avg_cumulative_vol_max_buy = moving_average(a1_cumulative_vol_max_buy,window_size)
    data_a1_mov_avg_cumulative_vol_max_buy.append(a1_mov_avg_cumulative_vol_max_buy)
    a1_mov_avg_cumulative_vol_sum_sell = moving_average(a1_cumulative_vol_sum_sell,window_size)
    data_a1_mov_avg_cumulative_vol_sum_sell.append(a1_mov_avg_cumulative_vol_sum_sell)
    a1_mov_avg_cumulative_vol_min_sell = moving_average(a1_cumulative_vol_min_sell,window_size)
    data_a1_mov_avg_cumulative_vol_min_sell.append(a1_mov_avg_cumulative_vol_min_sell)
    a1_mov_avg_cumulative_vol_max_sell = moving_average(a1_cumulative_vol_max_sell,window_size)
    data_a1_mov_avg_cumulative_vol_max_sell.append(a1_mov_avg_cumulative_vol_max_sell)
    a1_mov_avg_cumulative_vol_mean_sell = moving_average(a1_cumulative_vol_mean_sell,window_size)
    data_a1_mov_avg_cumulative_vol_mean_sell.append(a1_mov_avg_cumulative_vol_mean_sell)
    a1_mov_avg_cumulative_vol_mean_buy = moving_average(a1_cumulative_vol_mean_buy,window_size)
    data_a1_mov_avg_cumulative_vol_mean_buy.append(a1_mov_avg_cumulative_vol_mean_buy)
    a1_mov_avg_cumulative_vol_median_sell = moving_average(a1_cumulative_vol_median_sell,window_size)
    data_a1_mov_avg_cumulative_vol_median_sell.append(a1_mov_avg_cumulative_vol_median_sell)
    a1_mov_avg_cumulative_vol_median_buy = moving_average(a1_cumulative_vol_median_buy,window_size)
    data_a1_mov_avg_cumulative_vol_median_buy.append(a1_mov_avg_cumulative_vol_median_buy)
    a1_mov_avg_cumulative_vol_buy_stddev = moving_average(a1_cumulative_vol_buy_stddev,window_size)
    data_a1_mov_avg_cumulative_vol_buy_stddev.append(a1_mov_avg_cumulative_vol_buy_stddev)
    a1_mov_avg_cumulative_vol_sell_stddev = moving_average(a1_cumulative_vol_sell_stddev,window_size)
    data_a1_mov_avg_cumulative_vol_sell_stddev.append(a1_mov_avg_cumulative_vol_sell_stddev)

    print_accuracy([malicious_a1_sum_buy],[a1_sum_buy],j,"SUM BUY",a1_timestamps)
    print_accuracy([malicious_a1_sum_sell],[a1_sum_sell],j,"SUM SELL",a1_timestamps)
    print_accuracy([malicious_a1_min_buy],[a1_min_buy],j,"SUM MIN BUY",a1_timestamps_buy)
    print_accuracy([malicious_a1_max_buy],[a1_max_buy],j,"MAX BUY",a1_timestamps_buy)
    print_accuracy([malicious_a1_min_sell],[a1_min_sell],j,"MIN SELL",a1_timestamps_sell)
    print_accuracy([malicious_a1_max_sell],[a1_max_sell],j,"MAX SELL",a1_timestamps_sell)
    print_accuracy([malicious_a1_mean_sell],[a1_mean_sell],j,"MEAN SELL",a1_timestamps_sell)
    print_accuracy([malicious_a1_mean_buy],[a1_mean_buy],j,"MEAN BUY",a1_timestamps_buy)
    print_accuracy([malicious_a1_median_sell],[a1_median_sell],j,"MEDIAN SELL",a1_timestamps_sell)
    print_accuracy([malicious_a1_median_buy],[a1_median_buy],j,"MEDIAN BUY",a1_timestamps_buy)
    print_accuracy([malicious_a1_stddev_sell],[a1_sell_stddev],j,"STDEV SELL",a1_timestamps_buy_stddev)
    print_accuracy([malicious_a1_stddev_buy],[a1_buy_stddev],j,"STDEV BUY",a1_timestamps_sell_stddev)

    print_accuracy([malicious_a1_sum_buy_vol],[a1_vol_sum_buy],j,"VOL SUM BUY",a1_timestamps)
    print_accuracy([malicious_a1_sum_sell_vol],[a1_vol_sum_sell],j,"VOL SUM SELL",a1_timestamps)
    print_accuracy([malicious_a1_min_buy_vol],[a1_vol_min_buy],j,"VOL MIN BUY",a1_vol_timestamps_buy)
    print_accuracy([malicious_a1_max_buy_vol],[a1_vol_max_buy],j,"VOL MAX BUY",a1_vol_timestamps_buy)
    print_accuracy([malicious_a1_min_sell_vol],[a1_vol_min_sell],j,"VOL MIN SELL",a1_vol_timestamps_sell)
    print_accuracy([malicious_a1_max_sell_vol],[a1_vol_max_sell],j,"VOL MAX SELL",a1_vol_timestamps_sell)
    print_accuracy([malicious_a1_mean_sell_vol],[a1_vol_mean_sell],j,"VOL MEAN SELL",a1_vol_timestamps_sell)
    print_accuracy([malicious_a1_mean_buy_vol],[a1_vol_mean_buy],j,"VOL MEAN BUY",a1_vol_timestamps_buy)
    print_accuracy([malicious_a1_median_sell_vol],[a1_vol_median_sell],j,"VOL MEDIAN SELL",a1_vol_timestamps_sell)
    print_accuracy([malicious_a1_median_buy_vol],[a1_vol_median_buy],j,"VOL MEDIAN BUY",a1_vol_timestamps_buy)
    print_accuracy([malicious_a1_stddev_sell_vol],[a1_vol_sell_stddev],j,"VOL STDEV SELL",a1_vol_timestamps_sell_stddev)
    print_accuracy([malicious_a1_stddev_buy_vol],[a1_vol_stddev_buy],j,"VOL STDEV BUY",a1_vol_timestamps_buy_stddev)

    print_accuracy([malicious_a1_cumulative_sum_buy],[a1_cumulative_sum_buy],j,"CUMULATIVE SUM BUY",a1_timestamps)
    print_accuracy([malicious_a1_cumulative_sum_sell],[a1_cumulative_sum_sell],j,"CUMULATIVE SUM SELL",a1_timestamps)
    print_accuracy([malicious_a1_cumulative_min_buy],[a1_cumulative_min_buy],j,"CUMULATIVE MIN BUY",a1_timestamps_buy)
    print_accuracy([malicious_a1_cumulative_min_sell],[a1_cumulative_min_sell],j,"CUMULATIVE MIN SELL",a1_timestamps_buy)
    print_accuracy([malicious_a1_cumulative_max_buy],[a1_cumulative_max_buy],j,"CUMULATIVE MAX BUY",a1_timestamps_sell)
    print_accuracy([malicious_a1_cumulative_max_sell],[a1_cumulative_max_sell],j,"CUMULATIVE MAX SELL",a1_timestamps_sell)
    print_accuracy([malicious_a1_cumulative_median_buy],[a1_cumulative_median_buy],j,"CUMULATIVE MEDIAN BUY",a1_timestamps_buy)
    print_accuracy([malicious_a1_cumulative_median_sell],[a1_cumulative_median_sell],j,"CUMULATIVE MEDIAN SELL",a1_timestamps_buy)
    print_accuracy([malicious_a1_cumulative_buy_stddev],[a1_cumulative_buy_stddev],j,"CUMULATIVE STDEV BUY",a1_buy_stddev)
    print_accuracy([malicious_a1_cumulative_sell_stddev],[a1_cumulative_sell_stddev],j,"CUMULATIVE STDEV SELL",a1_timestamps_sell_stddev)

    print_accuracy([malicious_a1_cumulative_vol_sum_buy],[a1_cumulative_vol_sum_buy],j,"CUMULATIVE VOL SUM BUY",a1_timestamps)
    print_accuracy([malicious_a1_cumulative_vol_sum_sell],[a1_cumulative_vol_sum_sell],j,"CUMULATIVE VOL SUM SELL",a1_timestamps)
    print_accuracy([malicious_a1_cumulative_vol_min_buy],[a1_cumulative_vol_min_buy],j,"CUMULATIVE VOL MIN BUY",a1_vol_timestamps_buy)
    print_accuracy([malicious_a1_cumulative_vol_min_sell],[a1_cumulative_vol_min_sell],j,"CUMULATIVE VOL MIN SELL",a1_vol_timestamps_sell)
    print_accuracy([malicious_a1_cumulative_vol_max_buy],[a1_cumulative_vol_max_buy],j,"CUMULATIVE VOL MAX BUY",a1_vol_timestamps_buy)
    print_accuracy([malicious_a1_cumulative_vol_max_sell],[a1_cumulative_vol_max_sell],j,"CUMULATIVE VOL MAX SELL",a1_vol_timestamps_sell)
    print_accuracy([malicious_a1_cumulative_vol_median_buy],[a1_cumulative_vol_median_buy],j,"CUMULATIVE VOL MEDIAN BUY",a1_vol_timestamps_buy)
    print_accuracy([malicious_a1_cumulative_vol_median_sell],[a1_cumulative_vol_median_sell],j,"CUMULATIVE VOL MEDIAN SELL",a1_vol_timestamps_sell)
    print_accuracy([malicious_a1_cumulative_vol_buy_stddev],[a1_cumulative_vol_buy_stddev],j,"CUMULATIVE VOL STDEV BUY",a1_vol_timestamps_buy_stddev)
    print_accuracy([malicious_a1_cumulative_vol_sell_stddev],[a1_cumulative_vol_sell_stddev],j,"CUMULATIVE VOL STDEV SELL",a1_vol_timestamps_sell_stddev)

    same_timestamps1=[a1_sum_buy,a1_sum_sell,a1_vol_sum_buy,a1_vol_sum_sell,a1_cumulative_sum_buy,a1_cumulative_sum_sell,a1_cumulative_vol_sum_buy,a1_cumulative_vol_sum_sell]
    labels1=["SUM BUY","SUM SELL","VOL SUM BUY","VOL SUM SELL","CUM. SUM BUY","CUM. SUM SELL","CUM. VOL SUM BUY","CUM. VOL SUM SELL"]

    for i in range(len(same_timestamps1)):
        if i==0:
            train=same_timestamps1[1:]
            # print(train.shape)
            labels=','.join(labels1[1:])
        else:
            train=same_timestamps1[:i]+same_timestamps1[i+1:]
            # train=np.array(same_timestamps1[:i]+same_timestamps1[i+1:])
            # print(train.shape)
            labels=','.join(labels1[:i]+labels1[i+1:])
        print_accuracy1([train],j,labels,a1_timestamps)

    same_timestamps1=[a1_min_buy,a1_max_buy,a1_mean_buy,a1_median_buy,a1_cumulative_min_buy,a1_cumulative_max_buy,a1_cumulative_mean_buy,a1_cumulative_median_buy]
    labels1=["MIN BUY","MAX BUY","MEAN BUY","MEDIAN BUY","CUM. MIN BUY","CUM. MAX BUY","CUM. MEAN BUY","CUM. MEDIAN BUY"]

    for i in range(len(same_timestamps1)):
        if i==0:
            train=same_timestamps1[1:]
            # print(train.shape)
            labels=','.join(labels1[1:])
        else:
            train=same_timestamps1[:i]+same_timestamps1[i+1:]
            # train=np.array(same_timestamps1[:i]+same_timestamps1[i+1:])
            # print(train.shape)
            labels=','.join(labels1[:i]+labels1[i+1:])
        print_accuracy1([train],j,labels,a1_timestamps_buy)

    same_timestamps1=[a1_min_sell,a1_max_sell,a1_mean_sell,a1_median_sell,a1_cumulative_min_sell,a1_cumulative_max_sell,a1_cumulative_mean_sell,a1_cumulative_median_sell]
    labels1=["MIN SELL","MAX SELL","MEAN SELL","MEDIAN SELL","CUM. MIN SELL","CUM. MAX SELL","CUM. MEAN SELL","CUM. MEDIAN SELL"]

    for i in range(len(same_timestamps1)):
        if i==0:
            train=same_timestamps1[1:]
            # print(train.shape)
            labels=','.join(labels1[1:])
        else:
            train=same_timestamps1[:i]+same_timestamps1[i+1:]
            # train=np.array(same_timestamps1[:i]+same_timestamps1[i+1:])
            # print(train.shape)
            labels=','.join(labels1[:i]+labels1[i+1:])
        print_accuracy1([train],j,labels,a1_timestamps_sell)

    same_timestamps1=[a1_vol_min_buy,a1_vol_max_buy,a1_vol_mean_buy,a1_vol_median_buy,a1_cumulative_vol_min_buy,a1_cumulative_vol_max_buy,a1_cumulative_vol_mean_buy,a1_cumulative_vol_median_buy]
    labels1=["VOL MIN BUY","VOL MAX BUY","VOL MEAN BUY","VOL MEDIAN BUY","CUM. VOL MIN BUY","CUM. VOL MAX BUY","CUM. VOL MEAN BUY","CUM. VOL MEDIAN BUY"]

    for i in range(len(same_timestamps1)):
        if i==0:
            train=same_timestamps1[1:]
            # print(train.shape)
            labels=','.join(labels1[1:])
        else:
            train=same_timestamps1[:i]+same_timestamps1[i+1:]
            # train=np.array(same_timestamps1[:i]+same_timestamps1[i+1:])
            # print(train.shape)
            labels=','.join(labels1[:i]+labels1[i+1:])
        print_accuracy1([train],j,labels,a1_vol_timestamps_buy)

    same_timestamps1=[a1_vol_min_sell,a1_vol_max_sell,a1_vol_mean_sell,a1_vol_median_sell,a1_cumulative_vol_min_sell,a1_cumulative_vol_max_sell,a1_cumulative_vol_mean_sell,a1_cumulative_vol_median_sell]
    labels1=["VOL MIN SELL","VOL MAX SELL","VOL MEAN SELL","VOL MEDIAN SELL","CUM. VOL MIN SELL","CUM. VOL MAX SELL","CUM. VOL MEAN SELL","CUM. VOL MEDIAN SELL"]

    for i in range(len(same_timestamps1)):
        if i==0:
            train=same_timestamps1[1:]
            # print(train.shape)
            labels=','.join(labels1[1:])
        else:
            train=same_timestamps1[:i]+same_timestamps1[i+1:]
            # train=np.array(same_timestamps1[:i]+same_timestamps1[i+1:])
            # print(train.shape)
            labels=','.join(labels1[:i]+labels1[i+1:])
        print_accuracy1([train],j,labels,a1_vol_timestamps_sell)

print("\n\n","IMPORTANT FEATURES","\n\n")
imp_features=list(set(imp_features))
comb=[]
for i in imp_features:
    print(i)
    comb+=i.split(',')
comb=sorted(list(set(comb)))
print("\n\n","THEY INCLUDE","\n\n")
for i in comb:
    print(i)