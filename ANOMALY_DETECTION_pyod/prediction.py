import matplotlib.pyplot as plt
from statistics import mean,stdev,median
import csv

data_a1_buy_stddev=[]
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


with open('attack.csv', 'r') as f:
    reader = list(csv.reader(f))
    reader.pop(0)
    malicious_ids=[]
    for i in reader:
        malicious_ids.append(i[1])



    
orders = []  # contains all the order ids will help in getting indices for malicious orders
traders_val = []
price_val = []
dir_val = []
vol_val = []
time_stamp_val = []


with open('message.csv', 'r') as f:
    reader = list(csv.reader(f))
    trader_timestamp_dict={}
    for row in range(1,len(reader)):
        entry=reader[row]
        time_stamp=entry[1]
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
        # print(time_stamp,direction,trader_id)
        orders.append(order_id)
        traders_val.append(trader_id)
        price_val.append(price)
        dir_val.append(direction)
        vol_val.append(volume)
        time_stamp_val.append(time_stamp)
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
    keys=list(trader_timestamp_dict.keys())
    keys.sort()




indices = []  # indices of all malicious ids
mal_traders = []
mal_price = []
mal_volume = []
mal_direction = []
mal_timestamps = [] 
for val in malicious_ids:
    if val in orders:
        indices.append(orders.index(val))
for i in indices:
    mal_traders.append(traders_val[i])    
    mal_price.append(price_val[i])
    mal_volume.append(vol_val[i])
    mal_direction.append(dir_val[i])
    mal_timestamps.append(time_stamp_val[i])


# for i in range(10):
#     print(mal_timestamps[i])     






















