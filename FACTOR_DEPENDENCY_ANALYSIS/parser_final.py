import matplotlib.pyplot as plt
from statistics import mean,stdev,median
import csv

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
        plt.plot(x,y)
        plt.title(tit)
        plt.xlabel(strx)
        plt.ylabel(stry)
        plt.xticks(rotation=90)
        plt.savefig(tit.replace(' ','') + '.png')
        # plt.show()

def draw2(y,x,strx,stry,tit):
        plt.plot(x,y)
        plt.title(tit)
        plt.xlabel(strx)
        plt.ylabel(stry)
        plt.xticks(rotation=90)
        plt.savefig(tit.replace(' ','') + '.png')

def draw_hist(x,strx,stry,tit):
        plt.title(tit)
        plt.xlabel(strx)
        plt.ylabel(stry)
        plt.xticks(rotation=90)
        plt.hist(x,bins=20)
        plt.savefig(tit.replace(' ','') + '_hist.png')

with open('file_pure.csv', 'r') as f:
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
    keys=list(trader_timestamp_dict.keys())
    keys.sort()
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


    for i in keys:
        if i[1]=='T_3':
            a1_timestamps.append(i[0])
            a1_sum_buy.append(sum(trader_timestamp_dict[i]['buying']['price']))
            a1_sum_sell.append(sum(trader_timestamp_dict[i]['selling']['price']))
            if  len((trader_timestamp_dict[i]['buying']['price']))>0:
                a1_timestamps_buy.append(i[0])
                a1_mean_buy.append(mean(trader_timestamp_dict[i]['buying']['price']))
                a1_median_buy.append(median(trader_timestamp_dict[i]['buying']['price']))
                a1_min_buy.append(min(trader_timestamp_dict[i]['buying']['price']))
                a1_max_buy.append(max(trader_timestamp_dict[i]['buying']['price']))
            if  len((trader_timestamp_dict[i]['selling']['price']))>0:
                a1_timestamps_sell.append(i[0])
                a1_mean_sell.append(mean(trader_timestamp_dict[i]['selling']['price']))
                a1_median_sell.append(median(trader_timestamp_dict[i]['selling']['price']))
                a1_min_sell.append(min(trader_timestamp_dict[i]['selling']['price']))
                a1_max_sell.append(max(trader_timestamp_dict[i]['selling']['price']))  
            if  len((trader_timestamp_dict[i]['buying']['price']))>1:
                a1_timestamps_buy_stddev.append(i[0])
                a1_buy_stddev.append(stdev(trader_timestamp_dict[i]['buying']['price']))
            if  len((trader_timestamp_dict[i]['selling']['price']))>1:
                a1_timestamps_sell_stddev.append(i[0])
                a1_sell_stddev.append(stdev(trader_timestamp_dict[i]['selling']['price']))

            a1_vol_sum_buy.append(sum(trader_timestamp_dict[i]['buying']['volume']))
            a1_vol_sum_sell.append(sum(trader_timestamp_dict[i]['selling']['volume']))
            if  len((trader_timestamp_dict[i]['buying']['volume']))>0:
                a1_vol_timestamps_buy.append(i[0])
                a1_vol_mean_buy.append(mean(trader_timestamp_dict[i]['buying']['volume']))
                a1_vol_median_buy.append(median(trader_timestamp_dict[i]['buying']['volume']))
                a1_vol_min_buy.append(min(trader_timestamp_dict[i]['buying']['volume']))
                a1_vol_max_buy.append(max(trader_timestamp_dict[i]['buying']['volume']))
            if  len((trader_timestamp_dict[i]['selling']['volume']))>0:
                a1_vol_timestamps_sell.append(i[0])
                a1_vol_mean_sell.append(mean(trader_timestamp_dict[i]['selling']['volume']))
                a1_vol_median_sell.append(median(trader_timestamp_dict[i]['selling']['volume']))
                a1_vol_min_sell.append(min(trader_timestamp_dict[i]['selling']['volume']))
                a1_vol_max_sell.append(max(trader_timestamp_dict[i]['selling']['volume']))  
            if  len((trader_timestamp_dict[i]['buying']['volume']))>1:
                a1_vol_timestamps_buy_stddev.append(i[0])
                a1_vol_buy_stddev.append(stdev(trader_timestamp_dict[i]['buying']['volume']))
            if  len((trader_timestamp_dict[i]['selling']['volume']))>1:
                a1_vol_timestamps_sell_stddev.append(i[0])
                a1_vol_sell_stddev.append(stdev(trader_timestamp_dict[i]['selling']['volume']))


    #     try:
    #         mean_buy_price=mean(trader_timestamp_dict[i]['buying']['price'])
    #         mean_buy_volume=mean(trader_timestamp_dict[i]['buying']['volume'])
    #         print(*i,"Buying Mean Price, Volume: ",mean_buy_price,mean_buy_volume)
    #     except:
    #         pass
    #         # print(trader_timestamp_dict[i]['buying']['price'])
    #     try:
    #         stdev_buy_price=stdev(trader_timestamp_dict[i]['buying']['price'])
    #         stdev_buy_volume=stdev(trader_timestamp_dict[i]['buying']['volume'])
    #         print(*i,"Buying Stdev Price, Volume: ",stdev_buy_price,stdev_buy_volume)
    #     except:
    #         pass
    #         # print(trader_timestamp_dict[i]['buying']['volume'])
    #     try:
    #         mean_sell_price=mean(trader_timestamp_dict[i]['selling']['price'])
    #         mean_sell_volume=mean(trader_timestamp_dict[i]['selling']['volume'])
    #         print(*i,"Selling Mean Price, Volume: ",mean_sell_price,mean_sell_volume)
    #     except:
    #         pass
    #         # print(trader_timestamp_dict[i]['selling']['price'])
    #     try:            
    #         stdev_sell_price=stdev(trader_timestamp_dict[i]['selling']['price'])
    #         stdev_sell_volume=stdev(trader_timestamp_dict[i]['selling']['volume'])
    #         print(*i,"Selling Stdev Price, Volume: ",stdev_sell_price,stdev_sell_volume)
    #     except:
    #         pass
    #         # print(trader_timestamp_dict[i]['selling']['volume'])

    #     print(*i,"Buying orders number: ",len(trader_timestamp_dict[i]['buying']['price']),"Selling orders number: ",len(trader_timestamp_dict[i]['selling']['price']))
    

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



    # GRAPHS

    # 4 Graphs of sum
    draw(a1_timestamps, a1_sum_buy , "time stamps" , "Total Buying capacity of T_3" , "Total Buying capacity of a trader w.r.t. time")
    draw(a1_timestamps, a1_sum_sell , "time stamps" , "Total selling capacity of T_3" , "Total selling capacity of a trader w.r.t. time")
    draw(a1_timestamps, a1_vol_sum_buy , "time stamps" , "Total Buying volume of T_3" , "Total Buying volume of a trader w.r.t. time")
    draw(a1_timestamps, a1_vol_sum_sell , "time stamps" , "Total selling volume of T_3" , "Total selling volume of a trader w.r.t. time")

    # 4 Graphs of std-dev
    draw(a1_timestamps_buy_stddev, a1_buy_stddev , "time stamps" , "Std. deviation of buying price" , "Std. deviation of buying price of a trader w.r.t. time")
    draw(a1_timestamps_sell_stddev, a1_sell_stddev , "time stamps" , "Std. deviation of selling price" , "Std. deviation of selling price of a trader w.r.t. time")
    draw(a1_vol_timestamps_buy_stddev, a1_vol_buy_stddev , "time stamps" , "Std. deviation of buying volume" , "Std. deviation of buying volume of a trader w.r.t. time")
    draw(a1_vol_timestamps_sell_stddev, a1_vol_sell_stddev , "time stamps" , "Std. deviation of selling volume" , "Std. deviation of selling volume of a trader w.r.t. time")

    # 12 Graphs of Cumulative {Mean , Standard Deviation , Sum)-----> (Think about - Moving average}
    
    draw(a1_timestamps, a1_cumulative_sum_buy, "time stamps", "Cumulative sum of T_3" , "Cumulative sum of buying prices of a trader w.r.t time stamps")
    draw(a1_timestamps, a1_cumulative_sum_sell, "time stamps", "Cumulative sum of T_3" , "Cumulative sum of selling prices of a trader w.r.t time stamps")
    draw(a1_timestamps, a1_cumulative_vol_sum_buy, "time stamps", "Cumulative sum of T_3" , "Cumulative sum of buying volumes of a trader w.r.t time stamps")

    draw(a1_timestamps_buy, a1_cumulative_mean_buy, "time stamps", "Cumulative mean of T_3" , "Cumulative mean of buying prices of a trader w.r.t time stamps")
    draw(a1_timestamps, a1_cumulative_vol_sum_sell, "time stamps", "Cumulative sum of T_3" , "Cumulative sum of selling volumes of a trader w.r.t time stamps")
    draw(a1_timestamps_buy, a1_cumulative_vol_mean_buy, "time stamps", "Cumulative mean of T_3" , "Cumulative mean of buying volume of a trader w.r.t time stamps")
    draw(a1_timestamps_sell, a1_cumulative_mean_sell, "time stamps", "Cumulative mean of T_3" , "Cumulative mean of selling prices of a trader w.r.t time stamps")
    draw(a1_timestamps_sell, a1_cumulative_vol_mean_sell, "time stamps", "Cumulative mean of T_3" , "Cumulative mean of selling volume of a trader w.r.t time stamps")

    draw(a1_timestamps_buy_stddev, a1_cumulative_buy_stddev, "time stamps", "Cumulative std-dev of T_3" , "Cumulative std-dev of buying prices of a trader w.r.t time stamps")
    draw(a1_timestamps_buy_stddev, a1_cumulative_vol_buy_stddev, "time stamps", "Cumulative std-dev of T_3" , "Cumulative std-dev of buying volume of a trader w.r.t time stamps")
    draw(a1_timestamps_sell_stddev, a1_cumulative_sell_stddev, "time stamps", "Cumulative std-dev of T_3" , "Cumulative std-dev of selling prices of a trader w.r.t time stamps")
    draw(a1_timestamps_sell_stddev, a1_cumulative_vol_sell_stddev, "time stamps", "Cumulative std-dev of T_3" , "Cumulative std-dev of selling volume of a trader w.r.t time stamps")
    
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
    plt.plot(a1_timestamps_buy,a1_min_buy)
    plt.plot(a1_timestamps_buy,a1_max_buy)
    plt.plot(a1_timestamps_buy,a1_median_buy)
    plt.title("statistics of buying prices of trader w.r.t time stamps")
    plt.xlabel("time stamps")
    plt.ylabel("price")
    plt.legend(['mean of buying prices', 'minimum buying prices', 'maximum buying prices' , 'median of buying prices'], loc='lower right')
    plt.xticks(rotation=90)
    plt.savefig('BuyingPrices.png')
    # plt.show()


    # Plotting selling price statistics
    plt.plot(a1_timestamps_sell,a1_mean_sell)
    plt.plot(a1_timestamps_sell,a1_min_sell)
    plt.plot(a1_timestamps_sell,a1_max_sell)
    plt.plot(a1_timestamps_sell,a1_median_sell)
    plt.title("statistics of selling prices of trader w.r.t time stamps")
    plt.xlabel("time stamps")
    plt.ylabel("price")
    plt.legend(['mean of selling prices', 'minimum buying prices', 'maximum buying prices' , 'median of selling prices'], loc='lower right')
    plt.tight_layout()
    plt.xticks(rotation=90)
    # plt.show()
    plt.savefig('SellingPrices.png')

    # Plotting Buying Volume Statistics
    plt.plot(a1_vol_timestamps_buy,a1_vol_mean_buy)
    plt.plot(a1_vol_timestamps_buy,a1_vol_min_buy)
    plt.plot(a1_vol_timestamps_buy,a1_vol_max_buy)
    plt.plot(a1_vol_timestamps_buy,a1_vol_median_buy)
    plt.title("statistics of buying volume of trader w.r.t time stamps")
    plt.xlabel("time stamps")
    plt.ylabel("volume")
    plt.legend(['mean of buying volumes', 'minimum buying volumes', 'maximum buying volumes' , 'median of buying volumes'], loc='lower right')
    plt.xticks(rotation=90)
    # plt.show()
    plt.savefig('BuyingVolume.png')


    # Plotting selling Volume statistics
    plt.plot(a1_vol_timestamps_sell,a1_vol_mean_sell)
    plt.plot(a1_vol_timestamps_sell,a1_vol_min_sell)
    plt.plot(a1_vol_timestamps_sell,a1_vol_max_sell)
    plt.plot(a1_vol_timestamps_sell,a1_vol_median_sell)
    plt.title("statistics of selling volume of trader w.r.t time stamps")
    plt.xlabel("time stamps")
    plt.ylabel("volume")
    plt.legend(['mean of selling volumes', 'minimum buying volumes', 'maximum buying volumes' , 'median of selling volumes'], loc='lower right')
    plt.tight_layout()
    plt.xticks(rotation=90)
    # plt.show()
    plt.savefig('SellingVolume.png')


    # HISTOGRAMS

    # Plotting Buying Price Statistics
    plt.hist(a1_mean_buy,bins=20)
    plt.title("statistics of buying prices of trader w.r.t time stamps")
    plt.xlabel("time stamps")
    plt.ylabel("price")
    plt.legend(['mean of buying prices', 'minimum buying prices', 'maximum buying prices' , 'median of buying prices'], loc='lower right')
    plt.xticks(rotation=90)
    plt.savefig('BuyingPricesMean_hist.png')

    plt.hist(a1_min_buy,bins=20)
    plt.title("statistics of buying prices of trader w.r.t time stamps")
    plt.xlabel("time stamps")
    plt.ylabel("price")
    plt.legend(['mean of buying prices', 'minimum buying prices', 'maximum buying prices' , 'median of buying prices'], loc='lower right')
    plt.xticks(rotation=90)
    plt.savefig('BuyingPricesMin_hist.png')

    plt.hist(a1_max_buy,bins=20)
    plt.title("statistics of buying prices of trader w.r.t time stamps")
    plt.xlabel("time stamps")
    plt.ylabel("price")
    plt.legend(['mean of buying prices', 'minimum buying prices', 'maximum buying prices' , 'median of buying prices'], loc='lower right')
    plt.xticks(rotation=90)
    plt.savefig('BuyingPricesMax_hist.png')

    plt.hist(a1_median_buy,bins=20)
    plt.title("statistics of buying prices of trader w.r.t time stamps")
    plt.xlabel("time stamps")
    plt.ylabel("price")
    plt.legend(['mean of buying prices', 'minimum buying prices', 'maximum buying prices' , 'median of buying prices'], loc='lower right')
    plt.xticks(rotation=90)
    plt.savefig('BuyingPricesMedian_hist.png')

    # Plotting selling price statistics
    plt.hist(a1_mean_sell,bins=20)
    plt.title("statistics of selling prices of trader w.r.t time stamps")
    plt.xlabel("time stamps")
    plt.ylabel("price")
    plt.legend(['mean of selling prices', 'minimum selling prices', 'maximum selling prices' , 'median of selling prices'], loc='lower right')
    plt.xticks(rotation=90)
    plt.savefig('SellingPricesMean_hist.png')

    plt.hist(a1_min_sell,bins=20)
    plt.title("statistics of selling prices of trader w.r.t time stamps")
    plt.xlabel("time stamps")
    plt.ylabel("price")
    plt.legend(['mean of selling prices', 'minimum selling prices', 'maximum selling prices' , 'median of selling prices'], loc='lower right')
    plt.xticks(rotation=90)
    plt.savefig('SellingPricesMin_hist.png')

    plt.hist(a1_max_sell,bins=20)
    plt.title("statistics of selling prices of trader w.r.t time stamps")
    plt.xlabel("time stamps")
    plt.ylabel("price")
    plt.legend(['mean of selling prices', 'minimum selling prices', 'maximum selling prices' , 'median of selling prices'], loc='lower right')
    plt.xticks(rotation=90)
    plt.savefig('SellingPricesMax_hist.png')

    plt.hist(a1_median_sell,bins=20)
    plt.title("statistics of selling prices of trader w.r.t time stamps")
    plt.xlabel("time stamps")
    plt.ylabel("price")
    plt.legend(['mean of selling prices', 'minimum selling prices', 'maximum selling prices' , 'median of selling prices'], loc='lower right')
    plt.xticks(rotation=90)
    plt.savefig('SellingPricesMedian_hist.png')

    # Plotting Buying Volume Statistics
    plt.hist(a1_vol_mean_buy,bins=20)
    plt.title("statistics of buying volumes of trader w.r.t time stamps")
    plt.xlabel("time stamps")
    plt.ylabel("volume")
    plt.legend(['mean of buying volumes', 'minimum buying volumes', 'maximum buying volumes' , 'median of buying volumes'], loc='lower right')
    plt.xticks(rotation=90)
    plt.savefig('BuyingVolumesMean_hist.png')

    plt.hist(a1_vol_min_buy,bins=20)
    plt.title("statistics of buying volumes of trader w.r.t time stamps")
    plt.xlabel("time stamps")
    plt.ylabel("volume")
    plt.legend(['mean of buying volumes', 'minimum buying volumes', 'maximum buying volumes' , 'median of buying volumes'], loc='lower right')
    plt.xticks(rotation=90)
    plt.savefig('BuyingVolumesMin_hist.png')

    plt.hist(a1_vol_max_buy,bins=20)
    plt.title("statistics of buying volumes of trader w.r.t time stamps")
    plt.xlabel("time stamps")
    plt.ylabel("volume")
    plt.legend(['mean of buying volumes', 'minimum buying volumes', 'maximum buying volumes' , 'median of buying volumes'], loc='lower right')
    plt.xticks(rotation=90)
    plt.savefig('BuyingVolumesMax_hist.png')

    plt.hist(a1_vol_median_buy,bins=20)
    plt.title("statistics of buying volumes of trader w.r.t time stamps")
    plt.xlabel("time stamps")
    plt.ylabel("volume")
    plt.legend(['mean of buying volumes', 'minimum buying volumes', 'maximum buying volumes' , 'median of buying volumes'], loc='lower right')
    plt.xticks(rotation=90)
    plt.savefig('BuyingVolumesMedian_hist.png')


    # Plotting selling Volume statistics
    plt.hist(a1_vol_mean_sell,bins=20)
    plt.title("statistics of selling volumes of trader w.r.t time stamps")
    plt.xlabel("time stamps")
    plt.ylabel("volume")
    plt.legend(['mean of selling volumes', 'minimum selling volumes', 'maximum selling volumes' , 'median of selling volumes'], loc='lower right')
    plt.xticks(rotation=90)
    plt.savefig('SellingVolumesMean_hist.png')

    plt.hist(a1_vol_min_sell,bins=20)
    plt.title("statistics of selling volumes of trader w.r.t time stamps")
    plt.xlabel("time stamps")
    plt.ylabel("volume")
    plt.legend(['mean of selling volumes', 'minimum selling volumes', 'maximum selling volumes' , 'median of selling volumes'], loc='lower right')
    plt.xticks(rotation=90)
    plt.savefig('SellingVolumesMin_hist.png')

    plt.hist(a1_vol_max_sell,bins=20)
    plt.title("statistics of selling volumes of trader w.r.t time stamps")
    plt.xlabel("time stamps")
    plt.ylabel("volume")
    plt.legend(['mean of selling volumes', 'minimum selling volumes', 'maximum selling volumes' , 'median of selling volumes'], loc='lower right')
    plt.xticks(rotation=90)
    plt.savefig('SellingVolumesMax_hist.png')

    plt.hist(a1_vol_median_sell,bins=20)
    plt.title("statistics of selling volumes of trader w.r.t time stamps")
    plt.xlabel("time stamps")
    plt.ylabel("volume")
    plt.legend(['mean of selling volumes', 'minimum selling volumes', 'maximum selling volumes' , 'median of selling volumes'], loc='lower right')
    plt.xticks(rotation=90)
    plt.savefig('SellingVolumesMedian_hist.png')


    window_size = 5

    a1_mov_avg_sum_buy = moving_average(a1_sum_buy,window_size)
    a1_mov_avg_min_buy = moving_average(a1_min_buy,window_size)
    a1_mov_avg_max_buy = moving_average(a1_max_buy,window_size)
    a1_mov_avg_sum_sell = moving_average(a1_sum_sell,window_size)
    a1_mov_avg_min_sell = moving_average(a1_min_sell,window_size)
    a1_mov_avg_max_sell = moving_average(a1_max_sell,window_size)
    a1_mov_avg_mean_sell = moving_average(a1_mean_sell,window_size)
    a1_mov_avg_mean_buy = moving_average(a1_mean_buy,window_size)
    a1_mov_avg_median_sell = moving_average(a1_median_sell,window_size)
    a1_mov_avg_median_buy = moving_average(a1_median_buy,window_size)
    a1_mov_avg_buy_stddev = moving_average(a1_buy_stddev,window_size)
    a1_mov_avg_sell_stddev = moving_average(a1_sell_stddev,window_size)
    
    a1_mov_avg_vol_sum_buy = moving_average(a1_vol_sum_buy,window_size)
    a1_mov_avg_vol_min_buy = moving_average(a1_vol_min_buy,window_size)
    a1_mov_avg_vol_max_buy = moving_average(a1_vol_max_buy,window_size)
    a1_mov_avg_vol_sum_sell = moving_average(a1_vol_sum_sell,window_size)
    a1_mov_avg_vol_min_sell = moving_average(a1_vol_min_sell,window_size)
    a1_mov_avg_vol_max_sell = moving_average(a1_vol_max_sell,window_size)
    a1_mov_avg_vol_mean_sell = moving_average(a1_vol_mean_sell,window_size)
    a1_mov_avg_vol_mean_buy = moving_average(a1_vol_mean_buy,window_size)
    a1_mov_avg_vol_median_sell = moving_average(a1_vol_median_sell,window_size)
    a1_mov_avg_vol_median_buy = moving_average(a1_vol_median_buy,window_size)
    a1_mov_avg_vol_buy_stddev = moving_average(a1_vol_buy_stddev,window_size)
    a1_mov_avg_vol_sell_stddev = moving_average(a1_vol_sell_stddev,window_size)

    a1_mov_avg_cumulative_sum_buy = moving_average(a1_cumulative_sum_buy,window_size)
    a1_mov_avg_cumulative_min_buy = moving_average(a1_cumulative_min_buy,window_size)
    a1_mov_avg_cumulative_max_buy = moving_average(a1_cumulative_max_buy,window_size)
    a1_mov_avg_cumulative_sum_sell = moving_average(a1_cumulative_sum_sell,window_size)
    a1_mov_avg_cumulative_min_sell = moving_average(a1_cumulative_min_sell,window_size)
    a1_mov_avg_cumulative_max_sell = moving_average(a1_cumulative_max_sell,window_size)
    a1_mov_avg_cumulative_mean_sell = moving_average(a1_cumulative_mean_sell,window_size)
    a1_mov_avg_cumulative_mean_buy = moving_average(a1_cumulative_mean_buy,window_size)
    a1_mov_avg_cumulative_median_sell = moving_average(a1_cumulative_median_sell,window_size)
    a1_mov_avg_cumulative_median_buy = moving_average(a1_cumulative_median_buy,window_size)
    a1_mov_avg_cumulative_buy_stddev = moving_average(a1_cumulative_buy_stddev,window_size)
    a1_mov_avg_cumulative_sell_stddev = moving_average(a1_cumulative_sell_stddev,window_size)

    a1_mov_avg_cumulative_vol_sum_buy = moving_average(a1_cumulative_vol_sum_buy,window_size)
    a1_mov_avg_cumulative_vol_min_buy = moving_average(a1_cumulative_vol_min_buy,window_size)
    a1_mov_avg_cumulative_vol_max_buy = moving_average(a1_cumulative_vol_max_buy,window_size)
    a1_mov_avg_cumulative_vol_sum_sell = moving_average(a1_cumulative_vol_sum_sell,window_size)
    a1_mov_avg_cumulative_vol_min_sell = moving_average(a1_cumulative_vol_min_sell,window_size)
    a1_mov_avg_cumulative_vol_max_sell = moving_average(a1_cumulative_vol_max_sell,window_size)
    a1_mov_avg_cumulative_vol_mean_sell = moving_average(a1_cumulative_vol_mean_sell,window_size)
    a1_mov_avg_cumulative_vol_mean_buy = moving_average(a1_cumulative_vol_mean_buy,window_size)
    a1_mov_avg_cumulative_vol_median_sell = moving_average(a1_cumulative_vol_median_sell,window_size)
    a1_mov_avg_cumulative_vol_median_buy = moving_average(a1_cumulative_vol_median_buy,window_size)
    a1_mov_avg_cumulative_vol_buy_stddev = moving_average(a1_cumulative_vol_buy_stddev,window_size)
    a1_mov_avg_cumulative_vol_sell_stddev = moving_average(a1_cumulative_vol_sell_stddev,window_size)


#   GRAPHS

    draw2(a1_mov_avg_sum_buy , list(range(len(a1_mov_avg_sum_buy))), " " , "a1_mov_avg_sum_buy of T_3" , "a1_mov_avg_sum_buy of the trader")
    draw2(a1_mov_avg_min_buy , list(range(len(a1_mov_avg_min_buy))), " " , "a1_mov_avg_min_buy of T_3" , "a1_mov_avg_min_buy of the trader")
    draw2(a1_mov_avg_max_buy , list(range(len(a1_mov_avg_max_buy))), " " , "a1_mov_avg_max_buy of T_3" , "a1_mov_avg_max_buy of the trader")
    draw2(a1_mov_avg_sum_sell , list(range(len(a1_mov_avg_sum_sell))), " " , "a1_mov_avg_sum_sell of T_3" , "a1_mov_avg_sum_sell of the trader")
    draw2(a1_mov_avg_min_sell , list(range(len(a1_mov_avg_min_sell))), " " , "a1_mov_avg_min_sell of T_3" , "a1_mov_avg_min_sell of the trader")
    draw2(a1_mov_avg_max_sell , list(range(len(a1_mov_avg_max_sell))), " " , "a1_mov_avg_max_sell of T_3" , "a1_mov_avg_max_sell of the trader")
    draw2(a1_mov_avg_mean_sell , list(range(len(a1_mov_avg_mean_sell))), " " , "a1_mov_avg_mean_sell of T_3" , "a1_mov_avg_mean_sell of the trader")
    draw2(a1_mov_avg_mean_buy , list(range(len(a1_mov_avg_mean_buy))), " " , "a1_mov_avg_mean_buy of T_3" , "a1_mov_avg_mean_buy of the trader")
    draw2(a1_mov_avg_median_sell , list(range(len(a1_mov_avg_median_sell))), " " , "a1_mov_avg_median_sell of T_3" , "a1_mov_avg_median_sell of the trader")
    draw2(a1_mov_avg_median_buy , list(range(len(a1_mov_avg_median_buy))), " " , "a1_mov_avg_median_buy of T_3" , "a1_mov_avg_median_buy of the trader")
    draw2(a1_mov_avg_buy_stddev , list(range(len(a1_mov_avg_buy_stddev))), " " , "a1_mov_avg_buy_stddev of T_3" , "a1_mov_avg_buy_stddev of the trader")
    draw2(a1_mov_avg_sell_stddev , list(range(len(a1_mov_avg_sell_stddev))), " " , "a1_mov_avg_sell_stddev of T_3" , "a1_mov_avg_sell_stddev of the trader")

    draw2(a1_mov_avg_vol_sum_buy , list(range(len(a1_mov_avg_vol_sum_buy))), " " , "a1_mov_avg_vol_sum_buy of T_3", "a1_mov_avg_vol_sum_buy of the trader" )
    draw2(a1_mov_avg_vol_min_buy , list(range(len(a1_mov_avg_vol_min_buy))), " " , "a1_mov_avg_vol_min_buy of T_3", "a1_mov_avg_vol_min_buy of the trader" )
    draw2(a1_mov_avg_vol_max_buy , list(range(len(a1_mov_avg_vol_max_buy))), " " , "a1_mov_avg_vol_max_buy of T_3", "a1_mov_avg_vol_max_buy of the trader" )
    draw2(a1_mov_avg_vol_sum_sell , list(range(len(a1_mov_avg_vol_sum_sell))), " " , "a1_mov_avg_vol_sum_sell of T_3", "a1_mov_avg_vol_sum_sell of the trader" )
    draw2(a1_mov_avg_vol_min_sell , list(range(len(a1_mov_avg_vol_min_sell))), " " , "a1_mov_avg_vol_min_sell of T_3", "a1_mov_avg_vol_min_sell of the trader" )
    draw2(a1_mov_avg_vol_max_sell , list(range(len(a1_mov_avg_vol_max_sell))), " " , "a1_mov_avg_vol_max_sell of T_3", "a1_mov_avg_vol_max_sell of the trader" )
    draw2(a1_mov_avg_vol_mean_sell , list(range(len(a1_mov_avg_vol_mean_sell))), " " , "a1_mov_avg_vol_mean_sell of T_3", "a1_mov_avg_vol_mean_sell of the trader" )
    draw2(a1_mov_avg_vol_mean_buy , list(range(len(a1_mov_avg_vol_mean_buy))), " " , "a1_mov_avg_vol_mean_buy of T_3", "a1_mov_avg_vol_mean_buy of the trader" )
    draw2(a1_mov_avg_vol_median_sell , list(range(len(a1_mov_avg_vol_median_sell))), " " , "a1_mov_avg_vol_median_sell of T_3", "a1_mov_avg_vol_median_sell of the trader" )
    draw2(a1_mov_avg_vol_median_buy , list(range(len(a1_mov_avg_vol_median_buy))), " " , "a1_mov_avg_vol_median_buy of T_3", "a1_mov_avg_vol_median_buy of the trader" )
    draw2(a1_mov_avg_vol_buy_stddev , list(range(len(a1_mov_avg_vol_buy_stddev))), " " , "a1_mov_avg_vol_buy_stddev of T_3", "a1_mov_avg_vol_buy_stddev of the trader" )
    draw2(a1_mov_avg_vol_sell_stddev , list(range(len(a1_mov_avg_vol_sell_stddev))), " " , "a1_mov_avg_vol_sell_stddev of T_3", "a1_mov_avg_vol_sell_stddev of the trader" )

    draw2(a1_mov_avg_cumulative_sum_buy , list(range(len(a1_mov_avg_cumulative_sum_buy))), " " , "a1_mov_avg_cumulative_sum_buy of T_3", "a1_mov_avg_cumulative_sum_buy of the trader" )
    draw2(a1_mov_avg_cumulative_min_buy , list(range(len(a1_mov_avg_cumulative_min_buy))), " " , "a1_mov_avg_cumulative_min_buy of T_3", "a1_mov_avg_cumulative_min_buy of the trader" )
    draw2(a1_mov_avg_cumulative_max_buy , list(range(len(a1_mov_avg_cumulative_max_buy))), " " , "a1_mov_avg_cumulative_max_buy of T_3", "a1_mov_avg_cumulative_max_buy of the trader" )
    draw2(a1_mov_avg_cumulative_sum_sell , list(range(len(a1_mov_avg_cumulative_sum_sell))), " " , "a1_mov_avg_cumulative_sum_sell of T_3", "a1_mov_avg_cumulative_sum_sell of the trader" )
    draw2(a1_mov_avg_cumulative_min_sell , list(range(len(a1_mov_avg_cumulative_min_sell))), " " , "a1_mov_avg_cumulative_min_sell of T_3", "a1_mov_avg_cumulative_min_sell of the trader" )
    draw2(a1_mov_avg_cumulative_max_sell , list(range(len(a1_mov_avg_cumulative_max_sell))), " " , "a1_mov_avg_cumulative_max_sell of T_3", "a1_mov_avg_cumulative_max_sell of the trader" )
    draw2(a1_mov_avg_cumulative_mean_sell , list(range(len(a1_mov_avg_cumulative_mean_sell))), " " , "a1_mov_avg_cumulative_mean_sell of T_3", "a1_mov_avg_cumulative_mean_sell of the trader" )
    draw2(a1_mov_avg_cumulative_mean_buy , list(range(len(a1_mov_avg_cumulative_mean_buy))), " " , "a1_mov_avg_cumulative_mean_buy of T_3", "a1_mov_avg_cumulative_mean_buy of the trader" )
    draw2(a1_mov_avg_cumulative_median_sell , list(range(len(a1_mov_avg_cumulative_median_sell))), " " , "a1_mov_avg_cumulative_median_sell of T_3", "a1_mov_avg_cumulative_median_sell of the trader" )
    draw2(a1_mov_avg_cumulative_median_buy , list(range(len(a1_mov_avg_cumulative_median_buy))), " " , "a1_mov_avg_cumulative_median_buy of T_3", "a1_mov_avg_cumulative_median_buy of the trader" )
    draw2(a1_mov_avg_cumulative_buy_stddev , list(range(len(a1_mov_avg_cumulative_buy_stddev))), " " , "a1_mov_avg_cumulative_buy_stddev of T_3", "a1_mov_avg_cumulative_buy_stddev of the trader" )
    draw2(a1_mov_avg_cumulative_sell_stddev , list(range(len(a1_mov_avg_cumulative_sell_stddev))), " " , "a1_mov_avg_cumulative_sell_stddev of T_3", "a1_mov_avg_cumulative_sell_stddev of the trader" )

    draw2(a1_mov_avg_cumulative_vol_sum_buy , list(range(len(a1_mov_avg_cumulative_vol_sum_buy))), " " , "a1_mov_avg_cumulative_vol_sum_buy of T_3", "a1_mov_avg_cumulative_vol_sum_buy of the trader" )
    draw2(a1_mov_avg_cumulative_vol_min_buy , list(range(len(a1_mov_avg_cumulative_vol_min_buy))), " " , "a1_mov_avg_cumulative_vol_min_buy of T_3", "a1_mov_avg_cumulative_vol_min_buy of the trader" )
    draw2(a1_mov_avg_cumulative_vol_max_buy , list(range(len(a1_mov_avg_cumulative_vol_max_buy))), " " , "a1_mov_avg_cumulative_vol_max_buy of T_3", "a1_mov_avg_cumulative_vol_max_buy of the trader" )
    draw2(a1_mov_avg_cumulative_vol_sum_sell , list(range(len(a1_mov_avg_cumulative_vol_sum_sell))), " " , "a1_mov_avg_cumulative_vol_sum_sell of T_3", "a1_mov_avg_cumulative_vol_sum_sell of the trader" )
    draw2(a1_mov_avg_cumulative_vol_min_sell , list(range(len(a1_mov_avg_cumulative_vol_min_sell))), " " , "a1_mov_avg_cumulative_vol_min_sell of T_3", "a1_mov_avg_cumulative_vol_min_sell of the trader" )
    draw2(a1_mov_avg_cumulative_vol_max_sell , list(range(len(a1_mov_avg_cumulative_vol_max_sell))), " " , "a1_mov_avg_cumulative_vol_max_sell of T_3", "a1_mov_avg_cumulative_vol_max_sell of the trader" )
    draw2(a1_mov_avg_cumulative_vol_mean_sell , list(range(len(a1_mov_avg_cumulative_vol_mean_sell))), " " , "a1_mov_avg_cumulative_vol_mean_sell of T_3", "a1_mov_avg_cumulative_vol_mean_sell of the trader" )
    draw2(a1_mov_avg_cumulative_vol_mean_buy , list(range(len(a1_mov_avg_cumulative_vol_mean_buy))), " " , "a1_mov_avg_cumulative_vol_mean_buy of T_3", "a1_mov_avg_cumulative_vol_mean_buy of the trader" )
    draw2(a1_mov_avg_cumulative_vol_median_sell , list(range(len(a1_mov_avg_cumulative_vol_median_sell))), " " , "a1_mov_avg_cumulative_vol_median_sell of T_3", "a1_mov_avg_cumulative_vol_median_sell of the trader" )
    draw2(a1_mov_avg_cumulative_vol_median_buy , list(range(len(a1_mov_avg_cumulative_vol_median_buy))), " " , "a1_mov_avg_cumulative_vol_median_buy of T_3", "a1_mov_avg_cumulative_vol_median_buy of the trader" )
    draw2(a1_mov_avg_cumulative_vol_buy_stddev , list(range(len(a1_mov_avg_cumulative_vol_buy_stddev))), " " , "a1_mov_avg_cumulative_vol_buy_stddev of T_3", "a1_mov_avg_cumulative_vol_buy_stddev of the trader" )
    draw2(a1_mov_avg_cumulative_vol_sell_stddev , list(range(len(a1_mov_avg_cumulative_vol_sell_stddev))), " " , "a1_mov_avg_cumulative_vol_sell_stddev of T_3", "a1_mov_avg_cumulative_vol_sell_stddev of the trader" )
    
#   HISTOGRAM

    draw_hist(a1_mov_avg_sum_buy ," " , "a1_mov_avg_sum_buy of T_3" , "a1_mov_avg_sum_buy of the trader")
    draw_hist(a1_mov_avg_min_buy ," " , "a1_mov_avg_min_buy of T_3" , "a1_mov_avg_min_buy of the trader")
    draw_hist(a1_mov_avg_max_buy ," " , "a1_mov_avg_max_buy of T_3" , "a1_mov_avg_max_buy of the trader")
    draw_hist(a1_mov_avg_sum_sell, " " , "a1_mov_avg_sum_sell of T_3" , "a1_mov_avg_sum_sell of the trader")
    draw_hist(a1_mov_avg_min_sell , " " , "a1_mov_avg_min_sell of T_3" , "a1_mov_avg_min_sell of the trader")
    draw_hist(a1_mov_avg_max_sell , " " , "a1_mov_avg_max_sell of T_3" , "a1_mov_avg_max_sell of the trader")
    draw_hist(a1_mov_avg_mean_sell , " " , "a1_mov_avg_mean_sell of T_3" , "a1_mov_avg_mean_sell of the trader")
    draw_hist(a1_mov_avg_mean_buy ,  " " , "a1_mov_avg_mean_buy of T_3" , "a1_mov_avg_mean_buy of the trader")
    draw_hist(a1_mov_avg_median_sell , " " , "a1_mov_avg_median_sell of T_3" , "a1_mov_avg_median_sell of the trader")
    draw_hist(a1_mov_avg_median_buy ,  " " , "a1_mov_avg_median_buy of T_3" , "a1_mov_avg_median_buy of the trader")
    draw_hist(a1_mov_avg_buy_stddev ,  " " , "a1_mov_avg_buy_stddev of T_3" , "a1_mov_avg_buy_stddev of the trader")
    draw_hist(a1_mov_avg_sell_stddev , " " , "a1_mov_avg_sell_stddev of T_3" , "a1_mov_avg_sell_stddev of the trader")



    draw_hist(a1_mov_avg_vol_sum_buy , " " , "a1_mov_avg_vol_sum_buy of T_3", "a1_mov_avg_vol_sum_buy of the trader" )
    draw_hist(a1_mov_avg_vol_min_buy , " " , "a1_mov_avg_vol_min_buy of T_3", "a1_mov_avg_vol_min_buy of the trader" )
    draw_hist(a1_mov_avg_vol_max_buy , " " , "a1_mov_avg_vol_max_buy of T_3", "a1_mov_avg_vol_max_buy of the trader" )
    draw_hist(a1_mov_avg_vol_sum_sell , " " , "a1_mov_avg_vol_sum_sell of T_3", "a1_mov_avg_vol_sum_sell of the trader" )
    draw_hist(a1_mov_avg_vol_min_sell , " " , "a1_mov_avg_vol_min_sell of T_3", "a1_mov_avg_vol_min_sell of the trader" )
    draw_hist(a1_mov_avg_vol_max_sell , " " , "a1_mov_avg_vol_max_sell of T_3", "a1_mov_avg_vol_max_sell of the trader" )
    draw_hist(a1_mov_avg_vol_mean_sell , " " , "a1_mov_avg_vol_mean_sell of T_3", "a1_mov_avg_vol_mean_sell of the trader" )
    draw_hist(a1_mov_avg_vol_mean_buy ,  " " , "a1_mov_avg_vol_mean_buy of T_3", "a1_mov_avg_vol_mean_buy of the trader" )
    draw_hist(a1_mov_avg_vol_median_sell , " " , "a1_mov_avg_vol_median_sell of T_3", "a1_mov_avg_vol_median_sell of the trader" )
    draw_hist(a1_mov_avg_vol_median_buy ,  " " , "a1_mov_avg_vol_median_buy of T_3", "a1_mov_avg_vol_median_buy of the trader" )
    draw_hist(a1_mov_avg_vol_buy_stddev ,  " " , "a1_mov_avg_vol_buy_stddev of T_3", "a1_mov_avg_vol_buy_stddev of the trader" )
    draw_hist(a1_mov_avg_vol_sell_stddev , " " , "a1_mov_avg_vol_sell_stddev of T_3", "a1_mov_avg_vol_sell_stddev of the trader" )

    draw_hist(a1_mov_avg_cumulative_sum_buy , " " , "a1_mov_avg_cumulative_sum_buy of T_3", "a1_mov_avg_cumulative_sum_buy of the trader" )
    draw_hist(a1_mov_avg_cumulative_min_buy , " " , "a1_mov_avg_cumulative_min_buy of T_3", "a1_mov_avg_cumulative_min_buy of the trader" )
    draw_hist(a1_mov_avg_cumulative_max_buy , " " , "a1_mov_avg_cumulative_max_buy of T_3", "a1_mov_avg_cumulative_max_buy of the trader" )
    draw_hist(a1_mov_avg_cumulative_sum_sell , " " , "a1_mov_avg_cumulative_sum_sell of T_3", "a1_mov_avg_cumulative_sum_sell of the trader" )
    draw_hist(a1_mov_avg_cumulative_min_sell , " " , "a1_mov_avg_cumulative_min_sell of T_3", "a1_mov_avg_cumulative_min_sell of the trader" )
    draw_hist(a1_mov_avg_cumulative_max_sell , " " , "a1_mov_avg_cumulative_max_sell of T_3", "a1_mov_avg_cumulative_max_sell of the trader" )
    draw_hist(a1_mov_avg_cumulative_mean_sell , " " , "a1_mov_avg_cumulative_mean_sell of T_3", "a1_mov_avg_cumulative_mean_sell of the trader" )
    draw_hist(a1_mov_avg_cumulative_mean_buy ,  " " , "a1_mov_avg_cumulative_mean_buy of T_3", "a1_mov_avg_cumulative_mean_buy of the trader" )
    draw_hist(a1_mov_avg_cumulative_median_sell , " " , "a1_mov_avg_cumulative_median_sell of T_3", "a1_mov_avg_cumulative_median_sell of the trader" )
    draw_hist(a1_mov_avg_cumulative_median_buy ,  " " , "a1_mov_avg_cumulative_median_buy of T_3", "a1_mov_avg_cumulative_median_buy of the trader" )
    draw_hist(a1_mov_avg_cumulative_buy_stddev ,  " " , "a1_mov_avg_cumulative_buy_stddev of T_3", "a1_mov_avg_cumulative_buy_stddev of the trader" )
    draw_hist(a1_mov_avg_cumulative_sell_stddev , " " , "a1_mov_avg_cumulative_sell_stddev of T_3", "a1_mov_avg_cumulative_sell_stddev of the trader" )

    draw_hist(a1_mov_avg_cumulative_vol_sum_buy , " " , "a1_mov_avg_cumulative_vol_sum_buy of T_3", "a1_mov_avg_cumulative_vol_sum_buy of the trader" )
    draw_hist(a1_mov_avg_cumulative_vol_min_buy , " " , "a1_mov_avg_cumulative_vol_min_buy of T_3", "a1_mov_avg_cumulative_vol_min_buy of the trader" )
    draw_hist(a1_mov_avg_cumulative_vol_max_buy , " " , "a1_mov_avg_cumulative_vol_max_buy of T_3", "a1_mov_avg_cumulative_vol_max_buy of the trader" )
    draw_hist(a1_mov_avg_cumulative_vol_sum_sell , " " , "a1_mov_avg_cumulative_vol_sum_sell of T_3", "a1_mov_avg_cumulative_vol_sum_sell of the trader" )
    draw_hist(a1_mov_avg_cumulative_vol_min_sell , " " , "a1_mov_avg_cumulative_vol_min_sell of T_3", "a1_mov_avg_cumulative_vol_min_sell of the trader" )
    draw_hist(a1_mov_avg_cumulative_vol_max_sell , " " , "a1_mov_avg_cumulative_vol_max_sell of T_3", "a1_mov_avg_cumulative_vol_max_sell of the trader" )
    draw_hist(a1_mov_avg_cumulative_vol_mean_sell , " " , "a1_mov_avg_cumulative_vol_mean_sell of T_3", "a1_mov_avg_cumulative_vol_mean_sell of the trader" )
    draw_hist(a1_mov_avg_cumulative_vol_mean_buy ,  " " , "a1_mov_avg_cumulative_vol_mean_buy of T_3", "a1_mov_avg_cumulative_vol_mean_buy of the trader" )
    draw_hist(a1_mov_avg_cumulative_vol_median_sell , " " , "a1_mov_avg_cumulative_vol_median_sell of T_3", "a1_mov_avg_cumulative_vol_median_sell of the trader" )
    draw_hist(a1_mov_avg_cumulative_vol_median_buy ,  " " , "a1_mov_avg_cumulative_vol_median_buy of T_3", "a1_mov_avg_cumulative_vol_median_buy of the trader" )
    draw_hist(a1_mov_avg_cumulative_vol_buy_stddev ,  " " , "a1_mov_avg_cumulative_vol_buy_stddev of T_3", "a1_mov_avg_cumulative_vol_buy_stddev of the trader" )
    draw_hist(a1_mov_avg_cumulative_vol_sell_stddev , " " , "a1_mov_avg_cumulative_vol_sell_stddev of T_3", "a1_mov_avg_cumulative_vol_sell_stddev of the trader" )
    
