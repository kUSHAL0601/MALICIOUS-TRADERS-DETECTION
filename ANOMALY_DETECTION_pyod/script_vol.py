import csv
import numpy as np
from pyod.models.ocsvm import OCSVM

d_attack={}
d_transaction={}
traders=set()
with open('attack.csv', 'r') as f1:
	reader1 = list(csv.reader(f1))
	reader1.pop(0)
	for i in reader1:
		with open('message.csv', 'r') as f:
			reader = list(csv.reader(f))
			trader_timestamp_dict={}
			for row in range(1,len(reader)):
				entry=reader[row]
				# time_stamp=entry[1][:-7]
				time_stamp=entry[1]
				entry_type=entry[2]
				order_id=entry[3]
				price=float(entry[4])
				volume=float(entry[5])
				direction=int(entry[6])
				trader_id=entry[7]
				stock_id=entry[8]
				order_level=entry[9]
				matched_order_trader_id=entry[10]
				match_price=entry[11]
				match_volume=entry[12]
				match_timestamp=entry[13]
				traders.add(trader_id)
				try:
					if direction==1:
						d_transaction[trader_id].append((time_stamp,-price,volume))
					elif direction==-1:
						d_transaction[trader_id].append((time_stamp,price,volume))
				except:
					if direction==1:
						d_transaction[trader_id]=[(time_stamp,-price,volume)]
					elif direction==-1:
						d_transaction[trader_id]=[(time_stamp,price,volume)]
				if order_id==i[1]:
					try:
						d_attack[trader_id].append(time_stamp)
					except:
						d_attack[trader_id]=[time_stamp]

print("MALICIOUS TRADERS:",*sorted(list(d_attack.keys())))
for i in traders:
	if i not in d_attack:
		d_attack[i]=[]
	if i not in d_transaction:
		d_transaction[i]=[]

# print(d_attack)

for i in traders:
	# count1=0
	# true1=0
	# count2=0
	# true2=0
	# count3=0
	# true3=0
	# initial_set=d_transaction[i][:10]
	# data_vol=[i[2] for i in initial_set]
	# data_price=[i[1] for i in initial_set]
	# clf1=OCSVM()
	# clf2=OCSVM()
	# clf12=OCSVM()
	# clf1.fit(np.array([data_vol]).T)
	# clf2.fit(np.array([data_price]).T)
	# clf12.fit(np.array([data_vol,data_price]).T)
	# for j in range(11,len(d_transaction[i])):
	# 	p1=clf1.predict(np.array([d_transaction[i][j][2]]).reshape(1,-1))
	# 	p2=clf2.predict(np.array([d_transaction[i][j][1]]).reshape(1,-1))
	# 	p3=clf12.predict(np.array([d_transaction[i][j][2],d_transaction[i][j][1]]).T.reshape(1,-1))
	# 	if p1==1:
	# 		print(i,d_transaction[i][j][0])
	# 		count1+=1
	# 		if d_transaction[i][j][0] in d_attack[trader_id]:
	# 			true1+=1
	# 	if p2==1:
	# 		count2+=1
	# 		if d_transaction[i][j][0] in d_attack[trader_id]:
	# 			true2+=1
	# 	if p3==1:
	# 		count3+=1
	# 		if d_transaction[i][j][0] in d_attack[trader_id]:
	# 			true3+=1
	# 	data_vol.append(d_transaction[i][j][2])
	# 	data_price.append(d_transaction[i][j][1])
	# 	clf1.fit(np.array([data_vol]).T)
	# 	clf2.fit(np.array([data_price]).T)
	# 	clf12.fit(np.array([data_vol,data_price]).T)
	# print("TRADER",i,true1,count1,true1/count1,true2,count2,true2/count2,true3,count3,true3/count3)
	data_vol=[]
	data_price=[]
	data_vol_price=[]
	mal_t_stamps1=[]
	mal_t_stamps2=[]
	mal_t_stamps12=[]
	clf1=OCSVM()
	clf2=OCSVM()
	clf12=OCSVM()
	for j in d_transaction[i]:
		data_vol.append(j[2])
		data_price.append(j[1])
		data_vol_price.append([j[2],j[1]])
	clf1.fit(np.array([data_vol]).T)
	clf2.fit(np.array([data_price]).T)
	clf12.fit(np.array(data_vol_price))
	for j in d_transaction[i]:
		p1=clf1.predict(np.array(j[2]).reshape(1,-1))
		p2=clf2.predict(np.array(j[2]).reshape(1,-1))
		p3=clf12.predict(np.array([j[2],j[1]]).T.reshape(1,-1))
		if p1==1:
			mal_t_stamps1.append(j[0])
		if p2==1:
			mal_t_stamps2.append(j[0])
		if p3==1:
			mal_t_stamps12.append(j[0])
	s=set(d_attack[i])
	print("TRADER",i,"VOL",len(s&set(mal_t_stamps1)),"OUT OF",len(mal_t_stamps1),"PRICE",len(s&set(mal_t_stamps2)),"OUT OF",len(mal_t_stamps2),"VOL AND PRICE",len(s&set(mal_t_stamps12)),"OUT OF",len(mal_t_stamps12))
