from statistics import mean,stdev,median
import csv
import numpy as np
from random import shuffle

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

positives=[]
negatives=[]

with open('message.csv', 'r') as f:
    reader = list(csv.reader(f))
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

        if int(direction)==-1 and int(entry_type) == 1:
            price*=-1
        if order_id in malicious_ids:
            negatives.append([price, volume])
        else:
            positives.append([price, volume])
# print(positives,negatives)
indices_positives=list(range(len(positives)))
indices_negatives=list(range(len(negatives)))
shuffle(indices_negatives)
shuffle(indices_positives)

train_len_positive=int(len(positives)*0.8)
test_len_positive=len(positives)-train_len_positive
train_len_negative=int(len(negatives)*0.8)
test_len_negative=len(negatives)-train_len_negative

train_set=[]
pred_train_set=[]
test_set=[]
pred_test_set=[]
# from pyod.utils.data import generate_data
# X_train, y_train, X_test, y_test = \
#         generate_data(n_train=1000,
#                       n_test=100,
#                       n_features=2,
#                       contamination=0.1,
#                       random_state=42)
# print(X_train.shape,X_test.shape)

for i in range(train_len_positive):
    train_set.append(positives[indices_positives[i]])
    pred_train_set.append(0)
for i in range(train_len_positive,len(positives)):
    test_set.append(positives[indices_positives[i]])
    pred_test_set.append(0)

for i in range(train_len_negative):
    train_set.append(negatives[indices_negatives[i]])
    pred_train_set.append(1)
for i in range(train_len_negative,len(negatives)):
    test_set.append(negatives[indices_negatives[i]])
    pred_test_set.append(1)
# print(len(train_set),len(test_set))
import numpy as np
train_set=np.array(train_set)
test_set=np.array(test_set)
# print(train_set.shape,test_set.shape)
# # train_set=train_set.T
# # test_set=test_set.T

# from pyod.models.auto_encoder import AutoEncoder
# from pyod.utils.data import evaluate_print

# clf_name='AutoEncoder'
# clf = AutoEncoder(epochs=30)
# clf.fit(train_set)

# y_test_pred = clf.predict(train_set)
# y_test_scores = clf.decision_function(train_set)

# y_test_pred = clf.predict(test_set)
# y_test_scores = clf.decision_function(test_set)

# print("\nOn Training Data:")
# evaluate_print(clf_name, pred_train_set, y_train_scores)
# print("\nOn Test Data:")
# evaluate_print(clf_name, pred_test_set, y_test_scores)

from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA

clf1=PCA()
clf2=OCSVM()

clf1.fit(train_set)
clf2.fit(train_set)

y_pred_train_pca=clf1.predict(train_set)
y_pred_test_pca=clf1.predict(test_set)

y_pred_train_ocsvm=clf2.predict(train_set)
y_pred_test_ocsvm=clf2.predict(test_set)

# print(y_pred_test_pca,y_pred_test_ocsvm)
train_pca_correct=0
train_ocsvm_correct=0
print("TRAIN SET")
for i in range(len(pred_train_set)):
    print("Actual:",pred_train_set[i],"PCA",y_pred_train_pca[i],"OCSVM",y_pred_train_ocsvm[i])
    if pred_train_set[i]==y_pred_train_pca[i] and pred_train_set[i]==1:
        train_pca_correct+=1
    if pred_train_set[i]==y_pred_train_ocsvm[i] and y_pred_train_ocsvm[i]==1:
        train_ocsvm_correct+=1

test_pca_correct=0
test_ocsvm_correct=0
print("TEST SET")
for i in range(len(pred_test_set)):
    print("Actual:",pred_test_set[i],"PCA",y_pred_test_pca[i],"OCSVM",y_pred_test_ocsvm[i])
    if(pred_test_set[i]==y_pred_test_pca[i] and y_pred_test_pca[i]==1):
        test_pca_correct+=1
    if(pred_test_set[i]==y_pred_test_ocsvm[i] and y_pred_test_ocsvm[i]==1):
        test_ocsvm_correct+=1
print(train_len_negative,train_pca_correct,train_ocsvm_correct,test_len_negative,test_pca_correct,test_ocsvm_correct)
