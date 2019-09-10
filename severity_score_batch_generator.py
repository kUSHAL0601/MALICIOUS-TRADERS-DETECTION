## code works without computing keys

import matplotlib.pyplot as plt
from statistics import mean,stdev,median
import csv
import math
import numpy as np
# from pyod.models.ocsvm import OCSVM
# from pyod.models.pca import PCA 
from sklearn.metrics import roc_auc_score
from sklearn.svm import OneClassSVM as oc_svm 
import pickle


filename = 'classifier.sav' ## Load the classifier
clf1 = pickle.load(open(filename, 'rb')) 
roc = []
recall_list = []

predicted1 = clf1.predict(test_all_data_stnd)

score_test = clf1.score_samples(test_all_data_stnd) ## classification score on new data
#print("min score:%f and max score:%f"%(min(score_test),max(score_test)))
min_score_test = min(score_test)
max_score_test = max(score_test)
# print(predicted)
#print(clf1.score_samples(test_all_data_stnd))


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


# roc.append(roc_auc_score(predicted,test_labels))
# recall_list.append(recall)
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
# fpr, tpr, _ = roc_curve(test_labels,predicted)
# roc_auc = auc(fpr, tpr)
# print(fpr,tpr,_)
# plt.figure()
# lw = 2
# plt.plot(fpr, tpr, color='darkorange',
#          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate / Recall(here)')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()

score1 = clf1.score_samples(train_normal_arr_stnd)
score2 = clf1.score_samples(test_all_data_stnd)
print(score1.reshape(len(score1),1).shape)
score = np.vstack((score1.reshape(len(score1),1),score2.reshape(len(score2),1)))

# score = clf1.score_samples(test_all_data_stnd)
severity = []
for val in score:
    # print (np.round((1.8 - val)*10**4,2))
    severity.append(sigmoid(np.round((set_threshold - val),2)))
severity_cto = []

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

print(all_feature_vector.shape)
print(np.asarray(severity).reshape(len(severity),1).shape)
feature_write = np.hstack((all_trader_timestamp,all_feature_vector,severity.reshape(len(severity),1),np.asarray(all_labels).reshape(len(all_labels),1)))




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
    with open('features_rbf_per_sec/featute_'+val+'.csv','w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['trader', 'timestamp', 'a1_sum_buy','a1_mean_buy','a1_median_buy','a1_min_buy','a1_max_buy','a1_vol_sum_buy','a1_vol_mean_buy','a1_vol_median_buy','a1_vol_min_buy','a1_vol_max_buy',' a1_sum_sell','a1_mean_sell','a1_median_sell','a1_min_sell','a1_max_sell','a1_vol_sum_sell','a1_vol_mean_sell','a1_vol_median_sell','a1_vol_min_sell','a1_vol_max_sell','diff_a1_sum_buy','diff_a1_mean_buy','diff_a1_median_buy','diff_a1_min_buy','diff_a1_max_buy','diff_a1_vol_sum_buy','diff_a1_vol_mean_buy','diff_a1_vol_median_buy','diff_a1_vol_min_buy','diff_a1_vol_max_buy',' diff_a1_sum_sell','diff_a1_mean_sell','diff_a1_median_sell','diff_a1_min_sell','diff_a1_max_sell','diff_a1_vol_sum_sell','diff_a1_vol_mean_sell','diff_a1_vol_median_sell','diff_a1_vol_min_sell','diff_a1_vol_max_sell','severity','labels'])
        for keys,item in all_feature_dict[(val)].items():
            csv_writer.writerow(item)
print(all_feature_dict[(timestamp_set_list[3])])
# feature_write = np.vstack((head,feature_write))
# print(feature_write[0:2])
# np.sort(feature_write,order = 'timestamp')
# with open('feature_vector_rbf_all.csv','w') as f:
#     csv_writer = csv.writer(f)
#     csv_writer.writerow(['trader', 'timestamp', 'a1_sum_buy','a1_mean_buy','a1_median_buy','a1_min_buy','a1_max_buy','a1_vol_sum_buy','a1_vol_mean_buy','a1_vol_median_buy','a1_vol_min_buy','a1_vol_max_buy',' a1_sum_sell','a1_mean_sell','a1_median_sell','a1_min_sell','a1_max_sell','a1_vol_sum_sell','a1_vol_mean_sell','a1_vol_median_sell','a1_vol_min_sell','a1_vol_max_sell','diff_a1_sum_buy','diff_a1_mean_buy','diff_a1_median_buy','diff_a1_min_buy','diff_a1_max_buy','diff_a1_vol_sum_buy','diff_a1_vol_mean_buy','diff_a1_vol_median_buy','diff_a1_vol_min_buy','diff_a1_vol_max_buy',' diff_a1_sum_sell','diff_a1_mean_sell','diff_a1_median_sell','diff_a1_min_sell','diff_a1_max_sell','diff_a1_vol_sum_sell','diff_a1_vol_mean_sell','diff_a1_vol_median_sell','diff_a1_vol_min_sell','diff_a1_vol_max_sell','severity','labels'])
#     csv_writer.writerows(feature_write)

#usable
for i in range(int(len(feature_write)/100)+1):
    with open('features_rbf_batch/feature_vector_'+str(i)+'.csv','w',) as f:
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

