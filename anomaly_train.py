import matplotlib.pyplot as plt
from statistics import mean,stdev,median
import csv
import math
import numpy as np
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA 
from sklearn.metrics import roc_auc_score
from sklearn.svm import OneClassSVM as oc_svm 
import pickle

file2 = open('non_malicious_timestamps','rb')
non_malicious_timestamps = pickle.load(file2)

file2 = open('malicious_timestamps','rb')
malicious_timestamps = pickle.load(file2)

# file2 = open('malicious_timestamps','wb')
# pickle.dump(malicious_timestamps,file2)

file2 = open('non_malicious','rb')
non_malicious = pickle.load(file2)

file2 = open('malicious','rb')
mal_traders = pickle.load(file2)

file = open('data_normal','rb')
normal_complete_data_arr = pickle.load(file)

file = open('data_mal','rb')
malicious_complete_data_arr = pickle.load(file)

all_labels = []

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

test_keys = non_malicious[int(0.8*len(normal_complete_data_arr)):len(all_labels)] + mal_traders



train_data_params = [train_normal_arr.mean(axis = 0),train_normal_arr_stnd_2.std(axis = 0)+1]
file2 = open('train_mean_std.txt','wb')
pickle.dump(train_data_params,file2)

eta = 1
def sigmoid(x):
  return 1 / (1 + math.exp(eta*x))
import csv
## OCSVM
# clf1 = OCSVM(kernel = 'rbf',gamma = 1,nu = 0.4)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
# print(np.arange(2**-2,2**2,0.05))
tuned_parameters = [{'kernel': ['rbf'], 'gamma': np.arange(2**-2,2**2,0.05),
                     'nu':  np.arange(0.05,0.4,0.05)}]
scores = ['recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(oc_svm(), tuned_parameters, cv=2,
                       scoring='%s_macro' % score)
    clf.fit(train_normal_arr_stnd,train_labels)

    # print("Best parameters set found on development set:")
    # print()
    # print(clf.best_params_)
    # print()
    # print("Grid scores on development set:")
    # print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    #     print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))
    # print()

    # print("Detailed classification report:")
    # print()
    # print("The model is trained on the full development set.")
    # print("The scores are computed on the full evaluation set.")
    # print()
    y_true, y_pred = test_labels, clf.predict(test_all_data_stnd)
    # print(classification_report(y_true, y_pred))
    # print()

print(clf.best_params_)
params = clf.best_params_
clf1 = oc_svm(kernel = 'rbf',nu= params['nu'],gamma = params['gamma'])
clf1.fit(train_normal_arr_stnd)
filename = 'classifier.sav'
pickle.dump(clf1, open(filename, 'wb'))

roc = []
recall_list = []

# clf1.threshold_ = i
predicted = clf1.predict(test_all_data_stnd)


# print(predicted)
# print(clf1.score_samples(test_all_data_stnd))
score_test = clf1.score_samples(test_all_data_stnd)
# print("min score:%f and max score:%f"%(min(score_test),max(score_test)))
min_score_test = min(score_test)
max_score_test = max(score_test)
# print(test_labels)
# for k in np.arange(-0.01,0.01,0.0005):
# print("for",k)
from sklearn.metrics import roc_curve, auc
roc_auc = []
fpr_list = []
tpr_list = []
fpr_list.append(0.0)
tpr_list.append(0.0)
data_thresh = []
data_to_show = []
for k in np.arange(min_score_test,max_score_test,0.005):    
    predicted = []
    for val in range(len(test_all_data_stnd)):
        # print(clf1.score_samples(test_all_data_stnd[val].reshape(1,-1)))
        if clf1.score_samples(test_all_data_stnd[val].reshape(1,-1)) > k:
            predicted.append(1)
        else:
            predicted.append(-1)

    accuracy = 0
    recall = 0
    indices = []
    indices_positive = 0
    indices_negative = 0
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
        if predicted[i] == 1:
            # pred_indices_mal.append(i)
            indices_negative +=1
        
    precision = recall/indices_positive
    # print("OCSVM Precision",100*precision)
    # print("OCSVM accuracy",100*accuracy/len(test_labels))
    # print("OCSVM recall",100*recall/len(malicious_complete_data_arr))
    # print("f measure", ((precision*recall)/(precision+recall)))
    # print(len(clf1.support_vectors_))
    # print(clf1.score_samples(test_all_data_stnd))
    # print(len(clf1.decision_scores_[clf1.decision_scores_>clf1.threshold_])) 
    try: 
        roc.append(roc_auc_score(predicted,test_labels))
        fpr, tpr, _ = roc_curve(test_labels,predicted)
        true_pos_rate = recall/len(malicious_complete_data_arr)
        false_positive = indices_positive - recall
        true_negative = len(normal_complete_data_arr) -  false_positive
        true_negative = indices_negative - (len(malicious_complete_data_arr) - recall)
        false_pos_rate = false_positive/(false_positive +true_negative) 
        # print(len(clf1.support_vectors_))
        roc_auc.append(auc(fpr, tpr))
        fpr_list.append(false_pos_rate)
        tpr_list.append(true_pos_rate)
        #  fpr_list.append(fpr[1])
        # tpr_list.append(tpr[1])
        recall_list.append(recall)
        # print("roc",roc_auc_score(predicted,test_labels))
        data_thresh.append([k,100*precision,100*recall/len(malicious_complete_data_arr),100*accuracy/len(test_labels),roc_auc_score(predicted,test_labels)])
        data_to_show.append([round(k,5),round(100*precision),round(100*recall/len(malicious_complete_data_arr),3),round(100*accuracy/len(test_labels),3),round(roc_auc_score(predicted,test_labels),3)])
        
    except:
        roc.append(0)
        recall_list.append(0)
        pass
# score = clf1.score_samples(test_all_data_stnd)
# severity = []

data_to_show.sort(key = lambda i:(i[2],i[4]),reverse = True)
# print(data_thresh)
data_to_show = np.asarray(data_to_show)
# print(data_to_show)
# print(roc)
# result = Counter(traders_mal)
# [item for items, c in Counter(traders_mal).most_common() 
                                      # for item in [items] * c]
# print(severity)
with open('roc_values.txt','w') as f:
    f.writelines("%s \n" %('[threshold   prec   recall   accuracy   auc]'))
    f.writelines("%s \n" % val for val in data_to_show)