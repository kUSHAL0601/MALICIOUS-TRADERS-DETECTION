# MALICIOUS-TRADERS-DETECTION

## Factor Dependency Analysis

- The graphs for features for traders of different behaviours are developed in the factor dependency analysis.
- Firstly a higher order didctionary is used to store the data of stock prices in a structured and systematic manner.
- Then the plots are merged for various traders and as per their behaviour (malicious or non-malicious) the graphs are used to differentiate and compare the traders.
- More the feature differentiates the behaviour of a traders (malicious from non-malicious) , intuitively more important is the feature in determining the classification function.

We have used 2 data sets
- Completely pure with ( non malicious data)
- Mixed data with few malicious traders.

The Traders or users are as follows
1) Trader T1 who behaves as malicious from 2nd data set.(But to
experiment we put trader T1 in pure trader’s List as well as rogue
trader’s List.)
2) Trader T3 from pure data set. (This means no malicious behaviour
from T3).
3) Trader T2 which is non malicious trader from mixed data set.
4) Trader T6 who behaves maliciously in the 2nd data set.
Now we can analyze from differences in graphs and get the important
features.

## ANOMALY DETECTION using pyOD

- message.csv has the data of all trades occuring at the stock exchange
- attack.csv has the malicious trades listed in it.
- script1.py has the code used for training using One Class SVM (OCSVM) as well as testing the accuracy of classification with respect to each feature this gives the importance of a feature in determining the malicious traders.
