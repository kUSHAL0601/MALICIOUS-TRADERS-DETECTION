import numpy as np
def update_features(features,option="Mahalanobis"):
	'''
	Update features based on distance type

	Returns features changed based on distance type

	Keyword arguments:
	features --> Features taken into consideration
	option --> Distance type: Euclidean, Mahalanobis
	'''
	if option=="Mahalanobis":
		return features
	