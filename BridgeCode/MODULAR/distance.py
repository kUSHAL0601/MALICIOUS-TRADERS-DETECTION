import numpy as np
def update_features(features,option="Euclidean"):
	'''
	Update features based on distance type

	Returns features changed based on distance type

	Keyword arguments:
	features --> Features taken into consideration
	option --> Distance type: Euclidean, Mahalanobis
	'''
	if option=="Euclidean":
		return features
	elif option=="Mahalanobis":
		features=np.asarray(features)
		print(features.mean())
		return (features-features.mean())/features.std()
