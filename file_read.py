import os,os.path

def read_file(filename):
	'''
	Get trader,timestamp,features,severity and labels tuple from file

	Returns trader,timestamp,features,severity and labels tuple

	Keyword arguments:
	filename --> Name of file
	'''
	with open(filename,'r') as f:
		l=f.readlines()
		lines=[]
		for i in l:
			lines.append(i.strip().split(','))
	headings=lines.pop(0)
	#print(headings)
	trader=[]
	timestamp=[]
	features=[]
	severity=[]
	labels=[]
	for i in lines:
		trader.append(i[0])
		timestamp.append(i[1])
		features.append(i[2:-2])
		severity.append(float(i[-2]))
		labels.append(float(i[-1]))
	return (trader,timestamp,features,severity,labels)
