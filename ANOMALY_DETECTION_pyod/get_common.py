a=[]
i=0
a.append(set())
a.append(set())
a.append(set())
a.append(set())
with open('all.txt') as f:
	l=f.readlines()
	for j in l:
		if j=='\n':
			i+=1
		a[i].add(j.strip('\n'))
f.close()
common=a[0]&a[1]&a[2]&a[3]
for i in common:
	print(i)
