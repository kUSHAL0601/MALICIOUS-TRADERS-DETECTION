print("Enter number of tupples",end=":")
t=int(input())
a=[]
print("Enter: Trader id(string) Attack type(string) Severity(string) for each trader in new lines...")
for _ in range(t):
	a.append(input().split())

human_skills={}
with open('config.txt') as f:
	l=f.readlines()
	no_humans=int(l[0].split('\n')[0])
	humans_id=l[1].split('\n')[0].split()
	no_attack_types=int(l[2].split('\n')[0])
	attack_types=l[3].split('\n')[0].split()
	for i in humans_id:
		human_skills[i]=[]
	for i in range(4,len(l)):
		x=l[i].split('\n')[0].split()
		human_skills[x[0]]+=x[1:]
f.close()
# print(human_skills)

trader_capablities={}
scores={}
attack_traders={}
for i in a:
	scores[(i[0],i[1])]=int(i[2])
	try:
		trader_capablities[i[0]].append(i[1])
	except:
		trader_capablities[i[0]]=[i[1]]
	try:
		attack_traders[i[1]].append(i[0])
	except:
		attack_traders[i[1]]=[i[0]]

# print(trader_capablities)
# print(scores)

humans_traders={}
for i in human_skills:
	humans_traders[i]=[]
	for j in human_skills[i]:
		try:
			humans_traders[i]+=attack_traders[j]
		except:
			pass
	humans_traders[i]=sorted(list(set(humans_traders[i])))
print(humans_traders)