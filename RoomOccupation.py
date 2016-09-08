from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

def line_len(fname):
	i = 0
	with open(fname) as f:
		for line in f:
			i = i + 1
	return i 

def get_accuracy(results, test_respVar):
	length = len(results)
	count = 0
	for x in range (0, length):
		if(results[x]==test_respVar[x]):
			count = count + 1
	accuracy = count/length
	return(accuracy)

data = open('training.txt','r')# open data file
num = line_len('training.txt') # get number of lines
train_num = int(num *.9) # get num of training rows
#print("train num = ", train_num)
data.close()

train = pd.read_csv("training.txt", sep = "\t")
#train.head()
predVarNames = ['apr21__total','apr21__count','apr21__AVGlastHeardInSeconds','apr21__avgRssi', 'apr21__avgConfidence']
responseVarNames = ['apr21__Occupation']
predVar = train.as_matrix(predVarNames)
respVar = train.as_matrix(responseVarNames)


train_predVar = predVar[0:train_num-1]
train_respVar = respVar[0:train_num-1]

trainer = np.array(respVar[0:train_num-1])
trainer = np.ravel(trainer)
#print(train_predVar)

test_predVar = predVar[train_num:num]
test_respVar = respVar[train_num:num]


#print(test_predVar)


model = RandomForestClassifier(n_estimators = 2, bootstrap = True, max_leaf_nodes = 5)
model.fit(train_predVar,trainer)
results = model.predict(test_predVar)
acc = get_accuracy(results,test_respVar)
print("Accuracy = ", int(acc*100), "%")
