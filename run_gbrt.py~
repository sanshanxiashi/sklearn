import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn import ensemble

origin = '/home/hadoop/git/learn_sklearn/sklearn/out.csv'
#读数据
data = pd.read_csv(origin)	
my_X = data.iloc[:,:27].as_matrix()
my_y = data.iloc[:,27].as_matrix()
X = data.iloc[:,[0,1,3,4,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]].as_matrix()
y = data.iloc[:,27].as_matrix()

#标准化        
scaler = preprocessing.StandardScaler().fit(my_X) 
X = scaler.transform(my_X)
y = my_y


#训练svr
params = {'n_estimators': 1000, 'max_depth':8, 'min_samples_split': 2, 'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)
y_gbrt = clf.fit(X[0:22300], y[0:22300])
y_pred = y_gbrt.predict(X[22300:27879])
y_true = y[22300:27879]

#svr 测评
score1 = clf.score(X[0:22300], y[0:22300])
score2 = clf.score(X[22300:27879], y[22300:27879])
mae = mean_absolute_error(y_true, y_pred)
#2.929567667686277
mse = mean_squared_error(y_true, y_pred)
#25.569808144118888


