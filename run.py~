import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.svm import SVR

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
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
y_rbf = svr_rbf.fit(X[0:22300], y[0:22300])
y_pred = y_rbf.predict(X[22300:27879])
y_true = y[22300:27879]

#svr 测评
score1 = svr_rbf.score(X[0:22300], y[0:22300])
score2 = svr_rbf.score(X[22300:27879], y[22300:27879])
mae = mean_absolute_error(y_true, y_pred)
#3.4203856737975098
mse = mean_squared_error(y_true, y_pred)
#28.814921623085109

# kernal核心
#svr_poly = SVR(kernel='poly', C=1e3, degree=2)
