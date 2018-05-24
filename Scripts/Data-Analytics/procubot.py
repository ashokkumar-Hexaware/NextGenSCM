# -*- coding: utf-8 -*-
"""
Created on Tue May 15 11:44:17 2018

@author: 39018
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#import data from file
#data source will be changed to Oracle EBS once the tables are ready.
order_history=pd.read_csv('C:\\Users\\39018\\Desktop\\NextGenS\\order.csv')


#DataPreparation - To train the Regression Model
order_pdct=order_history.loc[order_history['p_id'] == 1001]
X = order_pdct[['p_id', 'qty']]
y = order_pdct['days']

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

#Build the regression Model to predict no.of days to ship
lm = LinearRegression()
lm.fit(X,y)

test_data=pd.DataFrame([[1001,90]],columns=['p_id','qty'])

#Predict the days to ship
suggested_days = lm.predict(test_data)

#from sklearn import metrics
#print('MAE:', metrics.mean_absolute_error(y_test, predictions))

#Determine the price of the product by going through the product price when last shipped
price=order_pdct.loc[order_pdct['order_id'] == max(order_pdct['order_id'])]
price_index=order_pdct.loc[order_pdct['order_id'] == max(order_pdct['order_id'])].index[0]
suggested_price= price['price_per_unit']-1

#Now with the P_id,qty,price&shipment_days : find out the best suited vendor.
#Use a Decision Tree classifier

X = order_pdct[['p_id', 'qty','price_per_unit','days']]
y = order_pdct['vendor_name']

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X,y)
test_data=pd.DataFrame([[1001,90,suggested_price[price_index],suggested_days]],columns=['p_id','qty','price_per_unit','days'])

predictions = dtree.predict(test_data)

test_data['vendor']=predictions
test_data['days']
response=test_data.to_json()
