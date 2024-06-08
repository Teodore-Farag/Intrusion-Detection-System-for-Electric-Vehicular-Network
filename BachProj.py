#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 01:02:22 2024

@author: osx
"""
import matplotlib.pyplot as plt  
import pandas as pd 
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler    
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix, ConfusionMatrixDisplay,precision_score,classification_report
import tkinter as tk
from tkinter import messagebox
import os
def show_alert(text):
    root = tk.Tk()
    root.withdraw() 
    os.system("""
              osascript -e 'tell application "System Events" to set frontmost of the first process whose unix id is {} to true'
              """.format(os.getpid()))
    messagebox.showinfo("Alert",text)

    root.destroy() 

csvFile = pd.read_csv('filtered.csv')


columns_to_exclude = csvFile.iloc[:, 16:25]
a=csvFile.drop(columns=columns_to_exclude)
x=a.iloc[:,:29]
y= a.iloc[:, 29].values  

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=58) 
# clf=RandomForestClassifier(random_state=58)
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [None, 10, 20],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }
# # # test_row=x_train[1]

# # # # print(test_row.reshape(1,-1))
# grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1)
# grid_search.fit(x_train, y_train)

# # best_params = grid_search.best_params_
# print("Best parameters:", best_params)

best_clf = RandomForestClassifier(random_state=58, max_depth=20,min_samples_leaf=1,min_samples_split=2,n_estimators=200)
best_clf.fit(x_train, y_train)



benigntest=pd.read_csv('Benign test.csv')
b_test=benigntest.iloc[:,:29]
b_pred =best_clf.predict(b_test)
Mirai_test=pd.read_csv('Test.csv')
M_test=Mirai_test.iloc[:,:29]
M_pred=best_clf.predict(M_test)
# flag=True
# #Testing benign data
# for index,a in enumerate(b_pred):
#     if a != "BenignTraffic":
#         flag=False
#         show_alert("Danger a "+ a+" attack")
#         print(index)
#         break
# if flag:
#     print("Safe")
##Attack data Test
for index,a in enumerate(M_pred):
    if a != "BenignTraffic":
        show_alert("Danger a "+ a+" attack")
        print(index)
        break

y_pred = best_clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# # Generate the confusion matrix
# cm = confusion_matrix(y_test, y_pred)
# fig, ax = plt.subplots(figsize=(12, 8))
# custom_labels = ['BenignTraffic', 'ICMP_Fragmentation', 'Mirai-greeth_flood','Mirai-greip_flood','HostDiscovery','PortScan']
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=custom_labels)
# disp.plot(cmap=plt.cm.Blues,ax=ax)
# plt.xticks(rotation=30) 
# plt.show()