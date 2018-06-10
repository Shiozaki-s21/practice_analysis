#import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#setting for plotting
plt.style.use('ggplot')
font = {'family' : 'meiryo'}
matplotlib.rc('font', **font)

#read data from csv
test = pd.read_csv("kaggle/Machine_Learning_From_Disaster/Titanic_Data/test.csv")
train = pd.read_csv("kaggle/Machine_Learning_From_Disaster/Titanic_Data/train.csv")

#discribe test
test.head()
test.describe()

#describe about train
train.head()

train.describe()

#TODO I want median about each data




#confirm to dificit
count_deficit(test)
count_deficit(train)

#TODO translate character to figure
#SEX Male → 0 Female → 1
#Embarked S → 0 Q → 1 C → 2
#test
test["Sex"][test["Sex"] == "Male"] = 0
test["Sex"][test["Sex"] == "Female"] = 1
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "Q"] = 1
test["Embarked"][test["Embarked"] == "C"] = 2
#train
train["Sex"][train["Sex"] == "Male"] = 0


train["Sex"][train["Sex"] == "Female"] = 1
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "Q"] = 1
train["Embarked"][train["Embarked"] == "C"] = 2

#For confirm
test.head(10)
train.head(10)




#summary data
indexer = np.arange(0,100,10)
test.plot(y =["Age"], bins=50,figsize=(16,4),alpha=0.5,kind='hist',grid = True, xticks = indexer)

#TODO ?:経済力(= Fara,Pclass )が同じであれば年齢も対して変わらないのでは？
#TODO A:相関はなかった

#median
test["Age"].median()
#TODO Ageと他の値との相関関係を調べたい
#各列の相関関係を調べる
test_corr = test.corr()
print(test_corr.shape)
sns.heatmap(test_corr, square=True, annot=True, vmax=1, vmin=-1, center=0)
train_corr = train.corr()
sns.heatmap(train_corr, annot=True,square=True, vmax=1, vmin=-1, center=0)

pd.tools.plotting.scatter_matrix(test,figsize=(16,4))
plt.savefig("test.png")


sns.set_style('whitegrid')
%matplotlib inline

test.plot(x = "PassengerId", y = "Age",figsize=(16,4),alpha=0.5,kind = "Scatter")


#補完処理の開始



#欠損データの存在確認




#Function for counting dificit
def count_deficit(data_frame):
    blank_data = data_frame.isnull().sum()
    percent = 100 * blank_data / len(data_frame)
    summary = pd.concat([blank_data, percent], axis = 1)
    summary_table = summary.rename(columns = {0: "dificit_count",1: "%"})
    return summary_table
