#Age = median
#Embarked = friqency


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib


test_data = pd.read_csv("kaggle/Machine_Learning_From_Disaster/Titanic_Data/test.csv")
train_data = pd.read_csv("kaggle/Machine_Learning_From_Disaster/Titanic_Data/train.csv")


# trancelate dataes
# test_data
test_data["Sex"][test_data["Sex"] == "Male"] = 0
test_data["Sex"][test_data["Sex"] == "Female"] = 1
test_data["Embarked"][test_data["Embarked"] == "S"] = 0
test_data["Embarked"][test_data["Embarked"] == "Q"] = 1
test_data["Embarked"][test_data["Embarked"] == "C"] = 2

# train_data
train_data["Sex"][train_data["Sex"] == "Male"] = 0
train_data["Sex"][train_data["Sex"] == "Female"] = 1
train_data["Embarked"][train_data["Embarked"] == "S"] = 0
train_data["Embarked"][train_data["Embarked"] == "Q"] = 1
train_data["Embarked"][train_data["Embarked"] == "C"] = 2


indexer = np.arange(0, 100, 10)
test_data.plot(y =["Age"], bins=50,figsize=(16,4),alpha=0.5,kind='hist',grid = True, xticks = indexer)
