#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：facialRecognition 
@File    ：test.py
@IDE     ：PyCharm 
@Author  ：Lianz
@Date    ：2023/3/23 21:45
"""

import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

from pca import PCA

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# Get the IRIS dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
data = pd.read_csv(url, names=['sepal length', 'sepal width', 'petal length', 'petal width', 'target'])

# prepare the data
x = data.iloc[:, 0:4]

# prepare the target
target = data.iloc[:, 4]

# Applying it to PCA function
mat_reduced = PCA(x, 2)

# Creating a Pandas DataFrame of reduced Dataset
principal_df = pd.DataFrame(mat_reduced, columns=['PC1', 'PC2'])

# Concat it with target variable to create a complete Dataset
principal_df = pd.concat([principal_df, pd.DataFrame(target)], axis=1)

plt.figure(figsize=(6, 6))
sb.scatterplot(data=principal_df, x='PC1', y='PC2', hue='target', s=60, palette='icefire')

plt.show()