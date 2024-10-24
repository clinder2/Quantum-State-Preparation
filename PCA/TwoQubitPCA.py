import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
matplotlib.use('MacOSX')
matplotlib.rcParams['interactive'] == True

if __name__ == "__main__":
    dataset = pd.read_csv('Testbuild.csv')
    a = dataset.iloc[:, 0:4].values
    fidelity = dataset.iloc[:, 4:5].values
    mse = dataset.iloc[:, 5:6].values
    data = []
    for i in range(0, 100):
        b = a[i]
        data.append(b)
    sc = StandardScaler()
    pca = PCA(n_components=4)
    #reduced_data = PCA(n_components=4).fit_transform(data)
    #pca.fit(data)
    #print(pca.singular_values_)
    reduced_data = sc.fit_transform(data)
    reduced_data = pca.fit_transform(data)
    print(pca.singular_values_)
    fig = plt.figure(1, figure=(8,8))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(
        reduced_data[:, 0],
        reduced_data[:, 1]
    )
    for i in range(0, 100):
        temp = reduced_data[i]
        fid = fidelity[i]
        m = mse[i]
        fid1 = round(fid[0], 3)
        m1 = round(m[0], 3)
        if fid[0] < 0.99 and m[0] > 0.1:
            ax.annotate("Fidelity: " + str(fid1) + ", MSE: " + str(m1), (temp[0], temp[1]))
            ax.scatter(temp[0], temp[1], c="red")
        elif fid[0] < 0.99:
            ax.annotate("Fidelity: " + str(fid1), (temp[0], temp[1]))
            ax.scatter(temp[0], temp[1], c="orange")
        elif m[0] > 0.1:
            ax.annotate("MSE: " + str(m1), (temp[0], temp[1]))
            ax.scatter(temp[0], temp[1], c="yellow")
    matplotlib.pyplot.show()
    #t = train_test_split()
    #data = sc.fit_transform(a)
    #data = pca.fit_transform(data)
    #ev = pca.explained_variance_ratio_
    #print(reduced_data)
    #print(ev)
