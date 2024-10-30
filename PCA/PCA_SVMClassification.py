import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn import svm
from sklearn.svm import LinearSVC
matplotlib.use('MacOSX')
matplotlib.rcParams['interactive'] == True

if __name__ == "__main__":
    dataset = pd.read_csv('Testbuild2.csv')
    a = dataset.iloc[:, 0:4].values
    fidelity = dataset.iloc[:, 4:5].values
    mse = dataset.iloc[:, 5:6].values
    data = []
    y = []
    for i in range(0, 301):
        b = a[i]
        data.append(b)
        y.append(fidelity[i][0])
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
    X = reduced_data[:, 0:2]
    for i in range(0, len(y)): #1 = fidelity below threshold, 0 = acceptable
        if y[i] < 0.99:
            y[i] = 1
        else:
            y[i] = 0
    clf = LinearSVC(C=1).fit(X, y)
    decision = clf.decision_function(X)
    sv_indices = np.where(np.abs(decision) <= .5)[0]
    svs = X[sv_indices]
    print(svs)
    DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        ax=ax,
        plot_method="contour"
    )
    for i in range(0, 300):
        temp = reduced_data[i]
        fid = fidelity[i]
        m = mse[i]
        fid1 = round(float(fid[0]), 3)
        m1 = round(float(m[0]), 3)
        if float(fid[0]) < 0.99 and float(m[0]) > 0.1:
            ax.annotate("Fidelity: " + str(fid1) + ", MSE: " + str(m1), (temp[0], temp[1]))
            ax.scatter(temp[0], temp[1], c="red")
        elif float(fid[0]) < 0.99:
            ax.annotate("Fidelity: " + str(fid1), (temp[0], temp[1]))
            ax.scatter(temp[0], temp[1], c="orange")
        elif float(m[0]) > 0.1:
            ax.annotate("MSE: " + str(m1), (temp[0], temp[1]))
            ax.scatter(temp[0], temp[1], c="yellow")
    #ax.scatter(
    #    svs[:, 0],
    #    svs[:, 1],
    #    linewidth=1
    #)
    matplotlib.pyplot.show()
    #t = train_test_split()
    #data = sc.fit_transform(a)
    #data = pca.fit_transform(data)
    #ev = pca.explained_variance_ratio_
    #print(reduced_data)
    #print(ev)
