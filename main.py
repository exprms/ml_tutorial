#
#
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import classes

df = pd.read_csv('iris.data', header=None)

# Auswahl setosa und versicolor
y = df.loc[:, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# Auswahl Kelch- und Blütenlänge
X = df.loc[:,[0,2]].values

# Scatter Plot
setosa_index = np.where(y==1)
versicolor_index = np.where(y==-1)

#plt.scatter(X[setosa_index,0], X[setosa_index,1], color='red', marker='o', label='setosa')
#plt.scatter(X[versicolor_index,0], X[versicolor_index,1], color='blue', marker='o', label='versicolor')
#plt.xlabel('Laenge Kelchblatt [cm]')
#plt.legend(loc='upper left')
#plt.show()

# Klassifikation
ppn = classes.Perceptron(eta=0.01, n_iter=10)
ppn.fit(X,y)

classes.plot_error(ppn)