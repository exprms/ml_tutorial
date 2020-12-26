#
#
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import classes

df = pd.read_csv('iris.data', header=None)
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Auswahl setosa und versicolor
y = df.iloc[:, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# Auswahl Kelch- und Blütenlänge
X = df.iloc[:,[0,2]].values

# scatterplot
sns.scatterplot(df['sepal_length'], df['sepal_width'], hue=df['species'])
plt.show()

# Klassifikation
ppn = classes.Perceptron(eta=0.01, n_iter=10)
ppn.fit(X,y)

classes.plot_error(ppn)