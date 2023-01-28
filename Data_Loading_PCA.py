import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import datasets

import warnings
warnings.filterwarnings("ignore")

iris = datasets.load_iris()

iris_df = pd.DataFrame(iris.data)

iris_df["class"] = iris.target

iris_df_columns = ['sepal_len','sepal_width','petal_len','petal_width','class']

# Reading dataset using pandas
#url = 'https://archive.ics.uci.edu/ml/machine-learningdatabases/iris/iris.data'
#names = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]
#dataset = pd.read_csv(url, names = names)

print(iris_df.head(10))
X = iris_df.drop("class", 1)
Y = iris_df["class"]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

print(f"\nDataSet before PCA :\n\nTrain :\n{x_train}\n\nTest :\n{x_test}")

pca = PCA()
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
pca = PCA(n_components=2)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

print(f"\nDataSet After PCA :\n\nTrain :\n{x_train}\n\nTest :\n{x_test}")
