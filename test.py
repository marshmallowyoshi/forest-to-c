from sklearn.ensemble import RandomForestClassifier
import sklearn.datasets as datasets
import forest_to_c as converter

iris = datasets.load_iris()
rf = RandomForestClassifier()
y = [iris.target_names[x] for x in iris.target]
rf.fit(iris.data, y)
converter.forest_to_c(rf)