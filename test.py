from sklearn.ensemble import RandomForestClassifier
import sklearn.datasets as datasets
import main as converter

iris = datasets.load_iris()
rf = RandomForestClassifier()
y = [iris.target_names[x] for x in iris.target]
rf.fit(iris.data, y)
converter.main(rf)