from sklearn.ensemble import RandomForestClassifier
import sklearn.datasets as datasets
import main as converter

iris = datasets.load_iris()
rf = RandomForestClassifier()
rf.fit(iris.data, iris.target)
converter.main(rf)