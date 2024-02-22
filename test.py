from sklearn.ensemble import RandomForestClassifier
import sklearn.datasets as datasets
import forest_to_c as converter

iris = datasets.load_iris()
x = iris['data']
y = [iris['target_names'][x] for x in iris['target']]
rf = RandomForestClassifier()
rf.fit(x, y)
converter.forest_to_c(rf)
