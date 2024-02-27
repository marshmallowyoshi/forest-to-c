"""
Module used to test functionality of converter using the Iris dataset.
"""
from sklearn.ensemble import RandomForestClassifier
import sklearn.datasets as datasets
import forest_to_c as converter

iris = datasets.load_iris()
x = iris['data']
y = [iris['target_names'][x] for x in iris['target']]
rf = RandomForestClassifier()
rf.fit(x, y)
converter.forest_to_c(rf, keep_temporary_files=True)

# predictions = rf.predict(x)

# x_predict = [list(sample)+ [predictions[idx]] for idx, sample in enumerate(x)]

# with open('test.csv', 'w') as f:
#     for sample in x_predict:
#         f.write(','.join([str(x) for x in sample]) + '\n')