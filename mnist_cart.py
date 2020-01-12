from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_digits

digits = load_digits()
features = digits.data
labels = digits.target

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25, random_state=33)

clf = DecisionTreeClassifier(criterion='gini')

clf = clf.fit(train_features, train_labels)

test_predict = clf.predict(test_features)

score = accuracy_score(test_labels, test_predict)
print('CART 分类树准确率 %.4lf' % score)

