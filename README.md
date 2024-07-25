# Ml_VertabrateDTree
Vertebrate Classification with Decision Trees
This project demonstrates the use of a decision tree classifier to categorize vertebrates into two classes: mammals and non-mammals. The classification model is implemented using the scikit-learn library.

Project Overview
The goal of this project is to build a decision tree classifier that can distinguish between mammals and non-mammals based on features such as whether the animal is warm-blooded, gives birth, or is an aquatic creature.

Files
vertebrate_classification.ipynb: Jupyter notebook containing the code for loading the dataset, preprocessing the data, training the decision tree classifier, and evaluating the model.
Dataset
The dataset used is a CSV file with information about various vertebrates. The dataset includes the following columns:

Name: The name of the animal
Warm-blooded: Whether the animal is warm-blooded (1 = Yes, 0 = No)
Gives Birth: Whether the animal gives birth (1 = Yes, 0 = No)
Aquatic Creature: Whether the animal is an aquatic creature (1 = Yes, 0 = No)
Aerial Creature: Whether the animal is an aerial creature (1 = Yes, 0 = No)
Has Legs: Whether the animal has legs (1 = Yes, 0 = No)
Hibernates: Whether the animal hibernates (1 = Yes, 0 = No)
Class: The class of the animal (mammals, reptiles, fishes, amphibians, birds)
Installation
To run this project, you need Python and the following packages:

pandas
scikit-learn
pydotplus
matplotlib
You can install these packages using pip:

bash
Copy code
pip install pandas scikit-learn pydotplus matplotlib
Usage
Load the dataset:

python
Copy code
import pandas as pd
data = pd.read_csv('/path/to/vertebrate_dataset.csv')
Preprocess the data:

python
Copy code
# Replace class values
data['Class'] = data['Class'].replace(['reptiles', 'fishes', 'amphibians', 'birds'], 'non-mammals')
Train the model:

python
Copy code
from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
X = data.drop(['Name', 'Class'], axis=1)
y = data['Class']
clf = clf.fit(X, y)
Visualize the decision tree:

python
Copy code
import pydotplus
from IPython.display import Image
dot_data = tree.export_graphviz(clf, feature_names=X.columns, class_names=['mammals', 'non-mammals'], filled=True, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
Evaluate the model:

python
Copy code
from sklearn.metrics import accuracy_score

# Test data
testData = pd.DataFrame([['gila monister', 0, 0, 0, 0, 1, 1, 'non-mammals'],
                         ['platypus', 1, 0, 0, 0, 1, 1, 'mammals'],
                         ['owl', 1, 0, 0, 1, 1, 0, 'non-mammals'],
                         ['dolphin', 1, 1, 1, 0, 0, 0, 'mammals']],
                        columns=data.columns)
testX = testData.drop(['Name', 'Class'], axis=1)
testY = testData['Class']
predY = clf.predict(testX)
accuracy = accuracy_score(testY, predY)
print('Accuracy on test data is %.2f' % accuracy)
Results
The decision tree model achieved an accuracy of 75% on the test data.

