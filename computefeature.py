# %matplotlib inline
import matplotlib.pyplot as plt
import mglearn
# from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_recall_curve, roc_curve, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from computedata import *
from loaddata import *

training_data = load_training_data()
# print('hello')
# print(training_data)
# print('bye')
# Compute features.
print("[*] Computing features...")
f = PhishFeatures()
training_features = f.compute_features(training_data.keys(), values_only=False)
# feature_vector = training_features['names']
print("[+] Features computed for the {} samples in the training set.".format(len(training_features['values'])))

# Train the Classifier

# Assign the labels (0s and 1s) to a numpy array.
labels = np.fromiter(training_data.values(), dtype=np.float)
print("[+] Assigned the labels to a numpy array.")

# Split the data into a training set and a test set.
X_train, X_test, y_train, y_test = train_test_split(training_features['values'], labels, random_state=5)
print("[+] Split the data into a training set and test set.")

# Insert silver bullet / black magic / david blaine / unicorn one-liner here :)
# classifier = LogisticRegression(C=10).fit(X_train, y_train)
# classifier = SVC(kernel='rbf', probability=True)
# classifier = GaussianNB()
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train, y_train)




# classifier = RandomForestClassifier(n_estimators=300, criterion='entropy')
# classifier.fit(X_train, y_train)
print("[+] Completed training the classifier: {}".format(classifier))

# See how well it performs against training and test sets.
print("Accuracy on training set: {:.3f}".format(classifier.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(classifier.score(X_test, y_test)))

# section5
# print("Number of features: {}".format(len(feature_vector)))

# Visualize the most important coefficients from the LogisticRegression model.
# coef = classifier.coef_
# mglearn.tools.visualize_coefficients(coef, feature_vector, n_top_features=10)


precision, recall, thresholds = precision_recall_curve(y_test, classifier.predict_proba(X_test)[:, 1])
close_zero = np.argmin(np.abs(thresholds - 0.5))
plt.plot(precision[close_zero], recall[close_zero], 'o', markersize=10, label="threshold 0.5", fillstyle="none",
         c='k', mew=2)
plt.plot(precision, recall, label="precision recall curve")
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.legend(loc="best")
print("Precision: {:.3f}\nRecall: {:.3f}\nThreshold: {:.3f}".format(precision[close_zero], recall[close_zero],
                                                                    thresholds[close_zero]))

fpr, tpr, thresholds = roc_curve(y_test, classifier.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr, label="ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR (recall)")
close_zero = np.argmin(np.abs(thresholds - 0.5))
plt.plot(fpr[close_zero], tpr[close_zero], 'o', markersize=10, label="threshold 0.5", fillstyle="none", c="k",
         mew=2)
plt.legend(loc=4)
print("TPR: {:.3f}\nFPR: {:.3f}\nThreshold: {:.3f}".format(tpr[close_zero], fpr[close_zero],
                                                           thresholds[close_zero]))

predictions = classifier.predict_proba(X_test)[:, 1] > 0.5
print(classification_report(y_test, predictions, target_names=['Not Phishing', 'Phishing']))

confusion = confusion_matrix(y_test, predictions)
print("Confusion matrix:\n{}".format(confusion))

# A prettier way to see the same data.
scores_image = mglearn.tools.heatmap(
    confusion_matrix(y_test, predictions), xlabel="Predicted Label", ylabel="True Label",
    xticklabels=["Not Phishing", "Phishing"],
    yticklabels=["Not Phishing", "Phishing"],
    cmap=plt.cm.gray_r, fmt="%d")
plt.title("Confusion Matrix")
plt.gca().invert_yaxis()
