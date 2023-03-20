from matplotlib import pyplot as plt
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report, \
    recall_score, precision_score
from sklearn.feature_selection import SelectKBest, chi2

# adapted from https://machinelearningmastery.com/feature-selection-with-categorical-data/
# load the dataset
def load_dataset(filename):
    data = read_csv(filename, sep=',')
    # data = data[data['v_gene'].str.contains('OR') == False].reset_index(drop=True)

    x_1 = data.iloc[:, 0:9]
    # print(x_1)
    enc = OrdinalEncoder()
    enc.fit(x_1[['v_gene', 'j_gene']])
    x_1[['v_gene', 'j_gene']] = enc.transform(x_1[['v_gene', 'j_gene']])
    X = x_1.values
    # print(X)

    Y_1 = data.iloc[:, -1]
    le = LabelEncoder()
    le.fit(Y_1)
    y = le.transform(Y_1)
    print(y)

    # features = data.columns[1:7].values
    # print(features)
    return X, y


# feature selection
def select_features(X_train, y_train, X_test):
    fs = SelectKBest(score_func=chi2, k=4)  # two possibilities: chi2, mutual_info_classif
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs


# load the dataset
X_train, y_train = load_dataset('./testtable.csv')
X_test, y_test = load_dataset('./test/MS5_test.csv')
# split into train and test.txt sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# prepare input data
# X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)
# print(X_train_enc)
# print(X_test_enc)
# # prepare output data
# y_train_enc, y_test_enc = prepare_targets(y_train, y_test)
# print(y_train_enc)
# print(y_test_enc)
# feature selection
X_train_fs, X_test_fs = select_features(X_train, y_train, X_test)
# fit the model
model = DecisionTreeClassifier()  # LogisticRegression(solver='liblinear')
model.fit(X_train_fs, y_train)
# evaluate the model
y_pred = model.predict(X_test_fs)
y_prob = model.predict_proba(X_test_fs)
print(y_test)
print('###', y_prob)

# evaluate predictions
con_matrix = confusion_matrix(y_test, y_pred)
report_dt = classification_report(y_test, y_pred)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred, zero_division=1))
print('Precision:', precision_score(y_test, y_pred, zero_division=1))
tp, tn, fn, fp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
cd4_num = tp + fn
p_estimated = cd4_num / len(y_test)
print(p_estimated)
print(report_dt)

# Plotting ROC curve
auc = roc_auc_score(y_test, y_prob[:, 1])
false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_prob[:, 1])
plt.plot(false_positive_rate, true_positive_rate, label="AUC=" + str(auc))
plt.title('ROC Curve')
plt.ylabel('True Positive Rate')
plt.xlabel('false Positive Rate')
plt.legend(loc=4)
plt.show()
print(round(auc, 2))

# Plotting Confusion Matrix
fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(con_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(con_matrix.shape[0]):
    for j in range(con_matrix.shape[1]):
        ax.text(x=j, y=i, s=con_matrix[i, j], va='center', ha='center', size='xx-large')

plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()
