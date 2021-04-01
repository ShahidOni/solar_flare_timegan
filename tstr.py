import numpy as np
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import seaborn as sns
import random 


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=(20, 5))

    axes[0].set_title(title)
    # if ylim is not None:
    #     axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    print(train_scores_mean)
    print(train_scores_std)
    print(test_scores_mean)
    print(test_scores_std)
    # print(train_sizes)
    # Plot learning curve

    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    # axes[1].grid()
    # axes[1].plot(train_sizes, fit_times_mean, 'o-')
    # axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
    #                      fit_times_mean + fit_times_std, alpha=0.1)
    # axes[1].set_xlabel("Training examples")
    # axes[1].set_ylabel("fit_times")
    # axes[1].set_title("Scalability of the model")

    # # Plot fit_time vs score
    # axes[2].grid()
    # axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    # axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
    #                      test_scores_mean + test_scores_std, alpha=0.1)
    # axes[2].set_xlabel("fit_times")
    # axes[2].set_ylabel("Score")
    # axes[2].set_title("Performance of the model")

    return plt

def delete_outlier(data):
    
    lst_idx = []
    counter = 0
    i = 0
    for el in data:
        if -99999. in el:
            counter= counter+1
            lst_idx.append(i)
        i=i+1
    X = np.delete(data, lst_idx, axis=0)
    return X, lst_idx

def load_solar_data(classLabel = 0):

    with open('./solar_flare.pck', 'rb') as solar_file:
        original = pickle.load(solar_file)

    upper_bound = int(original.shape[0]/4) * (classLabel + 1)
    lower_bound = upper_bound - int(original.shape[0]/4)

    ori_data = delete_outlier(original[lower_bound:upper_bound])

    return ori_data

def remove_outlier():

    data = None
    y = None
    for i in range(4):
        segmented_data, lst_idx = load_solar_data(classLabel=i)

        if i==0:
            data = segmented_data
        else:
            data = np.concatenate((data, segmented_data), axis=0)
        
        with open("solar_label.pck", 'rb') as ori_label:
            original_label = pickle.load(ori_label)
            if y is None:
                y = np.zeros(segmented_data.shape[0], dtype=np.int32)
            else :
                y_temp = np.zeros(segmented_data.shape[0], dtype=np.int32)
                y_temp.fill(i)
                y = np.concatenate((y,y_temp), axis=0)
    return data, y

with open('result1/generated.pck', 'rb') as part1:
    synthetic_1 = pickle.load(part1)
with open('result2/generated.pck', 'rb') as part2:
    synthetic_2 = pickle.load(part2)
with open('result3/generated.pck', 'rb') as part3:
    synthetic_3 = pickle.load(part3)

with open('result0/generated.pck', 'rb') as solar_file:
    synthetic_0 = pickle.load(solar_file)
    sythetic = np.concatenate((synthetic_0, synthetic_1, synthetic_2, synthetic_3), axis=0)
    y_0 = np.zeros(synthetic_0.shape[0], dtype= int)
    y_1 = np.ones(synthetic_1.shape[0], dtype= int)
    y_2 = np.zeros(synthetic_2.shape[0], dtype= int)
    y_2.fill(2)
    y_3 = np.zeros(synthetic_3.shape[0], dtype= int)
    y_3.fill(3)
    synthetic_label = np.concatenate((y_0,y_1,y_2,y_3), axis= 0)

original, original_label = remove_outlier()


#scaler = StandardScaler()
# Train On Synthetic Test on Real
rand_ = random.randrange(original.shape[0])
print(rand_)
X_train = sythetic.reshape((sythetic.shape[0], sythetic.shape[1]*sythetic.shape[2]))
y_train = synthetic_label

X_test = original.reshape((original.shape[0], original.shape[1]*original.shape[2]))
# x_ = range(X_train.shape[1])
# sns.lineplot(x_,X_test[rand_], legend="brief", label="Original")
# sns.lineplot(x_,X_train[rand_], legend="brief", label = "Synthetic")
# plt.show()
y_test = original_label

# X_train = scaler.fit_transform(X_train)
# X_test = scaler.fit_transform(X_test)

x_ = range(1980)
#sns.lineplot(x_,original[20][10], legend="brief", label="Original")
#sns.lineplot(x_,sythetic[119][10], legend="brief", label = "Synthetic")
#plt.show()
for i in range(200,210):
    sns.lineplot(x_,sythetic[i].flatten(), legend="brief", label="Synthetic")
    sns.lineplot(x_,original[i].flatten(), legend="brief", label="Original")
    #print(sythetic[i][1])
    plt.show()
exit(1)
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

#clf.fit(X_train,y_train)
#y_pred=clf.predict(X_test)

fig, axes = plt.subplots(3, 2, figsize=(10, 15))
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print("Train on Synthetic Test on Real")

gnb = GaussianNB()
gnb.fit(X_train,y_train)
y_pred = gnb.predict(X_test)

print("Gaussian Naive Bayes Accuracy:", metrics.accuracy_score(y_test, y_pred))

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
y_pred_ = clf.predict(X_test)
plt = plot_learning_curve(clf, "Random Forest", X_train, y_train, axes=axes[:, 0], ylim=(0.7, 1.01),
                     cv=cv, n_jobs=4)
plt.show()

print("Random Forest Accuracy Accuracy", metrics.accuracy_score(y_test, y_pred_))




# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# exit(1)
#y_pred = gnb.fit(X_train, y_train).predict(X_test)
# print(gnb.score(X_test,y_test))
# exit(1)

#print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))

# plt = plot_learning_curve(gnb, "Gaussian Naive Bayes", X_train, y_train, axes=axes[:, 0], ylim=(0.7, 1.01),
#                     cv=cv, n_jobs=4)
# plt.show()

#Train on Real Test on Synthetic

X_test = sythetic.reshape((sythetic.shape[0], sythetic.shape[1]*sythetic.shape[2]))
y_test = synthetic_label
X_train = original.reshape((original.shape[0], original.shape[1]*original.shape[2]))
y_train = original_label

# X_train = scaler.fit_transform(X_train[:,[0,400]])
# X_test = scaler.fit_transform(X_test[:, [0,400]])
print()
print("Train on Real Test on Synthetic")

gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
#print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
print("Gaussian Naive Bayes Accuracy:", metrics.accuracy_score(y_test, y_pred))

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

print("Random Forest Accuracy ", metrics.accuracy_score(y_test, y_pred))


