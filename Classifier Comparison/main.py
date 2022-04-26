import time

import warnings
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
warnings.filterwarnings('ignore')


def data_preprocessing(df):
    h_k_df = df[(df['T'] == 'H') | (df['T'] == 'K')]
    m_y_df = df[(df['T'] == 'M') | (df['T'] == 'Y')]
    a_b_df = df[(df['T'] == 'A') | (df['T'] == 'B')]
    validation_h_k = h_k_df.sample(frac=0.1)
    validation_m_y = m_y_df.sample(frac=0.1)
    validation_a_b = a_b_df.sample(frac=0.1)
    train_h_k = h_k_df.drop(labels=validation_h_k.index)
    train_m_y = m_y_df.drop(labels=validation_m_y.index)
    train_a_b = a_b_df.drop(labels=validation_a_b.index)
    return h_k_df, m_y_df, a_b_df, validation_h_k, validation_m_y, validation_a_b, train_h_k, train_m_y, train_a_b


def plot_model(grid, color, name): # https://stackoverflow.com/questions/37161563/how-to-graph-grid-scores-from-gridsearchcv
    results = grid.cv_results_
    means_test = results['mean_test_score']
    stds_test = results['std_test_score']
    masks = []
    masks_names = list(grid.best_params_.keys())
    for p_k, p_v in grid.best_params_.items():
        masks.append(list(results['param_' + p_k].data == p_v))
    params = grid.param_grid
    fig, ax = plt.subplots(1, len(params), sharex='none', sharey='all', figsize=(20, 5))
    fig.suptitle('Score per parameter')
    fig.text(0.04, 0.5, 'Accuracy', va='center', rotation='vertical')
    for i, p in enumerate(masks_names):
        m = np.stack(masks[:i] + masks[i + 1:])
        best_parms_mask = m.all(axis=0)
        best_index = np.where(best_parms_mask)[0]
        x = np.array(params[p])
        y_1 = np.array(means_test[best_index])
        e_1 = np.array(stds_test[best_index])
        ax[i].errorbar(x, y_1, e_1, linestyle='-.', marker='o', color=color)
        ax[i].set_xlabel(p)
        fig.savefig(name, bbox_inches='tight')
    plt.show()


def plot_naive_bayes(res, var_smoothing, color, name):
    sm1 = res['mean_test_score']
    sm1 = np.array(sm1)
    fig, ax = plt.subplots(1, 1)
    plt.scatter(var_smoothing, sm1, c=color)
    ax.set_title('Score in Grid Search')
    ax.set_xlabel('var_smoothing')
    ax.set_ylabel('Accuracy')
    ax.grid('on')
    fig.savefig(name, bbox_inches='tight')
    plt.show()


def model_naive_bayes(h_k_x, h_k_y, m_y_x, m_y_y, a_b_x, a_b_y, cv, classification_model, name):
    var_smoothing = np.logspace(0, -9, num=100)
    clf = GridSearchCV(classification_model, dict(var_smoothing=var_smoothing), cv=cv)
    model_fit_pair1 = clf.fit(h_k_x, h_k_y)
    print("For", name, ", Result for Pair 1 (H and K): ", model_fit_pair1.best_params_)
    print("For", name, ", Best accuracy for Pair 1 (H and K): ", model_fit_pair1.best_score_)
    plot_naive_bayes(model_fit_pair1.cv_results_, var_smoothing, 'black', name+'_h_k.pdf')

    clf = GridSearchCV(classification_model, dict(var_smoothing=var_smoothing), cv=cv)
    model_fit_pair2 = clf.fit(m_y_x, m_y_y)
    print("For", name, ", Result for Pair 2 (M and Y): ", model_fit_pair2.best_params_)
    print("For", name, ", Best accuracy for Pair 2 (M and Y): ", model_fit_pair2.best_score_)
    plot_naive_bayes(model_fit_pair2.cv_results_, var_smoothing, 'blue', name+'_m_y.pdf')

    clf = GridSearchCV(classification_model, dict(var_smoothing=var_smoothing), cv=cv)
    model_fit_pair3 = clf.fit(a_b_x, a_b_y)
    print("For", name, ", Result for Pair 3 (A and B): ", model_fit_pair3.best_params_)
    print("For", name, ", Best accuracy for Pair 3 (A and B): ", model_fit_pair3.best_score_)
    plot_naive_bayes(model_fit_pair3.cv_results_, var_smoothing, 'red', name+'_a_b.pdf')

    return [model_fit_pair1, model_fit_pair2, model_fit_pair3]


def model(h_k_x, h_k_y, m_y_x, m_y_y, a_b_x, a_b_y, cv, classification_model, hyper, name):
    clf = GridSearchCV(classification_model, hyper, cv=cv)
    model_fit_pair1 = clf.fit(h_k_x, h_k_y)
    print("For", name, ", Result for Pair 1 (H and K): ", model_fit_pair1.best_params_)
    print("For", name, ", Best accuracy for Pair 1 (H and K): ", model_fit_pair1.best_score_)
    plot_model(model_fit_pair1, 'black', name+'_h_k.pdf')

    clf = GridSearchCV(classification_model, hyper, cv=cv)
    model_fit_pair2 = clf.fit(m_y_x, m_y_y)
    print("For", name, ", Result for Pair 2 (M and Y): ", model_fit_pair2.best_params_)
    print("For", name, ", Best accuracy for Pair 2 (M and Y): ", model_fit_pair2.best_score_)
    plot_model(model_fit_pair2, 'blue', name+'_m_y.pdf')

    clf = GridSearchCV(classification_model, hyper, cv=cv)
    model_fit_pair3 = clf.fit(a_b_x, a_b_y)
    print("For", name, ", Result for Pair 3 (A and B): ", model_fit_pair3.best_params_)
    print("For", name, ", Best accuracy for Pair 3 (A and B): ", model_fit_pair3.best_score_)
    plot_model(model_fit_pair3, 'red', name+'_a_b.pdf')

    return [model_fit_pair1, model_fit_pair2, model_fit_pair3]


def dimension_reduction(h_k_x, m_y_x, a_b_x):
    s = h_k_x.var()
    s = s.sort_values(ascending=False)
    idx = s.index
    _features1 = []
    for i in range(4):
        _features1.append(idx[i])
    rd_h_k_x = h_k_x[_features1]
    # print(rd_h_k_x)

    s = m_y_x.var()
    s = s.sort_values(ascending=False)
    idx = s.index
    _features2 = []
    for i in range(4):
        _features2.append(idx[i])
    rd_m_y_x = m_y_x[_features2]
    # print(rd_m_y_x)

    s = a_b_x.var()
    s = s.sort_values(ascending=False)
    idx = s.index
    _features3 = []
    for i in range(4):
        _features3.append(idx[i])
    rd_a_b_x = a_b_x[_features3]
    # print(rd_a_b_x)
    return rd_h_k_x, rd_m_y_x, rd_a_b_x, _features1, _features2, _features3


def plot_importance(x, y, classification_model, color):
    rf_model = classification_model.fit(x, y)
    result = permutation_importance(rf_model, x, y)
    sorted_idx = result.importances_mean.argsort()
    plt.barh(range(len(sorted_idx)), result.importances_mean[sorted_idx], align='center', color=color)
    plt.yticks(range(len(sorted_idx)), np.array(x.columns)[sorted_idx])
    plt.title('Permutation Importance')
    plt.show()


def test_model_with_validation_set(validation_h_k, validation_m_y, validation_a_b):
    test_h_k_x = validation_h_k.drop(columns=['T'])
    test_h_k_y = validation_h_k['T']
    test_m_y_x = validation_m_y.drop(columns=['T'])
    test_m_y_y = validation_m_y['T']
    test_a_b_x = validation_a_b.drop(columns=['T'])
    test_a_b_y = validation_a_b['T']
    return test_h_k_x, test_h_k_y, test_m_y_x, test_m_y_y, test_a_b_x, test_a_b_y


def test_model(models, test_h_k_x, test_h_k_y, test_m_y_x, test_m_y_y, test_a_b_x, test_a_b_y, name):
    T1 = time.time()
    print("For ", name, ", Validation accuracy for Pair 1 (H and K): ", "{:.2f}".format(models[0].score(test_h_k_x, test_h_k_y) * 100))
    end1 = time.time()
    print("For ", name, ", Validation time for Pair 1 (H and K): ", (end1 - T1)*1000, "ms")
    T2 = time.time()
    print("For ", name, ", Validation accuracy for Pair 2 (M and Y): ", "{:.2f}".format(models[1].score(test_m_y_x, test_m_y_y) * 100))
    T3 = time.time()
    print("For ", name, ", Validation time for Pair 2 (M and Y): ", (T3 - T2)*1000, "ms")
    T4 = time.time()
    print("For ", name, ", Validation accuracy for Pair 3 (A and B): ", "{:.2f}".format(models[2].score(test_a_b_x, test_a_b_y) * 100))
    T5 = time.time()
    print("For ", name, ", Validation time for Pair 3 (A and B): ", (T5 - T4)*1000, "ms")


if __name__ == '__main__':
    data_frame = pd.read_csv("letter-recognition.data")
    h_k_df, m_y_df, a_b_df, validation_h_k, validation_m_y, validation_a_b, train_h_k, train_m_y, train_a_b = data_preprocessing(data_frame)
    h_k_x = train_h_k.drop(columns=['T'])
    h_k_y = train_h_k['T']
    m_y_x = train_m_y.drop(columns=['T'])
    m_y_y = train_m_y['T']
    a_b_x = train_a_b.drop(columns=['T'])
    a_b_y = train_a_b['T']

    ###########################################################    WITHOUT DIMENSION REDUCTION    ###################################################################

    print("############################   WITHOUT DIMENSION REDUCTION   #########################")
    # K_Nearest Neighbors: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    classification_model = KNeighborsClassifier()
    models_knn = model(h_k_x, h_k_y, m_y_x, m_y_y, a_b_x, a_b_y, 5, classification_model, dict(p=[1, 2, 3, 4, 5, 6, 7], n_neighbors=[1, 3, 5, 7, 9]), 'knn')

    # Decision Tree: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    print("\n")
    classification_model = DecisionTreeClassifier()
    models_dt = model(h_k_x, h_k_y, m_y_x, m_y_y, a_b_x, a_b_y, 5, classification_model, dict(max_depth=[2, 4, 6, 8, 10], max_features=['auto', 'sqrt', 'log2']), 'dt')

    # Random Forest: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    print("\n")
    classification_model = RandomForestClassifier()
    models_rf = model(h_k_x, h_k_y, m_y_x, m_y_y, a_b_x, a_b_y, 5, classification_model, dict(n_estimators=[20, 40, 60, 80, 100], max_depth=[4, 6, 8, 10, 12]), 'rf')

    # Support Vector Machine: https://scikit-learn.org/stable/modules/svm.html
    print("\n")
    classification_model = SVC()
    models_svm = model(h_k_x, h_k_y, m_y_x, m_y_y, a_b_x, a_b_y, 5, classification_model, dict(kernel=['rbf'], gamma=[1, 0.1, 0.01, 0.001, 0.0001], C=[0.1, 1, 10, 100, 1000]), 'svm')

    # Artificial Neural Network: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
    print("\n")
    classification_model = MLPClassifier()
    models_ann = model(h_k_x, h_k_y, m_y_x, m_y_y, a_b_x, a_b_y, 5, classification_model, dict(activation=['identity', 'logistic', 'tanh', 'relu'], learning_rate_init=[0.1, 0.01, 0.001, 0.002, 0.0001]), 'ann')

    # Logistic Regression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    print("\n")
    classification_model = LogisticRegression()
    models_lr = model(h_k_x, h_k_y, m_y_x, m_y_y, a_b_x, a_b_y, 5, classification_model, dict(solver=['newton-cg', 'lbfgs', 'liblinear'], C=[0.00001, 0.0001, 0.001, 0.01, 0.1]), 'lr')

    # Naive Bayes: https://scikit-learn.org/stable/modules/naive_bayes.html
    print("\n")
    classification_model = GaussianNB()
    models_nb = model_naive_bayes(h_k_x, h_k_y, m_y_x, m_y_y, a_b_x, a_b_y, 5, classification_model, 'nb')

    ###########################################################     DIMENSION REDUCTION STARTS     ###################################################################

    print("\n")
    print("######################################   WITH DIMENSION REDUCTION   ######################################")

    rd_h_k_x, rd_m_y_x, rd_a_b_x, _features1, _features2, _features3 = dimension_reduction(h_k_x, m_y_x, a_b_x)   # Features with max variance for KNN, Decision Tree, SVM

    # K-Nearest Neighbors I have used max variance
    classification_model = KNeighborsClassifier()
    dr_models_knn = model(rd_h_k_x, h_k_y, rd_m_y_x, m_y_y, rd_a_b_x, a_b_y, 5, classification_model, dict(p=[1, 2, 3, 4, 5, 6, 7], n_neighbors=[1, 3, 5, 7, 9]), 'knn_dim_reduction')

    # Decision Tree I have used max variance
    classification_model = DecisionTreeClassifier()
    dr_models_dt = model(rd_h_k_x, h_k_y, rd_m_y_x, m_y_y, rd_a_b_x, a_b_y, 5, classification_model, dict(max_depth=[2, 4, 6, 8, 10], max_features=['auto', 'sqrt', 'log2']), 'dt_dim_reduction')

    # Support Vector Machine I have used max variance
    classification_model = SVC()
    dr_models_svm = model(rd_h_k_x, h_k_y, rd_m_y_x, m_y_y, rd_a_b_x, a_b_y, 5, classification_model, dict(kernel=['rbf'], gamma=[1, 0.1, 0.01, 0.001, 0.0001], C=[0.1, 1, 10, 100, 1000]), 'svm_dim_reduction')

    # For Random forest, I have used the permutation importance to find the four most common features: https://scikit-learn.org/stable/modules/permutation_importance.html
    classification_model = RandomForestClassifier()
    plot_importance(h_k_x, h_k_y, classification_model, 'black')
    plot_importance(m_y_x, m_y_y, classification_model, 'blue')
    plot_importance(a_b_x, a_b_y, classification_model, 'coral')
    _features_h_k = ['8.1', '0', '8.2', '0.1']
    _features_m_y = ['0.1', '8.3', '10', '8.4']
    _features_a_b = ['0.2', '8.4', '10', '13']
    rf_h_k_x = h_k_x[_features_h_k]
    rf_m_y_x = m_y_x[_features_m_y]
    rf_a_b_x = a_b_x[_features_a_b]
    dr_models_rf = model(rf_h_k_x, h_k_y, rf_m_y_x, m_y_y, rf_a_b_x, a_b_y, 5, classification_model, dict(n_estimators=[20, 40, 60, 80, 100], max_depth=[4, 6, 8, 10, 12]), 'rf_dim_reduction')

    # For ANN, I have used the PCA to find the four principal components: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    pca = PCA(n_components=4)
    tran_h_k = pca.fit_transform(h_k_x)
    tran_m_y = pca.fit_transform(m_y_x)
    tran_a_b = pca.fit_transform(a_b_x)
    classification_model = MLPClassifier()
    dr_models_ann = model(tran_h_k, h_k_y, tran_m_y, m_y_y, tran_a_b, a_b_y, 5, classification_model, dict(activation=['identity', 'logistic', 'tanh', 'relu'], learning_rate_init=[0.1, 0.01, 0.001, 0.002, 0.0001]), 'ann_dim_reduction')

    ###########################################################     Testing Phase STARTS     ###################################################################

    #**********************************************************   Without Dimension Reduction   **********************************************************

    print("\n")
    print("#######################################   VALIDATION WITHOUT DIMENSION REDUCTION   ############################################")
    test_h_k_x, test_h_k_y, test_m_y_x, test_m_y_y, test_a_b_x, test_a_b_y = test_model_with_validation_set(validation_h_k, validation_m_y, validation_a_b)

    # k-Nearest Neighbors (KNN)
    print("\n")
    test_model(models_knn, test_h_k_x, test_h_k_y, test_m_y_x, test_m_y_y, test_a_b_x, test_a_b_y, 'KNN')

    # Decision Tree
    print("\n")
    test_model(models_dt, test_h_k_x, test_h_k_y, test_m_y_x, test_m_y_y, test_a_b_x, test_a_b_y, 'Decision Tree')

    # Random Forest
    print("\n")
    test_model(models_rf, test_h_k_x, test_h_k_y, test_m_y_x, test_m_y_y, test_a_b_x, test_a_b_y, 'Random Forest')

    # Support Vector Machine (SVM)
    print("\n")
    test_model(models_svm, test_h_k_x, test_h_k_y, test_m_y_x, test_m_y_y, test_a_b_x, test_a_b_y, 'SVM')

    # Artificial Neural Network (ANN)
    print("\n")
    test_model(models_ann, test_h_k_x, test_h_k_y, test_m_y_x, test_m_y_y, test_a_b_x, test_a_b_y, 'ANN')

    # Logistic Regression
    print("\n")
    test_model(models_lr, test_h_k_x, test_h_k_y, test_m_y_x, test_m_y_y, test_a_b_x, test_a_b_y, 'Logistic Regression')

    # Naive Bayes
    print("\n")
    test_model(models_nb, test_h_k_x, test_h_k_y, test_m_y_x, test_m_y_y, test_a_b_x, test_a_b_y, 'Naive Bayes')

    # **********************************************************   With Dimension Reduction   **********************************************************
    print("\n")
    print("###############################################   VALIDATION WITH DIMENSION REDUCTION   ##########################################")

    # Dimension reduction for KNN, Decision Tree, and SVM with max variance
    dr_test_h_k_x = test_h_k_x[_features1]
    dr_test_m_y_x = test_m_y_x[_features2]
    dr_test_a_b_x = test_a_b_x[_features3]

    # Dimension reduction for ANN with PCA
    pca_test_h_k_x = pca.transform(test_h_k_x)
    pca_test_m_y_x = pca.transform(test_m_y_x)
    pca_test_a_b_x = pca.transform(test_a_b_x)

    # Dimension reduction for Random Forest with permutation importance
    rf_test_h_k_x = test_h_k_x[_features_h_k]
    rf_test_m_y_x = test_m_y_x[_features_m_y]
    rf_test_a_b_x = test_a_b_x[_features_a_b]

    # k-Nearest Neighbors (KNN)
    print("\n")
    test_model(dr_models_knn, dr_test_h_k_x, test_h_k_y, dr_test_m_y_x, test_m_y_y, dr_test_a_b_x, test_a_b_y,
               'KNN (with dimension reduction)')

    # Decision Tree
    print("\n")
    test_model(dr_models_dt, dr_test_h_k_x, test_h_k_y, dr_test_m_y_x, test_m_y_y, dr_test_a_b_x, test_a_b_y,
               'Decision Tree (with dimension reduction)')

    # Support Vector Machine
    print("\n")
    test_model(dr_models_svm, dr_test_h_k_x, test_h_k_y, dr_test_m_y_x, test_m_y_y, dr_test_a_b_x, test_a_b_y,
               'SVM (with dimension reduction)')

    # Artificial Neural Network
    print("\n")
    test_model(dr_models_ann, pca_test_h_k_x, test_h_k_y, pca_test_m_y_x, test_m_y_y, pca_test_a_b_x, test_a_b_y,
               'ANN (with dimension reduction)')

    # Random Forest
    print("\n")
    test_model(dr_models_rf, rf_test_h_k_x, test_h_k_y, rf_test_m_y_x, test_m_y_y, rf_test_a_b_x, test_a_b_y,
               'Random Forest (with dimension reduction)')
