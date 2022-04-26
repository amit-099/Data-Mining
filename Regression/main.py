import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


def mean_squared_error(train_input, train_output, w_init):
    return (1 / len(train_input[0])) * np.sum(np.square(w_init.T@train_input - train_output))


def linear_regression(train_input, train_output):
    iterations = 10000
    learning_rate = 1e-07
    i = 0
    initial_weight = []
    for i in range(len(train_input)):
        initial_weight.append([random.uniform(0, 1)])
    initial_weight = np.array(initial_weight)
    mse = mean_squared_error(train_input, train_output, initial_weight)
    updated_mse = 0
    while i < iterations:
        gradient = np.reshape(((2 / len(train_input[0])) * ((initial_weight.T@train_input) - train_output)@train_input.T), (len(train_input), 1))
        initial_weight = initial_weight - learning_rate * gradient
        updated_mse = mean_squared_error(train_input, train_output, initial_weight)
        if updated_mse > mse:
            learning_rate = learning_rate * 0.8
        else:
            learning_rate = learning_rate * 1.15
        mse = updated_mse
        i = i + 1
    update_weight = initial_weight
    new_loss = updated_mse
    prediction = update_weight.T@train_input
    variance = (np.max(train_output) - np.min(train_output))
    variance_train = 1 - (new_loss / variance)
    return update_weight, new_loss, prediction, variance_train


def uni_variate_training(train_input, train_output, test_input):
    uni_variate = {}
    for i in range(1, len(train_input)):
        uni_variate_train = train_input[[0, i]]
        uni_variate_weight, uni_variate_loss, uni_variate_prediction, uni_variate_variance = linear_regression(uni_variate_train, train_output)
        uni_variate[str(i)] = [uni_variate_weight, uni_variate_loss, test_input[[0, i]], uni_variate_prediction, uni_variate_variance]
    return uni_variate


def draw_data_plot(x_, y_, train_out_, x_label_, y_label_, name):
    f = plt.figure(1, figsize=(10, 10))
    plt.scatter(x_, train_out_, c='coral')
    plt.xlabel(x_label_)
    plt.ylabel(y_label_)
    plt.plot(x_, y_, color='blue')
    f.savefig(name, bbox_inches='tight')
    plt.show()


def data_pre_process(df):
    input_data = df.loc[:, df.columns != "Concrete compressive strength(MPa, megapascals) "]
    input_data = input_data.loc[:, input_data.columns != "Bias"]
    label = df["Concrete compressive strength(MPa, megapascals) "]
    train_input = input_data.head(900)
    np_train_in = train_input.values.T
    test_input = input_data.tail(130)
    np_test_in = test_input.values.T
    train_label = label.head(900)
    np_train_out = train_label.values.T
    test_label = label.tail(130)
    np_test_out = test_label.values.T

    return np_train_in, np_train_out, np_test_in, np_test_out, input_data, label, df


def data_normalization(d_frame):
    for column_ in d_frame.columns:
        if column_ != 'Concrete compressive strength(MPa, megapascals) ' and column_ != 'bias':
            d_frame[column_] = (d_frame[column_] - d_frame[column_].mean()) / d_frame[column_].std()
    return d_frame


def print_result(result_dictionary, test_output):
    for key in result_dictionary.keys():
        train_mse = result_dictionary[key][1]
        test_mse = mean_squared_error(result_dictionary[key][2], test_output, result_dictionary[key][0])
        print("Feature is ", key)
        print('Training Loss is ', train_mse)
        print('Testing Loss is ', test_mse)
        print('Final Weight is ', result_dictionary[key][0])
        print('Variance Explained (Training Data) ', result_dictionary[key][4])
        print('Variance Explained (Test Data) ', 1 - (test_mse / (np.max(test_output) - np.min(test_output))))
        print('')


def draw_data_histograms(data_, name_, color_):
    f = plt.figure(1, figsize=(20, 28))
    ax = f.gca()
    axes_1 = data_.hist(ax=ax, color=color_)
    f.savefig(name_, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    ##############################################################   ORIGINAL DATA PART   ######################################################################
    data = pd.read_excel('Concrete_Data.xls')
    hist_data = data.copy()
    data.insert(loc=0, column='bias', value=1)
    train_in, train_out, test_in, test_out, input_, output_, data_frame = data_pre_process(data.copy())
    uni_variate_output = uni_variate_training(train_in, train_out, test_in)
    print_result(uni_variate_output, test_out)
    multi_variate_weight, multi_variate_loss_train, multi_variate_prediction, multi_variate_variance = linear_regression(train_in, train_out)
    multi_variate_loss_test = mean_squared_error(test_in, test_out, multi_variate_weight)
    print("The Training Loss for Multivariate Regression: ", multi_variate_loss_train)
    print("The Testing Loss for Multivariate Regression: ", multi_variate_loss_test)
    print("Variance Explained for Multivariate (Training Data): ", multi_variate_variance)
    print("Variance Explained for Multivariate (Test Data): ", 1 - multi_variate_loss_test / (np.max(test_out) - np.min(test_out)))
    print("Coefficient: ", multi_variate_weight)

    ##############################################################   NORMALIZATION PART   ######################################################################
    normalized_data = data_normalization(data.copy())
    normal_train_in, normal_train_out, normal_test_in, normal_test_out, normal_input_, normal_output_, normal_data_frame = data_pre_process(
        normalized_data.copy())
    normal_uni_variate_output = uni_variate_training(normal_train_in, normal_train_out, normal_test_in)
    print("--------------------Normalized Data Results--------------------")
    print_result(normal_uni_variate_output, normal_test_out)
    normal_multi_variate_weight, normal_multi_variate_loss_train, normal_multi_variate_prediction, normal_multi_variate_variance = linear_regression(
        normal_train_in, normal_train_out)
    normal_multi_variate_loss_test = mean_squared_error(test_in, test_out, multi_variate_weight)
    print("The Training Loss for Normalized Multivariate: ", multi_variate_loss_train)
    print("The Testing Loss for Normalized Multivariate: ", mean_squared_error(test_in, test_out, multi_variate_weight))
    print("Variance Explained for Multivariate (Normalized Training Data): ", normal_multi_variate_variance)
    print("The Explained Variance for Multivariate (Normalized Test Data): ",
          1 - normal_multi_variate_loss_test / (np.max(normal_test_out) - np.min(normal_test_out)))
    print("Coefficient: ", normal_multi_variate_weight)

    ##############################################################   Scatter plots original data   ######################################################################

    data_cols = []
    for column in data.columns:
        if column != 'Concrete compressive strength(MPa, megapascals) ':
            data_cols.append(column)
    for i in range(1, len(train_in)):
        w = uni_variate_output[str(i)][0]
        y = w[0] + train_in[i] * w[1]
        x_label = data_cols[i]
        y_label = "Concrete compressive strength(MPa, megapascals)"
        name = "plt_" + str(i) + ".pdf"
        draw_data_plot(train_in[i], y, train_out, x_label, y_label, name)

    ##############################################################   Scatter plots normalized data   ######################################################################
    for i in range(1, len(normal_train_in)):
        w = normal_uni_variate_output[str(i)][0]
        y = w[0] + normal_train_in[i] * w[1]
        x_label = data_cols[i]
        y_label = "Concrete compressive strength(MPa, megapascals)"
        name = "normalized_plt_" + str(i) + ".pdf"
        draw_data_plot(normal_train_in[i], y, normal_train_out, x_label, y_label, name)

    normal_data_frame = normal_data_frame.loc[:, normal_data_frame.columns != "bias"]
    draw_data_histograms(hist_data, "original_dataset_hist.pdf", "black")
    draw_data_histograms(normal_data_frame, "normalized_hist.pdf", "red")
