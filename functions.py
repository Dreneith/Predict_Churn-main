'''
functions module
Author: Akindele Abdulrasheed
Date: Aug 6th, 2022
'''
# Modules to be imported
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import RocCurveDisplay, classification_report
sns.set()



def univariate_plot_save(data, colume_name, fig_name, value_count=False):
    '''
    Creates a univariate plot or a bar plot for value_count
    Args:
        df: it takes a dataframe
        colume_name: The column on which the function acts on
        fig_name: Name of the file to save as an image
        value_count: condition to create a normalized bar plot
    '''
    if value_count:
        plt.figure(figsize=(20, 10))
        data[colume_name].value_counts('normalize').plot(kind='bar')
    else:
        plt.figure(figsize=(20, 10))
        data[colume_name].hist()
    plt.savefig(f"./images/eda/{fig_name}.png")


def distribution_plot(data, column, save_name):
    '''
    Function: Creates an histogram distribution on a column
            and add smoth curve using Kernel density estimate
    input:  df: dataframe that contains columns
            column: Numerical column to perofrm operation on
            save_name: image file name
    '''
    plt.figure(figsize=(20, 10))
    sns.histplot(data[column], stat='density', kde=True)
    plt.savefig(f"./images/eda/{save_name}.png")


def col_encoder_helper(data, col_name):
    '''
    Creates a new column with proportions to 'Churn' based on a category column
    and drops the original column from the DataFrame.

    Args:
        data (DataFrame): The DataFrame containing columns.
        col_name (str): The name of the category column.
    '''
    # Calculate the mean Churn rate for each category in col_name
    category_churn_proportions = data.groupby(col_name)['Churn'].mean()

    # Create a new column with proportions based on category_churn_proportions
    data[f'{col_name}_Churn'] = data[col_name].map(category_churn_proportions)

    # Drop the original category column
    data.drop(columns=[col_name], inplace=True)



def fit_predict(model, x_train, x_test, y_train, grid=False):
    '''
    train and fit predict model
    '''
    model.fit(x_train, y_train)
    if grid:
        train_output = model.best_estimator_.predict(x_train)
        test_output = model.best_estimator_.predict(x_test)
    else:
        train_output = model.predict(x_train)
        test_output = model.predict(x_test)

    return train_output, test_output


def plt_model_result(model, x_test, y_test, file_path):
    '''
    plots and save the roc_curve given a model
    '''
    plt.figure(figsize=(15, 8))
    RocCurveDisplay.from_estimator(model, x_test, y_test)
    plt.savefig(file_path)


def create_reports(y_train, y_train_pred, y_test,
                   y_pred, model_test, model_train):
    '''
    creates a classification report and save as a png file
    args:
        y_train: data to train on
        y_train_pred: a
    '''
    plt.figure(figsize=(15, 15))
    # plotting a heatmap from classification report dictionary
    report_test = classification_report(y_test, y_pred, output_dict=True)
    sns.heatmap(pd.DataFrame(report_test).iloc[:-1, :].T, annot=True,
                cmap='viridis')
    plt.title("Test report")
    plt.savefig(model_test)

    plt.figure(figsize=(15, 15))
    report_train = classification_report(
        y_train, y_train_pred, output_dict=True)
    sns.heatmap(pd.DataFrame(
        report_train).iloc[:-1, :].T, annot=True, cmap='inferno')
    plt.title("Train report")
    plt.savefig(model_train)
