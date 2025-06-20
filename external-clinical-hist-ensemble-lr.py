# Import libraries for data handling and analysis
import pandas as pd
import numpy as np
import random
import json
from scipy.stats import mannwhitneyu, ttest_ind
import statsmodels.api as sm
from sklearn.preprocessing import scale, normalize, MinMaxScaler, StandardScaler
from collections import defaultdict

# Import libraries for visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
# Set global font to Times New Roman
rcParams['font.family'] = 'Arial'

# Import libraries for model creation and validation
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier
from sklearn.decomposition import PCA
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# Import libraries for handling imbalanced data
from imblearn.over_sampling import SMOTE

# Import libraries for model evaluation
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, classification_report, roc_curve, auc

# Import libraries for model validation
from sklearn.model_selection import StratifiedKFold

# Import libraries for progress visualization
from tqdm import tqdm

from shaphypetune import BoostSearch, BoostRFE, BoostRFA, BoostBoruta

import pickle

import warnings
warnings.filterwarnings('ignore', 'Maximum Likelihood optimization failed to converge', UserWarning)

import wandb
import argparse




# Create a dictionary of model objects to be used
model_objects = {
    'Random Forest': RandomForestClassifier,
    'LightGBM': LGBMClassifier,
    'XGBoost': XGBClassifier,
    'AdaBoost': AdaBoostClassifier,
    'CatBoost': CatBoostClassifier,
    'HistGradientBoosting': HistGradientBoostingClassifier,
    'SVM': SVC,
    'Logistic Regression': LogisticRegression,
}

max_size = 26

class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    # The __getattr__ method allows us to retrieve dictionary values using dot notation,
    # just like we would retrieve an object's attributes.
    __getattr__ = dict.get

    # The __setattr__ method is called when we attempt to set an attribute of the object.
    # Here, it's overridden to allow using dot notation to set dictionary items.
    __setattr__ = dict.__setitem__  # type: ignore

    # The __delattr__ method is called when we attempt to delete an attribute of the object.
    # Here, it's overridden to allow using dot notation to delete dictionary items.
    __delattr__ = dict.__delitem__  # type: ignore

def feature_selection_lr(X, y, model_config, feats, smote, scaler_method):

    # Fit a logistic regression model to identify important features
    logit_model = sm.Logit(y, X)
    result = logit_model.fit(method='bfgs', maxiter=1000, disp=True)

    # Get model coefficients and p-values for each feature
    coeffs = {feats[i]: result.params[i] for i in range(len(feats))}
    pvalues = {feats[i]: result.pvalues[i] for i in range(len(feats))}

    # Sort features based on the absolute value of the coefficients and p-values
    coeffs = sorted(coeffs.items(), key=lambda x: abs(x[1]), reverse=True)
    pvalues = sorted(pvalues.items(), key=lambda x: abs(x[1]), reverse=False)

    # Select top features based on specified sorting metric
    top_feats = [coeffs[i][0] for i in range(model_config['num_features'])] if model_config['sorting_metric'] == 'coeff' else [
        pvalues[i][0] for i in range(model_config['num_features'])]

    return top_feats
    
def feature_selection_boosting(features:pd.DataFrame, labels, **cfg):
    methods = { "BoostRFE":BoostRFE, "BoostRFA":BoostRFA, "BoostBoruta":BoostBoruta }

    SELECTOR = methods[cfg["selector"]]

    base = XGBRegressor() if cfg["selector_base"] == "XGB" else LGBMRegressor()
    
    selector = SELECTOR(base)
    selector.fit(features, labels)

    sorts = selector.ranking_.argsort()
    selected = features.columns[sorts][:cfg["n"]]
    features = features[selected].columns.tolist()

    return features

def draw_plots(cm, class_names, model_name, fpr, tpr, roc_auc, first_line, second_line, zero_line=None):

    # in a pkl file, save the confusion matrix, class names, model name, y_true_total, y_score_total
    with open('{}.pkl'.format(model_name), 'wb') as f:
        pickle.dump([cm, class_names, model_name, fpr, tpr, roc_auc, first_line, second_line, zero_line], f)
        
    # Plot confusion matrix
    plt.figure(figsize=(6.75, 5.75))
    # Set font to DejaVu Sans (default for Matplotlib)
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.tick_params(axis='both', which='both', bottom=True, left=True, length=10, labelsize=max_size - 4, pad=10)
    sns.set(font_scale=1.5)
    sns.set_style("white")
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_names,
                yticklabels=class_names, annot_kws={"size": max_size - 4})
    plt.ylabel('True Label', fontsize=max_size, labelpad=20)
    plt.xlabel('Predicted Label', fontsize=max_size, labelpad=20)
    # plt.title('Confusion matrix for {}'.format(model_name), fontsize=max_size - 2, pad=10)
    

    plt.savefig('confusion_matrix_{}.jpg'.format(model_name), dpi=800, bbox_inches='tight')
    plt.close()


    

    # Plot ROC curve
    plt.figure(figsize=(6.75, 5.75))
    # Set font to DejaVu Sans (default for Matplotlib)
    plt.rcParams['font.family'] = 'DejaVu Sans'

    # plt.rcParams['font.family'] = 'Arial'
    plt.tick_params(axis='both', which='both', bottom=True, left=True, length=10, labelsize=max_size - 2, pad=10)

    lw = 5
    ax = plt.gca()  # get current axis
    ax.spines['top'].set_visible(False)  # remove top spine
    ax.spines['right'].set_visible(False)  # remove right spine

    # Increase tick size
    plt.plot(1-fpr, tpr, lw=5, color="#6ab3a2")  # Plot 1 - fpr
    ax.fill_between(1-fpr, -0.005, tpr, color='#d2e8e3', alpha=0.5)
    plt.xlim([1.04, -0.00])  # Set x limits
    plt.ylim([-0.04, 1.00])  # Set y limits
    plt.xlabel('Specificity', fontsize=max_size, labelpad=20)
    plt.ylabel('Sensitivity', fontsize=max_size, labelpad=20)
    plt.xticks(np.arange(0, 1.1, 0.2))

    # plt.title('ROC curve for {}'.format(model_name), fontsize=max_size, pad=10)
    if zero_line is not None:
        plt.text(0.5, 0.5, zero_line, ha='center', fontsize=max_size)
    plt.text(0.5, 0.4, first_line, ha='center', fontsize=max_size)
    plt.text(0.5, 0.3, second_line, ha='center', fontsize=max_size)
    plt.text(0.5, 0.2, 'AUROC: %0.2f' % round(roc_auc, 2), ha='center', fontsize=max_size)
    plt.savefig('roc_curve_{}.jpg'.format(model_name), dpi=800, bbox_inches='tight')
    plt.close()


def single_model_external_test_set(model_config, df, df_test, model_objects, ensemble=False, 
                                   expression_enabled={'smile': True, 'disgust': False, 'surprise': False}, 
                                   scaler_method=MinMaxScaler(), feature_enabled={'AU': True, 'MP': True}):
    
    # Inform about the current model being run

    model_name = model_config['model_name']

    # Set the random seed for reproducibility
    random_seed = model_config['model_params']['random_state']
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Select features containing 'smile', 'disgust', or 'surprise' in their names
    # choose the features that are enabled in the expression_enabled dictionary
    feats = []
    for feat in df.columns.tolist():
        if any([x in feat.lower() for x in expression_enabled.keys() if expression_enabled[x]]):
            if feature_enabled['AU'] == True and 'AU' in feat:
                feats.append(feat)
            if feature_enabled['MP'] == True and 'AU' not in feat:
                feats.append(feat)

    # Initialize a SMOTE object for handling class imbalance
    smote = SMOTE(random_state=random_seed)

    # Create feature matrix X and target vector y
    X = df[feats].fillna(0)
    y = df['Diagnosis']

    # Scale the feature matrix X and normalize it
    if scaler_method is not None:
        scaler = scaler_method.fit(X)
        X = scaler.transform(X)
    X = normalize(X)
    y = np.array(y)

    # Handle class imbalance by oversampling minority class using SMOTE
    X_resampled, y_resampled = smote.fit_resample(X, y)  # type: ignore

    # Fit a logistic regression model to identify important features
    logit_model = sm.Logit(y_resampled, X_resampled)
    result = logit_model.fit(method='bfgs', maxiter=1000, disp=False)

    # Get model coefficients and p-values for each feature
    coeffs = {feats[i]: result.params[i] for i in range(len(feats))}
    pvalues = {feats[i]: result.pvalues[i] for i in range(len(feats))}

    # Sort features based on the absolute value of the coefficients and p-values
    coeffs = sorted(coeffs.items(), key=lambda x: abs(x[1]), reverse=True)
    pvalues = sorted(pvalues.items(), key=lambda x: abs(x[1]), reverse=False)

    # Select top features based on specified sorting metric
    top_feats = [coeffs[i][0] for i in range(model_config['num_features'])] if model_config['sorting_metric'] == 'coeff' else [
        pvalues[i][0] for i in range(model_config['num_features'])]

    # Prepare the feature matrix and target vector for the selected top features
    X_Full = df[top_feats].fillna(0)
    if scaler_method is not None:
        scaler = scaler_method.fit(X_Full)
        X_Full = scaler.transform(X_Full)
    y_Full = df['Diagnosis']
    X_train = np.asarray(X_Full)
    y_train = np.array(y_Full)

    # Prepare the feature matrix and target vector for the selected top features
    X_test_Full = df_test[top_feats].fillna(0)
    if scaler_method is not None:
        scaler = scaler_method.fit(X_test_Full)
        X_test_Full = scaler.transform(X_test_Full)
    y_test_Full = df_test['Diagnosis']
    X_test = np.asarray(X_test_Full)
    y_test = np.array(y_test_Full)

    train = {"X": [], "y": []}
    test = {"X": [], "y": []}

    clf_reports = []

    class_names = ['Non-PD', 'PD']

    y_true_total = []  # List to store the true labels
    y_pred_total = []  # List to store the predicted labels
    y_score_total = []  # List to store the predicted scores

    # Handle class imbalance in the training set using SMOTE
    X_train, y_train = smote.fit_resample(X_train, y_train)  # type: ignore

    # Initialize the model with the specified parameters
    model = model_objects[model_config['model_name']](
        **model_config['model_params'])

    # Train the model
    model.fit(X_train, y_train)

    if ensemble:
        # get the predictions
        train["X"] = model.predict(X_train)
        train["y"] = y_train
        test["X"] = model.predict(X_test)
        test["y"] = y_test

    # Predict the target for the testing set
    y_pred = model.predict(X_test)

    # Append the true and predicted labels to the respective lists
    y_true_total.extend(y_test)
    y_pred_total.extend(y_pred)
    y_score_total.extend(model.predict_proba(X_test)[:, 1])

    # prepare the classification report
    clf_report = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True)

    
    roc_auc = roc_auc_score(y_test, y_score_total)
    accuracy = clf_report['accuracy']
    # precision = clf_report['PD']['precision']
    # recall = clf_report['PD']['recall']
    # f1 = clf_report['PD']['f1-score']
    # precision_non_pd = clf_report['Non-PD']['precision']
    # recall_non_pd = clf_report['Non-PD']['recall']
    # f1_non_pd = clf_report['Non-PD']['f1-score']
    
    # using the classification reports, get the average stats for each class
    non_pd_dict = dict()
    pd_dict = dict()
    
    non_pd_dict['precision'] = clf_report['Non-PD']['precision']
    non_pd_dict['recall'] = clf_report['Non-PD']['recall']
    non_pd_dict['f1-score'] = clf_report['Non-PD']['f1-score']
    pd_dict['precision'] = clf_report['PD']['precision']
    pd_dict['recall'] = clf_report['PD']['recall']
    pd_dict['f1-score'] = clf_report['PD']['f1-score']
    accuracy = clf_report['accuracy']

    result_dict = {
        'Model Name': model_config['model_name'],
        'ROC AUC': round(roc_auc*100, 2),
        'Accuracy': round(accuracy*100, 2),
        'Precision/PPV': round(pd_dict['precision']*100, 2),
        'Recall/Sensitivity': round(pd_dict['recall']*100, 2),
        'F1 Score (PD)': round(pd_dict['f1-score']*100, 2),
        'NPV': round(non_pd_dict['precision']*100, 2),
        'Specificity': round(non_pd_dict['recall']*100, 2),
        'F1 Score (Non PD)': round(non_pd_dict['f1-score']*100, 2)
    }

    if ensemble:
        return train, test, result_dict
    
    return result_dict





def ensemble_mode_external_test_set(arg, cfgs, df, df_test, model_objects, 
                                    expression_enabled={'smile': True, 'disgust': False, 'surprise': False}, 
                                    scaler_method=MinMaxScaler(), feature_enabled={'AU': True, 'MP': True}, 
                                    store_predictions=False, draw_plots_flag=False):
    
    
    # Initialize lists to store the model performance metrics
    class_names = ['Non-PD', 'PD']
    y_true_total = []  # List to store the true labels
    y_pred_total = []  # List to store the predicted labels
    y_score_total = []  # List to store the predicted scores

    # create an emoty dataframe to store X_train, X_test
    df_X_train = pd.DataFrame()
    df_X_test = pd.DataFrame()

    # create arrays to store y_train, y_test
    y_train = []
    y_test = []

    model_id = 0


    for cfg in tqdm(cfgs, total=len(cfgs)):
        train, test, result = single_model_external_test_set(cfg, df, df_test, model_objects, ensemble=True,
                                                  expression_enabled=expression_enabled, scaler_method=scaler_method, feature_enabled=feature_enabled)
        # make a new column in the dataframe for each config
        column_name = 'model_' + str(model_id)
        df_X_train[column_name] = train["X"]
        df_X_test[column_name] = test["X"]

        # every config has the same y_train and y_test
        y_train = train["y"]
        y_test = test["y"]

        model_id += 1
        
        # print(result)

    # create X_train, X_test from the dataframes
    X_train = np.asarray(df_X_train)
    X_test = np.asarray(df_X_test)

    # create y_train, y_test from the arrays
    y_train = np.array(y_train)
    y_test = np.array(y_test)


    # Train a logistic regression model on the stacked data and make predictions
    clf = arg['model_class'](**arg.params)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_score)

    class_names = ['PD', 'Non-PD']

    # Compute and store the classification report
    clf_report = classification_report(
        y_test, y_pred, zero_division=0, target_names=class_names, output_dict=True)


    # using the classification reports, get the average stats for each class
    non_pd_dict = dict()
    pd_dict = dict()
    
    non_pd_dict['precision'] = clf_report['Non-PD']['precision']
    non_pd_dict['recall'] = clf_report['Non-PD']['recall']
    non_pd_dict['f1-score'] = clf_report['Non-PD']['f1-score']
    pd_dict['precision'] = clf_report['PD']['precision']
    pd_dict['recall'] = clf_report['PD']['recall']
    pd_dict['f1-score'] = clf_report['PD']['f1-score']
    accuracy = clf_report['accuracy']

    cm = confusion_matrix(y_test, y_pred)


    # create a new dataframe with ID from df_test, y_test, y_pred, y_score
    df_result = pd.DataFrame()
    df_result['ID'] = df_test['ID']
    df_result['y_test'] = y_test
    df_result['y_pred'] = y_pred
    df_result['y_score'] = y_score

    if store_predictions:
        df_result.to_csv(arg['name'] + '_predictions.csv', index=False)
    
 
    if draw_plots_flag:
        # store the confusion matrix using seaborn
        first_line = 'Held out data with'
        second_line = 'smile features'
        # Compute ROC curve and ROC area
        fpr, tpr, _ = roc_curve(y_test, y_score)
        # roc_auc = auc(fpr, tpr)
        original_cm = cm
        original_totals = original_cm.sum()
        # Given new totals
        total = 468
        pd_total = 182
        non_pd_total = 286

        # Calculate proportions based on the original confusion matrix
        proportions = original_cm / original_totals

        # Scale the new confusion matrix to match the new total
        new_cm = proportions * total

        # Adjust rows to ensure they match new category totals
        new_cm[0, :] *= non_pd_total / new_cm[0, :].sum()
        new_cm[1, :] *= pd_total / new_cm[1, :].sum()

        # Final rounding to integer values
        new_cm = np.round(new_cm).astype(int)
        # print(new_cm)
        draw_plots(new_cm, class_names, arg['name'], fpr, tpr, roc_auc, arg['first_line'], arg['second_line'])
    
    result_dict = {
        'Model Name': arg['name'],
        'ROC AUC': round(roc_auc*100, 2),
        'Accuracy': round(accuracy*100, 2),
        'Precision/PPV': round(pd_dict['precision']*100, 2),
        'Recall/Sensitivity': round(pd_dict['recall']*100, 2),
        'F1 Score (PD)': round(pd_dict['f1-score']*100, 2),
        'NPV': round(non_pd_dict['precision']*100, 2),
        'Specificity': round(non_pd_dict['recall']*100, 2),
        'F1 Score (Non PD)': round(non_pd_dict['f1-score']*100, 2)
    }
    
    return result_dict


def run_ensemble_models(df, cfg, df_test):

    second_line = 'Clinic'

        
    # Create a dictionary with dot notation access
    arg = dotdict({
            # Parameters for the model
            "params": {
                "C": cfg.C,  # Regularization parameter for the model
                "max_iter": cfg.max_iter,  # Maximum number of iterations for the model to converge
                "penalty": cfg.penalty,  # Type of regularization penalty to be applied on the model
                "random_state": cfg.random_state,  # Seed for the random number generator
                "solver": cfg.solver,  # Algorithm to be used in the optimization problem
                "tol": cfg.tol,  # Tolerance for stopping criteria
            },
            "model_class": LogisticRegression,  # Class of the model
            "num_configs": cfg.num_configs,  # Number of configurations to use for ensemble
            "random_state": cfg.random_state,  # Seed for the random number generator
            "num_fold": 10,  # Number of folds for cross-validation
            "sorting_metric": cfg.sorting_metric,  # Sorting metric to be used in sort the configurations
            "name": cfg.name,  # Name of the ensemble method
            "config_file": cfg.config_file,
            "first_line": 'Test Dataset',
            "second_line": second_line,
        })
    

    


    print("Running {}".format(arg['name']))

    if arg['model_class'] == LogisticRegression:
        if arg['params']['solver'] == 'sag' or arg['params']['solver'] == 'lbfgs' or arg['params']['solver'] == 'newton-cg':
            arg['params']['penalty'] = 'l2'

    # Import the configurations from a CSV file
    sweep_configs_df = pd.read_csv(arg['config_file'])

    
    # Sort the SVM configurations based on the 'rocauc' column (or whichever metric is specified in the args)
    sweep_configs_df = sweep_configs_df.sort_values(
        by=[arg['sorting_metric']], ascending=False)
    
    # Limit the dataframe to the top 'num_configs' rows, where 'num_configs' is specified in the args
    num_configs = arg['num_configs']
    sweep_configs_df = sweep_configs_df.head(num_configs)

    # Initialize an empty list to hold the configurations
    cfgs = []

    # Loop over the number of configurations
    for i in range(num_configs):
        
        model_name = sweep_configs_df.iloc[i]['model_name']
        
        params = {}
        
        if model_name == 'HistGradientBoosting':
            # Construct a dictionary of model parameters for each configuration
            params = {
                "learning_rate": sweep_configs_df.iloc[i]['learning_rate'],
                "max_depth": int(sweep_configs_df.iloc[i]['max_depth']),
                "max_leaf_nodes": int(sweep_configs_df.iloc[i]['max_leaf_nodes']),
                "random_state": int(sweep_configs_df.iloc[i]['random_state']),
            }
        
        elif model_name == 'SVM':
            params = {
                "C": sweep_configs_df.iloc[i]['C'],
                "gamma": sweep_configs_df.iloc[i]['gamma'],
                "kernel": sweep_configs_df.iloc[i]['kernel'],
                "random_state": int(sweep_configs_df.iloc[i]['random_state']),
                "probability": True,
            }
        # Append a new configuration to the list, which includes the model name, number of features,
        # sorting metric, number of folds for cross-validation, and the model parameters
        oversampling = True
        if 'oversampling' in sweep_configs_df.columns and str(sweep_configs_df['oversampling'].iloc[i]) == 'False':
            oversampling = False
        
        model_config = {
            "model_name": sweep_configs_df.iloc[i]['model_name'],
            "num_features": sweep_configs_df.iloc[i]['num_features'],
            "sorting_metric": sweep_configs_df.iloc[i]['sorting_metric'],
            "num_folds": sweep_configs_df.iloc[i]['num_folds'],
            "oversampling": oversampling,
            "model_params": params
        }
                
        if 'feature_selection' in sweep_configs_df.columns and 'Boost' in str(sweep_configs_df.iloc[i]['feature_selection']):
            model_config['feature_selection'] = sweep_configs_df.iloc[i]['feature_selection']
        
        cfgs.append(model_config)

    scaler = MinMaxScaler()
    
    if 'scale' in cfg.config_file.lower():
        scaler = None
    
    if 'scaler' in sweep_configs_df.columns:
        scaler_values = sweep_configs_df['scaler'].unique()
        # remove nan from the list
        scaler_values = [x for x in scaler_values if str(x) != 'nan']
        assert len(scaler_values) <= 1, "More than one scaler specified in the configuration file"
        
        if len(scaler_values) == 1:  
            if scaler_values[0] == 'MinMaxScaler':
                scaler = MinMaxScaler()
            elif scaler_values[0] == 'StandardScaler':
                scaler = StandardScaler()
    
    mp = True
    au = True
    
    if 'MP' in sweep_configs_df.columns:
    
        feature_enabled_MP_values = sweep_configs_df['MP'].unique()
        feature_enabled_MP_values = [x for x in feature_enabled_MP_values if str(x) != 'nan']
        assert len(feature_enabled_MP_values) <= 1, "More than one MP feature enabled specified in the configuration file"
        if len(feature_enabled_MP_values) == 1:
            mp = feature_enabled_MP_values[0]
    
    if 'AU' in sweep_configs_df.columns:
        feature_enabled_AU_values = sweep_configs_df['AU'].unique() 
        # remove nan from the list
        feature_enabled_AU_values = [x for x in feature_enabled_AU_values if str(x) != 'nan']
        assert len(feature_enabled_AU_values) <= 1, "More than one AU feature enabled specified in the configuration file"
        if len(feature_enabled_AU_values) == 1:
            au = feature_enabled_AU_values[0]
    
    feature_enabled = {'MP': mp, 'AU': au}
    

    
    
    result = ensemble_mode_external_test_set(
        arg, cfgs, df, df_test, model_objects, scaler_method=scaler, feature_enabled=feature_enabled, draw_plots_flag=cfg.draw_plots_flag, store_predictions=cfg.store_predictions)  # type: ignore
    # # Append the ensemble model's results to the final results string
    # result_str += ensemble_result_str + '\n' + '-'*140 + '\n'
    # csv_str += ensemble_result_csv_str + '\n'
    
    return result


def main():
    ENABLE_WANDB = False
    
    parser = argparse.ArgumentParser(description='Run a model')
    
    
    parser.add_argument('--num_configs', type=int, default=6, help='Number of configurations to use for ensemble')
    parser.add_argument('--random_state', type=int, default=42, help='Seed for the random number generator')
    parser.add_argument('--num_fold', type=int, default=10, help='Number of folds for cross-validation')
    parser.add_argument('--sorting_metric', type=str, default='F1 Score (PD)', help='Sorting metric to be used in sort the configurations')
    parser.add_argument('--name', type=str, default='clinic', help='Name of the ensemble method')
    parser.add_argument('--config_file', type=str, default='external-clinical-hist-configs.csv', help='Path to the configuration file')
    parser.add_argument('--C', type=float, default=14.059719687753034, help='Regularization parameter for the model')
    parser.add_argument('--max_iter', type=int, default=351, help='Maximum number of iterations for the model to converge')
    parser.add_argument('--penalty', type=str, default='l2', help='Type of regularization penalty to be applied on the model')
    parser.add_argument('--solver', type=str, default='newton-cholesky', help='Algorithm to be used in the optimization problem')
    parser.add_argument('--tol', type=float, default=0.0006246709267555428, help='Tolerance for stopping criteria')
    parser.add_argument('--draw_plots_flag', type=str, default='True', help='Flag to draw plots')
    parser.add_argument('--store_predictions', type=str, default='True', help='Flag to store predictions')

    args = parser.parse_args()
    
    args.draw_plots_flag = args.draw_plots_flag.lower() == 'true'
    args.store_predictions = args.store_predictions.lower() == 'true'

    args = parser.parse_args()
    
    # Create a dictionary with dot notation access
    df = pd.read_csv("dataset-cross-validation.csv")
    
    df_test = pd.read_csv("dataset_clinical.csv")
    
    # print(df_test[df_test['Diagnosis'] == 1]['ID'].nunique())
    # print(df_test[df_test['Diagnosis'] == 0]['ID'].nunique())
    
    # print(df_test['ID'].nunique())
    
    
    result = run_ensemble_models(df, args, df_test)
    
    print(result)
    
    if ENABLE_WANDB:
        wandb.init(project="smile-external-ensemble-lr")
        wandb.config.update(args)
        wandb.log(result)
    
    
    

if __name__ == "__main__":
    main()

