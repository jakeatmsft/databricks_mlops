# Databricks notebook source
# MAGIC %md Copyright (c) Microsoft Corporation. All rights reserved.
# MAGIC 
# MAGIC Licensed under the MIT License.

# COMMAND ----------

# MAGIC %md ![Impressions](https://PixelServer20190423114238.azurewebsites.net/api/impressions/MachineLearningNotebooks/how-to-use-azureml/explain-model/azure-integration/run-history/save-retrieve-explanations-run-history.png)

# COMMAND ----------

# MAGIC %md # Save and retrieve explanations via Azure Machine Learning Run History
# MAGIC 
# MAGIC _**This notebook showcases how to use the Azure Machine Learning Interpretability SDK to save and retrieve classification model explanations to/from Azure Machine Learning Run History.**_
# MAGIC 
# MAGIC 
# MAGIC ## Table of Contents
# MAGIC 
# MAGIC 1. [Introduction](#Introduction)
# MAGIC 1. [Setup](#Setup)
# MAGIC 1. [Run model explainer locally at training time](#Explain)
# MAGIC     1. Apply feature transformations
# MAGIC     1. Train a binary classification model
# MAGIC     1. Explain the model on raw features
# MAGIC         1. Generate global explanations
# MAGIC         1. Generate local explanations
# MAGIC 1. [Upload model explanations to Azure Machine Learning Run History](#Upload)
# MAGIC 1. [Download model explanations from Azure Machine Learning Run History](#Download)
# MAGIC 1. [Visualize explanations](#Visualize)
# MAGIC 1. [Next steps](#Next)

# COMMAND ----------

# MAGIC %md ## Introduction
# MAGIC 
# MAGIC This notebook showcases how to explain a classification model predictions locally at training time, upload explanations to the Azure Machine Learning's run history, and download previously-uploaded explanations from the Run History.
# MAGIC It demonstrates the API calls that you need to make to upload/download the global and local explanations and a visualization dashboard that provides an interactive way of discovering patterns in data and downloaded explanations.
# MAGIC 
# MAGIC We will showcase three tabular data explainers: TabularExplainer (SHAP), MimicExplainer (global surrogate), and PFIExplainer.
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Problem: IBM employee attrition classification with scikit-learn (run model explainer locally and upload explanation to the Azure Machine Learning Run History)
# MAGIC 
# MAGIC 1. Train a SVM classification model using Scikit-learn
# MAGIC 2. Run 'explain_model' with AML Run History, which leverages run history service to store and manage the explanation data
# MAGIC ---
# MAGIC 
# MAGIC Setup: If you are using Jupyter notebooks, the extensions should be installed automatically with the package.
# MAGIC If you are using Jupyter Labs run the following command:
# MAGIC ```
# MAGIC (myenv) $ jupyter labextension install @jupyter-widgets/jupyterlab-manager
# MAGIC ```

# COMMAND ----------

# MAGIC %sh
# MAGIC conda install numpy
# MAGIC conda install pandas
# MAGIC conda install scikit-learn

# COMMAND ----------

dbutils.library.installPyPI("scipy")
dbutils.library.installPyPI("pyscaffold")
dbutils.library.installPyPI("interpret")
dbutils.library.installPyPI("azure")
dbutils.library.installPyPI("azureml-core")
dbutils.library.installPyPI("azure-storage")
dbutils.library.installPyPI("azure-storage-blob")
dbutils.library.installPyPI("azureml-defaults")
dbutils.library.installPyPI("azureml-contrib-explain-model")
dbutils.library.installPyPI("azureml-contrib-interpret")
dbutils.library.installPyPI("azureml-telemetry")
dbutils.library.installPyPI("azureml-interpret")
dbutils.library.installPyPI("sklearn-pandas")
dbutils.library.installPyPI("azureml-dataprep")

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ## Explain
# MAGIC 
# MAGIC ### Run model explainer locally at training time

# COMMAND ----------

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, Imputer
from sklearn.svm import SVC
import pandas as pd
import numpy as np

# Explainers:
# 1. SHAP Tabular Explainer
from interpret.ext.blackbox import TabularExplainer

# OR

# 2. Mimic Explainer
from interpret.ext.blackbox import MimicExplainer
# You can use one of the following four interpretable models as a global surrogate to the black box model
from interpret.ext.glassbox import LGBMExplainableModel
from interpret.ext.glassbox import LinearExplainableModel
from interpret.ext.glassbox import SGDExplainableModel
from interpret.ext.glassbox import DecisionTreeExplainableModel

# OR

# 3. PFI Explainer
from interpret.ext.blackbox import PFIExplainer 

# COMMAND ----------

# MAGIC %md ### Load the IBM employee attrition data

# COMMAND ----------

# get the IBM employee attrition dataset
outdirname = 'dataset.6.21.19'
try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve
import zipfile
zipfilename = outdirname + '.zip'
urlretrieve('https://publictestdatasets.blob.core.windows.net/data/' + zipfilename, zipfilename)
with zipfile.ZipFile(zipfilename, 'r') as unzip:
    unzip.extractall('.')
attritionData = pd.read_csv('./WA_Fn-UseC_-HR-Employee-Attrition.csv')

# Dropping Employee count as all values are 1 and hence attrition is independent of this feature
attritionData = attritionData.drop(['EmployeeCount'], axis=1)
# Dropping Employee Number since it is merely an identifier
attritionData = attritionData.drop(['EmployeeNumber'], axis=1)

attritionData = attritionData.drop(['Over18'], axis=1)

# Since all values are 80
attritionData = attritionData.drop(['StandardHours'], axis=1)

# Converting target variables from string to numerical values
target_map = {'Yes': 1, 'No': 0}
attritionData["Attrition_numerical"] = attritionData["Attrition"].apply(lambda x: target_map[x])
target = attritionData["Attrition_numerical"]

attritionXData = attritionData.drop(['Attrition_numerical', 'Attrition'], axis=1)

# COMMAND ----------

# Split data into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(attritionXData, 
                                                    target, 
                                                    test_size = 0.2,
                                                    random_state=0,
                                                    stratify=target)

# COMMAND ----------

# Creating dummy columns for each categorical feature
categorical = []
for col, value in attritionXData.iteritems():
    if value.dtype == 'object':
        categorical.append(col)
        
# Store the numerical columns in a list numerical
numerical = attritionXData.columns.difference(categorical)        

# COMMAND ----------

# MAGIC %md ### Transform raw features

# COMMAND ----------

# MAGIC %md We can explain raw features by either using a `sklearn.compose.ColumnTransformer` or a list of fitted transformer tuples. The cell below uses `sklearn.compose.ColumnTransformer`. In case you want to run the example with the list of fitted transformer tuples, comment the cell below and uncomment the cell that follows after. 

# COMMAND ----------

from sklearn.compose import ColumnTransformer

# We create the preprocessing pipelines for both numeric and categorical data.
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

transformations = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical),
        ('cat', categorical_transformer, categorical)])

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
clf = Pipeline(steps=[('preprocessor', transformations),
                      ('classifier', SVC(kernel='linear', C = 1.0, probability=True))])

# COMMAND ----------

'''
# Uncomment below if sklearn-pandas is not installed
#!pip install sklearn-pandas
from sklearn_pandas import DataFrameMapper

# Impute, standardize the numeric features and one-hot encode the categorical features.    


numeric_transformations = [([f], Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])) for f in numerical]

categorical_transformations = [([f], OneHotEncoder(handle_unknown='ignore', sparse=False)) for f in categorical]

transformations = numeric_transformations + categorical_transformations

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
clf = Pipeline(steps=[('preprocessor', transformations),
                      ('classifier', SVC(kernel='linear', C = 1.0, probability=True))]) 



'''

# COMMAND ----------

# MAGIC %md ### Train a SVM classification model, which you want to explain

# COMMAND ----------

model = clf.fit(x_train, y_train)

# COMMAND ----------

# MAGIC %md ### Explain predictions on your local machine

# COMMAND ----------

# 1. Using SHAP TabularExplainer
# clf.steps[-1][1] returns the trained classification model
explainer = TabularExplainer(clf.steps[-1][1], 
                             initialization_examples=x_train, 
                             features=attritionXData.columns, 
                             classes=["Not leaving", "leaving"], 
                             transformations=transformations)




# 2. Using MimicExplainer
# augment_data is optional and if true, oversamples the initialization examples to improve surrogate model accuracy to fit original model.  Useful for high-dimensional data where the number of rows is less than the number of columns. 
# max_num_of_augmentations is optional and defines max number of times we can increase the input data size.
# LGBMExplainableModel can be replaced with LinearExplainableModel, SGDExplainableModel, or DecisionTreeExplainableModel
# explainer = MimicExplainer(clf.steps[-1][1], 
#                            x_train, 
#                            LGBMExplainableModel, 
#                            augment_data=True, 
#                            max_num_of_augmentations=10, 
#                            features=attritionXData.columns, 
#                            classes=["Not leaving", "leaving"], 
#                            transformations=transformations)





# 3. Using PFIExplainer

# Use the parameter "metric" to pass a metric name or function to evaluate the permutation. 
# Note that if a metric function is provided a higher value must be better.
# Otherwise, take the negative of the function or set the parameter "is_error_metric" to True.
# Default metrics: 
# F1 Score for binary classification, F1 Score with micro average for multiclass classification and
# Mean absolute error for regression

# explainer = PFIExplainer(clf.steps[-1][1], 
#                          features=x_train.columns, 
#                          transformations=transformations,
#                          classes=["Not leaving", "leaving"])

# COMMAND ----------

# MAGIC %md ### Generate global explanations
# MAGIC Explain overall model predictions (global explanation)

# COMMAND ----------

# Passing in test dataset for evaluation examples - note it must be a representative sample of the original data
# x_train can be passed as well, but with more examples explanations will take longer although they may be more accurate
global_explanation = explainer.explain_global(x_test)

# Note: if you used the PFIExplainer in the previous step, use the next line of code instead
# global_explanation = explainer.explain_global(x_test, true_labels=y_test)

# COMMAND ----------

# Sorted SHAP values
print('ranked global importance values: {}'.format(global_explanation.get_ranked_global_values()))
# Corresponding feature names
print('ranked global importance names: {}'.format(global_explanation.get_ranked_global_names()))
# Feature ranks (based on original order of features)
print('global importance rank: {}'.format(global_explanation.global_importance_rank))

# Note: PFIExplainer does not support per class explanations
# Per class feature names
print('ranked per class feature names: {}'.format(global_explanation.get_ranked_per_class_names()))
# Per class feature importance values
print('ranked per class feature values: {}'.format(global_explanation.get_ranked_per_class_values()))

# COMMAND ----------

# Print out a dictionary that holds the sorted feature importance names and values
print('global importance rank: {}'.format(global_explanation.get_feature_importance_dict()))

# COMMAND ----------

# MAGIC %md ### Explain overall model predictions as a collection of local (instance-level) explanations

# COMMAND ----------

# feature shap values for all features and all data points in the training data
print('local importance values: {}'.format(global_explanation.local_importance_values))

# COMMAND ----------

# MAGIC %md ### Generate local explanations
# MAGIC Explain local data points (individual instances)

# COMMAND ----------

# Note: PFIExplainer does not support local explanations
# You can pass a specific data point or a group of data points to the explain_local function

# E.g., Explain the first data point in the test set
instance_num = 1
local_explanation = explainer.explain_local(x_test[:instance_num])

# COMMAND ----------

# Get the prediction for the first member of the test set and explain why model made that prediction
prediction_value = clf.predict(x_test)[instance_num]

sorted_local_importance_values = local_explanation.get_ranked_local_values()[prediction_value]
sorted_local_importance_names = local_explanation.get_ranked_local_names()[prediction_value]

print('local importance values: {}'.format(sorted_local_importance_values))
print('local importance names: {}'.format(sorted_local_importance_names))

# COMMAND ----------

# MAGIC %md ## Upload
# MAGIC Upload explanations to Azure Machine Learning Run History

# COMMAND ----------

import azureml.core
from azureml.core import Workspace, Experiment, Run
from interpret.ext.blackbox import TabularExplainer
from azureml.contrib.interpret.explanation.explanation_client import ExplanationClient
# Check core SDK version number
print("SDK version:", azureml.core.VERSION)

# COMMAND ----------

# MAGIC %md
# MAGIC <h1><a href ='https://microsoft.com/devicelogin' target ='_new' >Device Login Link</a></h1>

# COMMAND ----------

import os

subscription_id = os.getenv("SUBSCRIPTION_ID", default="7fd76d0f-84f2-498b-a997-e0d059af5ce1")
resource_group = os.getenv("RESOURCE_GROUP", default="sdbolts-AML-RG")
workspace_name = os.getenv("WORKSPACE_NAME", default="sdbolts-AML-WS")
workspace_region = os.getenv("WORKSPACE_REGION", default="centralus")

from azureml.core import Workspace

try:
    ws = Workspace(subscription_id = subscription_id, resource_group = resource_group, workspace_name = workspace_name)
    # write the details of the workspace to a configuration file to the notebook library
    ws.write_config()
    print("Workspace configuration succeeded. Skip the workspace creation steps below")
except:
    print("Workspace not accessible. Change your parameters or create a new workspace below")

# COMMAND ----------

ws = Workspace.from_config()
print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep = '\n')

# COMMAND ----------

experiment_name = 'explain_model'
experiment = Experiment(ws, experiment_name)
run = experiment.start_logging()
client = ExplanationClient.from_run(run)

# COMMAND ----------

# Uploading model explanation data for storage or visualization in webUX
# The explanation can then be downloaded on any compute
# Multiple explanations can be uploaded
client.upload_model_explanation(global_explanation, comment='global explanation: all features')
# Or you can only upload the explanation object with the top k feature info
#client.upload_model_explanation(global_explanation, top_k=2, comment='global explanation: Only top 2 features')

# COMMAND ----------

# Uploading model explanation data for storage or visualization in webUX
# The explanation can then be downloaded on any compute
# Multiple explanations can be uploaded
client.upload_model_explanation(local_explanation, comment='local explanation for test point 1: all features')

# Alterntively, you can only upload the local explanation object with the top k feature info
#client.upload_model_explanation(local_explanation, top_k=2, comment='local explanation: top 2 features')

# COMMAND ----------

run.complete()

# COMMAND ----------

# MAGIC %md ## Download
# MAGIC Download explanations from Azure Machine Learning Run History

# COMMAND ----------

# List uploaded explanations
client.list_model_explanations()

# COMMAND ----------

for explanation in client.list_model_explanations():
    
    if explanation['comment'] == 'local explanation for test point 1: all features':
        downloaded_local_explanation = client.download_model_explanation(explanation_id=explanation['id'])
        # You can pass a k value to only download the top k feature importance values
        downloaded_local_explanation_top2 = client.download_model_explanation(top_k=2, explanation_id=explanation['id'])
    
    
    elif explanation['comment'] == 'global explanation: all features':
        downloaded_global_explanation = client.download_model_explanation(explanation_id=explanation['id'])
        # You can pass a k value to only download the top k feature importance values
        downloaded_global_explanation_top2 = client.download_model_explanation(top_k=2, explanation_id=explanation['id'])
    

# COMMAND ----------

# MAGIC %md ## Visualize
# MAGIC Load the visualization dashboard

# COMMAND ----------

from azureml.contrib.interpret.visualize import ExplanationDashboard

# COMMAND ----------

ExplanationDashboard(downloaded_global_explanation, model, x_test)

# COMMAND ----------

# MAGIC %md ## Next
# MAGIC Learn about other use cases of the explain package on a:
# MAGIC 1. [Training time: regression problem](../../tabular-data/explain-binary-classification-local.ipynb)       
# MAGIC 1. [Training time: binary classification problem](../../tabular-data/explain-binary-classification-local.ipynb)
# MAGIC 1. [Training time: multiclass classification problem](../../tabular-data/explain-multiclass-classification-local.ipynb)
# MAGIC 1. Explain models with engineered features:
# MAGIC     1. [Simple feature transformations](../../tabular-data/simple-feature-transformations-explain-local.ipynb)
# MAGIC     1. [Advanced feature transformations](../../tabular-data/advanced-feature-transformations-explain-local.ipynb)
# MAGIC 1. [Run explainers remotely on Azure Machine Learning Compute (AMLCompute)](../remote-explanation/explain-model-on-amlcompute.ipynb)
# MAGIC 1. Inferencing time: deploy a classification model and explainer:
# MAGIC     1. [Deploy a locally-trained model and explainer](../scoring-time/train-explain-model-locally-and-deploy.ipynb)
# MAGIC     1. [Deploy a remotely-trained model and explainer](../scoring-time/train-explain-model-on-amlcompute-and-deploy.ipynb)

# COMMAND ----------

