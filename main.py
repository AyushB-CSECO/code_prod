# Import Librairies
# Data Processing Libraries
import pandas as pd 
# Machine Learning Libraries
from sklearn import datasets 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import precision_score, recall_score 
from sklearn.metrics import f1_score, roc_auc_score 
from sklearn.model_selection import train_test_split 
# MLOps Libraries
import mlflow 
from mlflow.models import infer_signature 

mlflow.set_tracking_uri("https://127.0.0.1:5000")  # Set the tracking URI

#Get Data 
data = pd.read_csv("Churn_Modelling.csv")
y = data.pop("Exited")
# X = data.drop(columns=['RowNumber','CustomerId'])
X = data.loc[:,['IsActiveMember', 'EstimatedSalary', 'CreditScore']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2
                                                    , random_state=42)

# Define model hyper parameters
params = {"penalty": "l2",
          "solver": "lbfgs",
          "max_iter": 1000,
          "multi_class": "auto",
          "random_state": 2}

# Model Building
lg_reg = LogisticRegression(**params)
lg_reg.fit(X_train, y_train)

y_pred = lg_reg.predict(X_test)
print(y_pred[0:10])

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print(f"Precision: {precision}")
print(f"Recall: {recall}")


# MLFlow Code
mlflow.set_experiment("Churn Prediction Experiment (CPE)")
with mlflow.start_run() as run:
    # Log the Hyperparameters
    mlflow.log_params(params)
    # Log Experiment Metrics
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    # Set a tag that we can use to remind ourselves what this run is for
    mlflow.set_tag("Traing Info", "Base Case Model for Customer Churn Prediction")

    #Infer the model signature
    signature = infer_signature(X_train, lg_reg.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model = lg_reg,
        artifact_path = "CPE_V1",
        signature = signature,
        input_example = X_train,
        registered_model_name = "CPE_V1_Model"
    )