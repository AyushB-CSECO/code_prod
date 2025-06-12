import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("Customer Churn Prediction") 

with mlflow.start_run() as run:
    mlflow.log_metric("Precision",1)
    mlflow.log_metric("Recall",2)
    mlflow.log_metric("F1 Score",3)
    mlflow.log_metric("ROC AUC",4)

