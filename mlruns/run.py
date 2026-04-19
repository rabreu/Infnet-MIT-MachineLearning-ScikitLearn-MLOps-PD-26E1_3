import mlflow

def run():
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    mlflow.sklearn.autolog()
    mkflow

if __name__ == "__main__":
    run()
