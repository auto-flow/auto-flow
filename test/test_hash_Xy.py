from hash_problem import get_hash_of_Xy
from hyperflow import XYDataManager
import pandas as pd

df = pd.read_csv("../examples/classification/train_classification.csv")
column_descriptions = {
    "id": "PassengerId",
    "target": "Survived"
}
data_manager = XYDataManager(X=df, column_descriptions=column_descriptions)
hash_value = get_hash_of_Xy(data_manager.X_train, data_manager.y_train)
print(hash_value)