import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("data/insurance.csv")

dataset["sex"].replace({"male": 0, "female": 1}, inplace=True)
dataset["smoker"].replace({"yes": 1, "no": 0}, inplace=True)
dataset["region"].replace(
    {"southwest": 0, "southeast": 1, "northwest": 2, "northeast": 3}
)

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_dev, X_test, y_dev, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)
