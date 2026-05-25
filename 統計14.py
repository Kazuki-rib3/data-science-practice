import seaborn as sns
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = sns.load_dataset("titanic")
df = df[["survived", "pclass", "sex", "age"]].dropna()
df["sex"] = df["sex"].map({"male": 0, "female": 1})

X = df[["pclass", "sex", "age"]]
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_score = accuracy_score(y_test, dt.predict(X_test))

rf = RandomForestClassifier(n_estimators=500, random_state=42)
rf.fit(X_train, y_train)
rf_score = accuracy_score(y_test, rf.predict(X_test))

print(f"決定木:      {dt_score:.2f}")
print(f"ランダムフォレスト: {rf_score:.2f}")

importances = pd.Series(rf.feature_importances_, index=["pclass","sex","age"])
print(importances.sort_values(ascending=False))