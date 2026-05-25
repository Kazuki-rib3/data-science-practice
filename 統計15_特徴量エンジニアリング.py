import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = sns.load_dataset("titanic")
df = df[["survived","pclass","sex","age","sibsp","parch","fare"]].dropna()

df["sex"] = df["sex"].map({"male":0, "female":1})

df["age_group"] = pd.cut(df["age"], bins=[0,16,60,100], labels=[0,1,2])
df["age_group"] = df["age_group"].cat.codes
df["family_size"] = df["sibsp"] + df["parch"]
df["pclass_reversed"] = 4 - df["pclass"]
df["fare_rank"] = pd.cut(df["fare"], bins=3, labels=[0,1,2])
df["fare_rank"] = df["fare_rank"].cat.codes
df["wealth_score"] = df["pclass_reversed"] * 3 + df["fare_rank"]

#X = df[["pclass", "sex", "age", "family_size", "age_group", "wealth_score"]]
#X = df[["pclass", "sex", "age", "family_size", "wealth_score"]]
X = df[["sex", "age_group", "family_size", "wealth_score"]]

y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_score = accuracy_score(y_test, rf.predict(X_test))

print(f"ランダムフォレスト（特徴量あり）:{rf_score:.2f}")

importances = pd.Series(rf.feature_importances_, index=X.columns)
print(importances.sort_values(ascending=False))
