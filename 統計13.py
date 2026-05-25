import seaborn as sns 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = sns.load_dataset("titanic")

df = df[["survived", "pclass", "sex", "age"]].dropna()
df = df[["survived", "pclass", "sex", "age"]]

df["sex"] = df["sex"].map({"male":0, "female":1})

X = df[["pclass","sex","sex"]]
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
print(f"正解率: {accuracy_score(y_test, y_pred):.2f}")

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))
plot_tree(model, feature_names=["pclass", "sex", "age"], class_names=["死亡","生存"],filled=True)
plt.savefig("decision_tree.png")
print("画像を保存しました")