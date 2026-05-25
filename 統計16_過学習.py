import seaborn as sns
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#データの読み込み
df = sns.load_dataset("titanic")
df =  df[["survived", "pclass", "sex","age", "sibsp", "parch", "fare"]].dropna()
df["sex"] = df["sex"].map({"male":0,"female" :1})
df["family_size"] = df["sibsp"] + df["parch"]

X = df[["pclass","sex", "age","family_size"]]
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#深さを変えて比較
for depth in[1,2,3,5,10,None]:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    train_score = accuracy_score(y_train, model.predict(X_train))
    test_score = accuracy_score(y_test, model.predict(X_test))
    print(f"max_depth={str(depth):>4} | 訓練 {train_score:.2f} | テスト {test_score:.2f}")

from sklearn.model_selection import cross_val_score

print("\n--- 交差検証 ---")
for depth in [1,2,3,5,10,None]:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    scores =cross_val_score(model,X,y,cv=5)
    print(f"max_depth={str(depth):>4} | 平均 {scores.mean():.2f} | 標準偏差: {scores.std():.2f}")