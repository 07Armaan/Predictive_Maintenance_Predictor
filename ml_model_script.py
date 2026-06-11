# %%
import pandas as pd
import numpy as np

# %%
df = pd.read_csv("predictive_maintenance.csv")

# %% [markdown]
# Data Understanding

# %%
df.head()

# %%
df.sample(10)

# %%
df.shape

# %%
df = df.drop(columns=["UDI","Product ID","Target"])

# %%
df.shape

# %%
df.duplicated().sum()

# %%
df.isnull().sum()

# %%
df.shape

# %%
df.info()

# %%
df.dtypes

# %%
df.describe()

# %%
df["Failure Type"].value_counts()

# %%
df["Failure Type"].value_counts(normalize=True)*100

# %%
df = df[df["Failure Type"] != "Random Failures"]
df = df[df["Failure Type"] != "Tool Wear Failure"]

# %%
df["Failure Type"].value_counts()

# %%
df["Failure Type"].value_counts(normalize=True)*100

# %%
df.select_dtypes(include=["int64"]).columns

# %%
df.select_dtypes(include=["float64"]).columns

# %%
df.select_dtypes(include=["str"]).columns

# %%
num_cols = df.select_dtypes(include=["int64","float64"]).columns
cat_cols = df.select_dtypes(include=["str"]).drop(columns=["Failure Type"]).columns

# %%
tar_col = df["Failure Type"]

# %% [markdown]
# EDA

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# Univariate Analysis

# %%
for col in num_cols:
    sns.histplot(x=col,data=df,kde=True)
    plt.title(col)
    plt.show()

# %%
for col in num_cols:
    sns.boxplot(x=col,data=df)
    plt.title(col)
    plt.show()

# %%
for col in cat_cols:
    sns.countplot(x=col,data=df)
    plt.title(col)
    plt.show()

# %%
sns.histplot(x=tar_col,kde=True)
plt.title("Target col")
plt.show()

# %%
sns.boxplot(x=tar_col)
plt.title("Target col")
plt.show()

# %% [markdown]
# Bivariate Analysis

# %%
for col in num_cols:
    sns.violinplot(x=col,y=tar_col,data=df)
    plt.title(col)
    plt.show()

# %%
for col in cat_cols:
    sns.countplot(hue=col,x=tar_col,data=df)
    plt.title(col)
    plt.show()

# %% [markdown]
# Multivariate Analysis

# %%
sns.heatmap(df[num_cols].corr(),annot=True,cmap="Blues")

# %% [markdown]
# Preprocessing

# %%
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import  ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from ouliers import outliers_handler
import optuna

# %%
df.sample(1)

# %%
x = df.drop(columns=["Failure Type"])
y = df["Failure Type"]

# %%
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

# %%
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# %%
for i in range(len(le.classes_)):
    print(f"{i} : {le.classes_[i]}")

# %%
num_pipeline = Pipeline(steps=[
    ("outliers_handler",outliers_handler()),
    ("scaling",StandardScaler())
])

# %%
cat_pipeline = Pipeline(steps=[
    ("ohe",OneHotEncoder(drop="first",handle_unknown="ignore"))
])

# %%
preprocessing = ColumnTransformer(transformers=[
    ("num_pipeline",num_pipeline,num_cols),
    ("cat_pipeline",cat_pipeline,cat_cols)
])

# %%
def objective(trial):
    model_name = trial.suggest_categorical("model", ["dt", "rf", "xgb"])


    if model_name == "dt":
        model = DecisionTreeClassifier(
        max_depth=trial.suggest_int("max_depth", 3, 20),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
        criterion=trial.suggest_categorical("criterion", ["gini", "entropy"])
    )

    elif model_name == "rf":
        model = RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 500),
        max_depth=trial.suggest_int("max_depth", 5, 30),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        n_jobs=-1
    )

    elif model_name == "xgb":
        model = XGBClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 500),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        gamma=trial.suggest_float("gamma", 0, 5),
        reg_alpha=trial.suggest_float("reg_alpha", 0, 5),
        reg_lambda=trial.suggest_float("reg_lambda", 0, 5),
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1,
        verbosity=0
    )
        
    pipeline = Pipeline(steps=[
        ("preprocessing",preprocessing),
        ("SMOTE",SMOTE(random_state=42,k_neighbors=2)),
        ("model",model)
    ])

    score = cross_val_score(pipeline,x_test,y_test,cv=5,scoring="f1_macro")
    return score.mean()

# %%
study = optuna.create_study(direction="maximize")
study.optimize(objective,n_trials=100)

# %%
params = study.best_params

# %%
print(params)

# %%
model_name = params["model"]

if model_name == "dt":
    final_model = DecisionTreeClassifier(
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        criterion=params["criterion"]
    )

elif model_name == "rf":
    final_model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        n_jobs=-1
    )

elif model_name == "xgb":
    final_model = XGBClassifier(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        gamma=params["gamma"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1,
        verbosity=0
    )


# %%
final_pipeline = Pipeline(steps=[
    ("preprocessing",preprocessing),
    ("SMOTE",SMOTE(random_state=42,k_neighbors=2)),
    ("final_model",final_model)
])

# %% [markdown]
# Training tuned model

# %%
final_pipeline.fit(x_train,y_train)

# %% [markdown]
# Predicting on test data

# %%
y_train_pred = final_pipeline.predict(x_train)
y_test_pred = final_pipeline.predict(x_test)

# %% [markdown]
# Metrics

# %%
train_acc = accuracy_score(y_train,y_train_pred)
print(f"Train acc: {train_acc}")
test_acc = accuracy_score(y_test,y_test_pred)
print(f"Test acc: {test_acc}")

# %%
train_cm = confusion_matrix(y_train,y_train_pred)
print(f"Train cm: \n{train_cm}")
sns.heatmap(train_cm,annot=True,fmt=".2f",cmap="Blues")
plt.title("Train Confusion Matrix")
plt.savefig("train_confusion_matrix.png")
plt.show()
plt.close()

# %%
test_cm = confusion_matrix(y_test,y_test_pred)
print(f"\nTest cm: \n{test_cm}")
sns.heatmap(test_cm,annot=True,fmt=".2f",cmap="Blues")
plt.title("Test Confusion Matrix")
plt.savefig("test_confusion_matrix.png")
plt.show()
plt.close()

# %%
train_cr = classification_report(y_train,y_train_pred,target_names=le.classes_)
print(f"Train cr: \n{train_cr}")
with open("train_class_report.txt","w") as f:
    f.write(train_cr)

# %%
test_cr = classification_report(y_test,y_test_pred,target_names=le.classes_)
print(f"\nTest cr: \n{test_cr}")
with open("test_class_report.txt","w") as f:
    f.write(test_cr)

# %% [markdown]
# Saving important thing

# %%
import joblib

# %%
joblib.dump(final_pipeline,"final_pipeline.pkl")

# %%
joblib.dump(le,"le.pkl")

# %% [markdown]
# SHAP

# %%
import shap 

# %%
shap_preprocessing = final_pipeline.named_steps["preprocessing"]
shap_model = final_pipeline.named_steps["final_model"]

# %%
feature_names = []
for col in shap_preprocessing.get_feature_names_out():
    feature_names.append(col.split("__")[-1])

# %%
feature_names

# %%
x_test_t = shap_preprocessing.transform(x_test)

# %%
explainer = shap.TreeExplainer(shap_model)

# %%
shap_values = explainer(x_test_t)

# %%
shap_values.shape

# %%
for i in range(len(le.classes_)):
    print(le.classes_[i])
    shap.plots.beeswarm(shap_values[:,:,i])

# %%
for i in range(len(le.classes_)):
    print(le.classes_[i])
    shap.plots.bar(shap_values[:,:,i])

# %% [markdown]
# MLFLOW

# %%
import mlflow

# %%
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Predictive_Maintenance_Model")
with mlflow.start_run():

    mlflow.log_params(params)

    mlflow.log_metric("train_accuracy",train_acc)
    mlflow.log_metric("test_accuracy",test_acc)

    mlflow.log_artifact("train_class_report.txt")
    mlflow.log_artifact("test_class_report.txt")
    mlflow.log_artifact("train_confusion_matrix.png")
    mlflow.log_artifact("test_confusion_matrix.png")

    mlflow.sklearn.log_model(final_pipeline,artifact_path="final_pipeline",registered_model_name="PredictiveMaintenance")

# %%



