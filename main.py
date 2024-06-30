import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from ydata_profiling import ProfileReport
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, \
    ExtraTreesClassifier, BaggingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from xgboost import XGBClassifier

# Load the dataset
df = pd.read_csv("smart_home_device_usage_data.csv")

# Drop the "UserID" column
df = df.drop(["UserID"], axis=1)

# Define the target variable
target = "SmartHomeEfficiency"
x = df.drop(target, axis=1)
y = df[target]

# Identify numerical and categorical columns
num_col = list(x.select_dtypes(["float", "int"]).columns)
cat_col = list(x.select_dtypes(["object"]).columns)


# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=df[target], random_state=1)

# Print column information
print("Numerical Columns:", num_col)
print("Categorical Columns:", cat_col)

# Define the column transformer
transformer = ColumnTransformer(
    transformers=[
        ("cat_trans", OneHotEncoder(), cat_col),
        ("num_trans", RobustScaler(), num_col)
    ],
    remainder="passthrough"
)

# Define the pipeline
model = Pipeline(
    steps=[
        ("transform", transformer),
        ("model", XGBClassifier(random_state=1))
    ]
)


models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM Linear': SVC(kernel='linear'),
    'SVM RBF': SVC(kernel='rbf'),
    'Gradient Boosting': GradientBoostingClassifier(),
    'KNN': KNeighborsClassifier(),
    'XGBoost': XGBClassifier(eval_metric='logloss'),
    'AdaBoost': AdaBoostClassifier(random_state=1),
    'Extra Trees': ExtraTreesClassifier(),
    'MLP Classifier': MLPClassifier(max_iter=1000),
    'Bagging Classifier': BaggingClassifier(),
    'LGB Classifier' : LGBMClassifier()

}

result = {}

for models_name, model in models.items():
    model = Pipeline(
        steps=[
            ('transform', transformer),
            ('model', model)
        ]
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    report = classification_report(y_test, y_pred, zero_division = np.nan)
    print(models_name)
    print(report)
    print("****************************")
    accuracy = accuracy_score(y_test, y_pred)
    result[models_name] = accuracy*100


x = list(result.keys())
y = list(result.values())
plt.figure(figsize=(20, 20))
sns.barplot(x=x, y=y, hue=x, palette='magma', dodge=False)
plt.title('Model Comparison')
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.ylim(0, 100)
plt.yticks(range(0, 101, 10))
plt.legend([], [], frameon=False)
plt.show()




# Fit the transformer part of the pipeline
# model.fit(x_train, y_train)

# Transform the training data
# x_train_transformed = model.named_steps['transform'].transform(x_train)
#
# # Convert the transformed data back to a DataFrame
# transformed_features = model.named_steps['transform'].get_feature_names_out()
# x_train_transformed_df = pd.DataFrame(x_train_transformed, columns=transformed_features)

# Display the transformed DataFrame
# print(x_train_transformed_df.head())
#
# # Define the parameter grid for GridSearchCV
# # params = {
# #     "model__n_estimators": [50, 100, 150, 200, 250],
# #     "model__max_depth": [None, 2, 4, 6, 8, 20, 30, 40]
# # }
#
# # Set up GridSearchCV
# # grid_search = GridSearchCV(estimator=model, param_grid=params, cv=6, scoring="accuracy", n_jobs=-1, verbose=2)
#
# # Fit the model
# # grid_search.fit(x_train, y_train)
#
# # Get the best model
# best_model = grid_search.best_estimator_
#
# # Make predictions with the best model
# y_pred = best_model.predict(x_test)
#
# # Print the classification report
# from sklearn.metrics import classification_report
# report = classification_report(y_test, y_pred)
# print(report)
#
# # Print the best score and parameters
# print(grid_search.best_score_)
# print(grid_search.best_params_)



