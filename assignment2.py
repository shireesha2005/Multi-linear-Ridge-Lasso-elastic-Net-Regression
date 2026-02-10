#import librabries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.metrics import mean_squared_error

#giving the real world data 
df = pd.read_csv(r"C:\Users\shire\Downloads\Shireesha_dataset_.csv")
X = df.drop("Price",axis=1)
y = df["Price"]

# splittign the data into train and test 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#scaling the freatures 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#creating model 
mlr_Linear_model = Pipeline([
    ("scaler",StandardScaler()),
    ("multilinear",LinearRegression())
])
ridge_model = Pipeline([
    ("scaler",StandardScaler()),
    ("ridge",Ridge())
])
lasso_model = Pipeline([
    ("scaler",StandardScaler()),
    ("lasso",Lasso(max_iter=5000))
])
elasticnet_model = Pipeline([
    ("scaler",StandardScaler()),
    ("elasticnet",ElasticNet(l1_ratio=0.5,max_iter=5000))
])

#training models
mlr_Linear_model.fit(X_train_scaled, y_train)
ridge_model.fit(X_train_scaled, y_train)
lasso_model.fit(X_train_scaled,y_train)
elasticnet_model.fit(X_train_scaled,y_train)

#predicting the model 
y_pred_linear_test = mlr_Linear_model.predict(X_test)
y_pred_ridge_test = ridge_model.predict(X_test)
y_pred_lasso_test = lasso_model.predict(X_test)
y_pred_elasticnet_model = elasticnet_model.predict(X_test)

#caluculate mse
mse_Linear_train = mean_squared_error(y_train, mlr_Linear_model.predict(X_train))
mse_ridge_train = mean_squared_error(y_train, ridge_model.predict(X_train))
mse_lasso_train = mean_squared_error(y_train,lasso_model.predict(X_train))
mse_elasticnet_train = mean_squared_error(y_train,elasticnet_model.predict(X_train))

mse_linear_test = mean_squared_error(y_test, y_pred_linear_test)
mse_ridge_test = mean_squared_error(y_test,y_pred_ridge_test)
mse_lasso_test = mean_squared_error(y_test,y_pred_lasso_test)
mse_elasticnet_test = mean_squared_error(y_test,y_pred_elasticnet_model)

#visualization part
alphas = [0.001, 0.01, 0.1, 1, 10, 100]

ridge_train, ridge_test = [], []
lasso_train, lasso_test = [], []
enet_train, enet_test = [], []

for a in alphas:
    # Ridge
    ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=a))
    ])
    ridge.fit(X_train, y_train)
    ridge_train.append(mean_squared_error(y_train, ridge.predict(X_train)))
    ridge_test.append(mean_squared_error(y_test, ridge.predict(X_test)))

    # Lasso
    lasso = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", Lasso(alpha=a, max_iter=5000))
    ])
    lasso.fit(X_train, y_train)
    lasso_train.append(mean_squared_error(y_train, lasso.predict(X_train)))
    lasso_test.append(mean_squared_error(y_test, lasso.predict(X_test)))

    # Elastic Net
    enet = Pipeline([
        ("scaler", StandardScaler()),
        ("elastic", ElasticNet(alpha=a, l1_ratio=0.5, max_iter=5000))
    ])
    enet.fit(X_train, y_train)
    enet_train.append(mean_squared_error(y_train, enet.predict(X_train)))
    enet_test.append(mean_squared_error(y_test, enet.predict(X_test)))
plt.figure(figsize=(10,6))

plt.plot(alphas, ridge_train, label="Ridge Train")
plt.plot(alphas, ridge_test, label="Ridge Test")

plt.plot(alphas, lasso_train, label="Lasso Train")
plt.plot(alphas, lasso_test, label="Lasso Test")

plt.plot(alphas, enet_train, label="ElasticNet Train")
plt.plot(alphas, enet_test, label="ElasticNet Test")

plt.xscale("log")
plt.xlabel("Regularization Strength (alpha)")
plt.ylabel("Mean Squared Error")
plt.title("Training vs Testing Error for Different Regularization Strengths")
plt.legend()
plt.show()
 #coefficent shrinkage path
ridge_coefs = []
lasso_coefs = []
enet_coefs = []

for a in alphas:
    # Ridge
    ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=a))
    ])
    ridge.fit(X_train, y_train)
    ridge_coefs.append(ridge.named_steps["ridge"].coef_)

    # Lasso
    lasso = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", Lasso(alpha=a, max_iter=5000))
    ])
    lasso.fit(X_train, y_train)
    lasso_coefs.append(lasso.named_steps["lasso"].coef_)

    # Elastic Net
    enet = Pipeline([
        ("scaler", StandardScaler()),
        ("elastic", ElasticNet(alpha=a, l1_ratio=0.5, max_iter=5000))
    ])
    enet.fit(X_train, y_train)
    enet_coefs.append(enet.named_steps["elastic"].coef_)
plt.figure(figsize=(10,6))

# Ridge paths
for i in range(len(ridge_coefs[0])):
    plt.plot(alphas, [coef[i] for coef in ridge_coefs], linestyle="--")

# Lasso paths
for i in range(len(lasso_coefs[0])):
    plt.plot(alphas, [coef[i] for coef in lasso_coefs])

# Elastic Net paths
for i in range(len(enet_coefs[0])):
    plt.plot(alphas, [coef[i] for coef in enet_coefs], linestyle=":")

plt.xscale("log")
plt.xlabel("Regularization Strength (alpha)")
plt.ylabel("Coefficient Value")
plt.title("Coefficient Shrinkage Path: Ridge vs Lasso vs Elastic Net")
plt.show()






