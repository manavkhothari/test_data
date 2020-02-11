import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score


def plot(df, column, X='', Y_pred='', title="Plot"):
    plt.clf()
    plt.plot(X, Y_pred, color='red')
    plt.plot(df['day'], df[column])
    plt.title(title + " " + column)
    plt.ylabel("Visitors")
    plt.xlabel("Days")
    plt.grid(linestyle='--')
    # plt.show()
    plt.savefig("Prediction/" + title + " " +
                column + " Month.jpg", bbox_inches='tight')


def rmse(targets, predictions): return mse(targets, predictions) ** 0.5


data_file = "forecast1month"
df = pd.read_csv("data/" + data_file + ".csv")
# plot(df, "visitors", "", "", title="1 month")

X_train_tree = pd.get_dummies(np.array(df["day"] % 11))
X_day = np.array(df["day"]).reshape(-1, 1)

X_train = np.array(df["day"]).reshape(-1, 1)
pred_df = pd.DataFrame()

col = "visitors"
# ----------- Train - fit ------------------------
reg1 = LinearRegression().fit(X_day, df[col])
dtree = DecisionTreeRegressor(random_state=0)
dtree.fit(X_train_tree, df[col])

# ------------- X_test (1-36 days) -------------
X_test_day = (np.array(range(1, 41))).reshape(-1, 1)
X_test_tree = (np.array(range(1, 41)))

# ------------ Prediction (1-36 days) -----------
tree_pred = dtree.predict(pd.get_dummies(np.array(X_test_tree % 11)))
# print(tree_pred)
reg_pred1 = reg1.predict(X_test_day)
# print(reg_pred1)

# ------------- Calculate intermediate -------------
mean_val = np.mean(tree_pred)
tree_pred_norm = tree_pred - mean_val

# ------------- Combine - components ---------------
final_prediction = tree_pred_norm + reg_pred1
# print(final_prediction)
plot(df, col, X_test_tree.reshape(-1, 1), final_prediction, "Final Prediction")
