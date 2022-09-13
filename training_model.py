import pandas as pd 
import numpy as np 

lr = 0.01 # learning rate


def estimate_price(theta, x): #predict (w[0] + w[1] * x)
    return theta[0] + theta[1] * x

def gradient_descent(y_pred, df, x, tmp_theta):
    tmp_theta[0] -= lr / df["km"].shape[0] * np.sum(y_pred - df["km"]) 
    tmp_theta[1] -= lr / df["km"].shape[0] * np.sum((y_pred - df["km"]) * x)
    return tmp_theta

def train(df, epochs = 5000):
    tmp_theta = [ 0 for i in range(2)]
    x = 0
    for i in range(epochs):
        y_pred = estimate_price(tmp_theta, x) # 0 - вычислить что подставить
        tmp_theta = gradient_descent(y_pred, df, x,  tmp_theta0, tmp_theta1)


if __name__ == '__main__':
    df = pd.read_csv('data.csv', delimiter=',')
    #print(df.head())
    #print(df["km"])