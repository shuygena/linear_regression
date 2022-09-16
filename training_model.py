import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

lr = 0.01 

def plot_data_graph(df):
    plt.scatter(df["km"], df["price"], color='b') 
    plt.title("Dataset") 
    plt.xlabel("mileage") 
    plt.ylabel("price") 
    plt.savefig('data.png')

    
def plot_line(df, y_pred):
    plt.scatter(df["km"], df["price"], color='b')
    plt.plot(df["km"], y_pred, color="r")
    plt.title("Regression line")
    plt.xlabel("mileage") 
    plt.ylabel("price")
    plt.savefig('result.png')
    
def estimate_price(theta, x):
    return theta[0] + theta[1] * x

def gradient_descent(y_pred, df, x, tmp_theta): 
    tmp_theta[0] -= lr / df.shape[0] * np.sum(y_pred - df["price"]) 
    tmp_theta[1] -= lr / df.shape[0] * np.sum((y_pred - df["price"]) * x)
    return tmp_theta

def min_max_scaler(x, dataset):
    x_min = dataset.min()
    x_max = dataset.max()
    return (x - x_min) / (x_max - x_min)

def fit(df, epochs = 5000): 
    tmp_theta = [ 0 for i in range(2)]
    x = min_max_scaler(df["km"], df["km"])
    for i in range(epochs):
        y_pred = estimate_price(tmp_theta, x) 
        tmp_theta = gradient_descent(y_pred, df, x,  tmp_theta)
    plot_line(df, y_pred)
    return tmp_theta


if __name__ == '__main__':
    df = pd.read_csv('data.csv', delimiter=',')
    plot_data_graph(df)
    columns = ['theta']
    index = [0, 1]
    theta = pd.DataFrame(fit(df, 5000), index, columns)
    theta.to_csv('theta.csv')

