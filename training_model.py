import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

lr = 0.01 # learning rate

def plot_data_graph(df):
    plt.scatter(df["km"], df["price"], color='b') 
    #plt.plot(X_test, y_pred,color='k')
    plt.title("Dataset") # заголовок
    plt.xlabel("mileage") # ось абсцисс
    plt.ylabel("price") # ось ординат
    plt.savefig('data.png')
    #plt.show()
    
def plot_line(df, y_pred):
    plt.scatter(df["km"], df["price"], color='b')
    plt.plot(df["km"], y_pred, color="r")
    plt.title("Regression line")
    plt.xlabel("mileage") 
    plt.ylabel("price")
    plt.savefig('result.png')
    #plt.show()
    
def estimate_price(theta, x): # predict (w[0] + w[1] * x)
    return theta[0] + theta[1] * x

def gradient_descent(y_pred, df, x, tmp_theta): 
    tmp_theta[0] -= lr / df.shape[0] * np.sum(y_pred - df["price"]) 
    tmp_theta[1] -= lr / df.shape[0] * np.sum((y_pred - df["price"]) * x)
    return tmp_theta

def min_max_scaler(x, dataset): # предобработка данных
    x_min = dataset.min()
    x_max = dataset.max()
    return (x - x_min) / (x_max - x_min)

def fit(df, epochs = 5000): # обучение
    # df = pd.read_csv('data.csv', delimiter=',')
    # plot_data_graph(df)
    tmp_theta = [ 0 for i in range(2)]
    x = min_max_scaler(df["km"], df["km"])
    #print("x:", x)
    for i in range(epochs):
        y_pred = estimate_price(tmp_theta, x) # 0 - вычислить что подставить
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
#     print("price =", estimate_price(theta, min_max_scaler(50000, df["km"]))) # пробуем предсказать цену
    #print("theta[0] =", theta[0], "theta[1] =", theta[1])
    
    #print(df.head())
    # print(df["km"])
