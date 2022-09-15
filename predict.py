import pandas as pd
# import min_max_scaler from training_model
# import estimate_price from training_model
from training_model import min_max_scaler
from training_model import estimate_price

def predict():
    # mileage = float(input("Please, enter mileage: "))
    # theta = pd.read_csv('theta.csv', delimiter=',')
    # theta = theta["theta"].tolist()
    # df = pd.read_csv('data.csv', delimiter=',')      
    # x = min_max_scaler(mileage, df["km"])
    # price = estimate_price(theta, x)

    try:
        mileage = float(input("\033[34mPlease, enter mileage: \033[0m"))
        if mileage < 0:
            print("\033[31mError:\033[0m incorrect mileage")
        else: 
            df = pd.read_csv('data.csv', delimiter=',')
            theta = pd.read_csv('theta.csv', delimiter=',')
            theta = theta["theta"].tolist()
            x = min_max_scaler(mileage, df["km"])
            price = estimate_price(theta, x)
            if price > 0:
                print("Price = ", int(price))
            else:
                print("Price is too low")
    except Exception:
        print("\033[31mError:\033[0m incorrect mileage")

if __name__ == '__main__':
    predict()