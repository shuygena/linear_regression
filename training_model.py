import pandas as pd 
import numpy as np 

if __name__ == '__main__':
    df = pd.read_csv('dat.csv', delimiter=',')
    df.head()