import os
import numpy as np
import pandas as pd


def get_synth_data(x_min=-4, x_max=4, n=1000, train=True):
    x = np.linspace(x_min, x_max, n)
    x = np.expand_dims(x, -1).astype(np.float32)
    
    sigma = 3 * abs(x)+0.2 if train else np.zeros_like(x)

    #sigma = 3 * np.ones_like(x) if train else np.zeros_like(x)    
    y = x**3 + np.random.normal(0, sigma).astype(np.float32)
    
    # now just for testing
    #y = x*3 + np.random.normal(0, sigma).astype(np.float32)
    #return x, y
    #x = np.linspace(x_min, x_max, n)
    #x = np.expand_dims(x, -1).astype(np.float32)

    #sigma = 0.15 * abs(x) if train else np.zeros_like(x)
    sigma = 3.0* abs(x)+0.2 if train else np.zeros_like(x)
    y = x**3 - np.random.exponential(sigma)
    
    return x, y

def standardize(data):
    mu = data.mean(axis=0, keepdims=1)
    scale = data.std(axis=0, keepdims=1)
    scale[scale<1e-10] = 1.0

    data = (data - mu) / scale
    return data, mu, scale

vb_dir   = os.path.dirname(__file__)
data_dir = os.path.join(vb_dir, "data")

def load_boston(seed=123):
    """
    Attribute Information:
    1. CRIM: per capita crime rate by town
    2. ZN: proportion of residential land zoned for lots over 25,000 sq.ft.
    3. INDUS: proportion of non-retail business acres per town
    4. CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
    5. NOX: nitric oxides concentration (parts per 10 million)
    6. RM: average number of rooms per dwelling
    7. AGE: proportion of owner-occupied units built prior to 1940
    8. DIS: weighted distances to five Boston employment centres
    9. RAD: index of accessibility to radial highways
    10. TAX: full-value property-tax rate per $10,000
    11. PTRATIO: pupil-teacher ratio by town
    12. B: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    13. LSTAT: % lower status of the population
    14. MEDV: Median value of owner-occupied homes in $1000's
    """
    data = np.loadtxt(os.path.join(data_dir,
                                   'boston-housing/boston_housing.txt'))
    X    = data[:, :-1]
    y    = data[:,  -1]


    rs = np.random.RandomState(123+seed)
    permutation = rs.permutation(X.shape[0])
    test_fraction = 0.2
    size_train  = int(np.round(X.shape[ 0 ] * (1 - test_fraction)))
    index_train = permutation[ 0 : size_train ]
    index_test  = permutation[ size_train : ]

    X_train = X[index_train, : ]
    X_test  = X[index_test, : ]

    y_train = y[index_train, None]
    y_test = y[index_test, None]

    X_train, x_train_mu, x_train_scale = standardize(X_train)
    X_test = (X_test - x_train_mu) / x_train_scale

    #y_train, y_train_mu, y_train_scale = standardize(y_train)
    #y_test = (y_test - y_train_mu) / y_train_scale
    return X_train, y_train, X_test, y_test

def load_concrete(seed=123):
    """
    Summary Statistics:
    Number of instances (observations): 1030
    Number of Attributes: 9
    Attribute breakdown: 8 quantitative input variables, and 1 quantitative output variable
    Missing Attribute Values: None
    Name -- Data Type -- Measurement -- Description
    Cement (component 1) -- quantitative -- kg in a m3 mixture -- Input Variable
    Blast Furnace Slag (component 2) -- quantitative -- kg in a m3 mixture -- Input Variable
    Fly Ash (component 3) -- quantitative -- kg in a m3 mixture -- Input Variable
    Water (component 4) -- quantitative -- kg in a m3 mixture -- Input Variable
    Superplasticizer (component 5) -- quantitative -- kg in a m3 mixture -- Input Variable
    Coarse Aggregate (component 6) -- quantitative -- kg in a m3 mixture -- Input Variable
    Fine Aggregate (component 7) -- quantitative -- kg in a m3 mixture -- Input Variable
    Age -- quantitative -- Day (1~365) -- Input Variable
    Concrete compressive strength -- quantitative -- MPa -- Output Variable
    ---------------------------------
    """
    data_file = os.path.join(data_dir, 'concrete/Concrete_Data.xls')
    data = pd.read_excel(data_file)
    X    = data.values[:, :-1]
    y    = data.values[:,  -1]

    rs = np.random.RandomState(123+seed)
    permutation = rs.permutation(X.shape[0])
    test_fraction = 0.1
    size_train  = int(np.round(X.shape[ 0 ] * (1 - test_fraction)))
    index_train = permutation[ 0 : size_train ]
    index_test  = permutation[ size_train : ]

    X_train = X[index_train, : ]
    X_test  = X[index_test, : ]

    y_train = y[index_train, None]
    y_test = y[index_test, None]

    X_train, x_train_mu, x_train_scale = standardize(X_train)
    X_test = (X_test - x_train_mu) / x_train_scale

    #y_train, y_train_mu, y_train_scale = standardize(y_train)
    #y_test = (y_test - y_train_mu) / y_train_scale
    return X_train, y_train, X_test, y_test