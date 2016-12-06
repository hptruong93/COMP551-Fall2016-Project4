import numpy as np
from sklearn import linear_model

# model = linear_model.LinearRegression()
# model = linear_model.Ridge(alpha = 1.0, max_iter = None, tol = 0.001)
model = linear_model.Lasso(alpha=0.1)

if __name__ == "__main__":
    X = [[1,2],[3,4],[5,6]]
    y = [1.5, 3.5, 5.5]

    X_test = [[1,2], [7, 8], [9, 10]]
    y_test = [1.5, 7.5, 9.5]

    print "Here"

    model.fit(X, y)
    print "Square mean error: %s" % np.mean((model.predict(X_test) - y_test)**2)

    print model.coef_
    print model.intercept_