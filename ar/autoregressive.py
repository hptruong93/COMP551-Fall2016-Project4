from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model

model = linear_model.LinearRegression()
# model = linear_model.Ridge(alpha = 1.0, max_iter = None, tol = 0.001)
linear_model.Lasso(alpha=0.1)


if __name__ == "___main__":
    X = [[1,2],[3,4],[5,6]]
    y = [1.5, 3.5, 5.5]

    X_test = [[1,2], [7, 8], [9, 10]]
    y_test = [1.5, 7.5, 9.5]

    model.fit(X, y)
    print "Square mean error: %s" % np.mean((model.predict(X_test) - y_test)**2)

    print model.coef_
    print model.intercept_