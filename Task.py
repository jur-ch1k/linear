import numpy as np


class Preprocesser:

    def __init__(self):
        pass

    def fit(self, X, Y=None):
        pass

    def transform(self, X):
        pass

    def fit_transform(self, X, Y=None):
        pass


class MyOneHotEncoder(Preprocesser):

    def __init__(self, dtype=np.float64):
        super(Preprocesser).__init__()
        self.dtype = dtype
        self.arr = []

    def fit(self, X, Y=None):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: unused
        """
        cols = []
        for name in X.columns:
            print(name)
            cols += [X[name].to_list()]

        for arr in cols:
            arr = np.array(arr)
            dict = {}
            unique = np.unique(arr)
            matr = np.eye(len(np.unique(arr)))
            i = 0
            for elem in unique:
                dict[elem] = matr[i]
                i += 1
            self.arr.append(dict)
        print(self.arr)

    def transform(self, X):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        returns: transformed objects, numpy-array, shape [n_objects, |f1| + |f2| + ...]
        """
        res = []
        for i in range(X.shape[0]):  # 6
            obj_arr = []
            for j in range(X.shape[1]):   # 2
                obj_arr += [self.arr[j][X.iloc[i][j]]]
            res += [np.hstack(obj_arr)]
        print(np.array(res))
        return np.array(res)

    def fit_transform(self, X, Y=None):
        self.fit(X)
        return self.transform(X)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


class SimpleCounterEncoder:

    def __init__(self, dtype=np.float64):
        self.dtype = dtype
        self.arr = []

    def fit(self, X, Y):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        """
        cols = []
        ex_flag = False
        try:
            for name in X.columns:
                cols += [X[name].to_list()]
            print('TRY')
        except AttributeError:
            for i in range(X.shape[1]):
                cols += [X[:, i]]
            ex_flag = True
            print('EXCEPTION')

        for col in cols:
            col = np.array(col)
            dict = {}
            print('+-+- ', col)
            unique, cnt = np.unique(col, return_counts=True)
            i = 0
            for elem in unique:
                if ex_flag:
                    mask = col == elem
                    successes = np.sum(Y[mask])/cnt[i]
                else:
                    successes = np.sum(Y[X[col == elem].index])/cnt[i]
                counters = cnt[i]/Y.shape[0]
                dict[elem] = [successes, counters]
                i += 1
            self.arr.append(dict)

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3]
        """
        res = []
        for i in range(X.shape[0]):
            features = []
            for j in range(X.shape[1]):
                successes = self.arr[j][X.iloc[i][j]][0]
                counters = self.arr[j][X.iloc[i][j]][1]
                relation = (successes + a) / (counters + b)
                features.append(np.array([successes, counters, relation]))
            res.append(np.hstack(features))
        return np.array(res)

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


def group_k_fold(size, n_splits=3, seed=1):
    idx = np.arange(size)
    np.random.seed(seed)
    idx = np.random.permutation(idx)
    n_ = size // n_splits
    for i in range(n_splits - 1):
        yield idx[i * n_: (i + 1) * n_], np.hstack((idx[:i * n_], idx[(i + 1) * n_:]))
    yield idx[(n_splits - 1) * n_:], idx[:(n_splits - 1) * n_]


class FoldCounters:

    def __init__(self, n_folds=3, dtype=np.float64):
        self.dtype = dtype
        self.n_folds = n_folds
        self.test = []
        self.arr = []

    def fit(self, X, Y, seed=1):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        param seed: random seed, int
        """
        X = X.to_numpy()
        Y = Y.to_numpy()
        folds = group_k_fold(Y.shape[0], n_splits=self.n_folds, seed=seed)
        for test, train in folds:
            X_train, Y_train = X[train], Y[train]
            model = SimpleCounterEncoder()
            model.fit(X_train, Y_train)
            self.test += [test]
            self.arr += [model.arr]

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3]
        """
        res = np.zeros((len(X.index), len(X.columns) * 3))
        j = 0
        for arr in self.test:
            print(arr)
            X_test = X.iloc[arr]
            model = SimpleCounterEncoder()
            model.arr = self.arr[j]
            j += 1
            X_trans = model.transform(X_test, a, b)
            for i in range(len(X_trans)):
                res[arr[i]] = X_trans[i]
        return np.array(res)

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)


def weights(x, y):
    """
    param x: training set of one feature, numpy-array, shape [n_objects,]
    param y: target for training objects, numpy-array, shape [n_objects,]
    returns: optimal weights, numpy-array, shape [|x unique values|,]
    """
    weight = []
    n = x.shape[0]
    new, k = np.unique(x, return_counts=True)
    for i in range(len(new)):
        cnt = 0
        for j in range(n):
            if new[i] == x[j] and y[j] == 1:
                cnt += 1
        weight += [cnt / k[i]]
    return np.array(weight)


# data = {'col_1': [0, 0, 0, 1, 1, 1], 'col_2': ['a', 'b', 'c', 'a', 'b', 'c'], 'col_3': [0, 1, 2, 3, 4, 5]}
# df_test = pd.DataFrame.from_dict(data)
# enc = SimpleCounterEncoder()
# enc.fit(df_test[['col_1', 'col_2']], df_test['col_3'])
# counts = enc.transform(df_test[['col_1', 'col_2']], a=1, b=1)

# x = np.random.rand(3, 3)
# print(x)
# print('\n')
# print(x[:,0])
# print('\n')
# print(x[:,1])
# print('\n')
# print(x[:,2])
# #print(x[:,3])


# data = {'col_1': [0,1,0,1,0,1,0,1,0,1,0,1], 'col_2':['a','b','c','a','b','c','a','b','c','a','b','c'], 'col_3': [1,2,3,4,1,2,3,4,1,2,3,4]}
# df_test = pd.DataFrame.from_dict(data)
# enc = FoldCounters(n_folds=2)
# enc.fit(df_test[['col_1', 'col_2']], df_test['col_3'], seed=6)
# counts = enc.transform(df_test[['col_1', 'col_2']], a=0, b=0)