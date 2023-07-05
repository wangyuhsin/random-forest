import numpy as np
from sklearn.utils import resample

from dtree import *


class RandomForestRegressor621:
    def __init__(
        self, n_estimators=10, min_samples_leaf=3, max_features=0.3, oob_score=False
    ):
        # super().__init__(n_estimators, oob_score=oob_score)
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.oob_score = oob_score
        self.oob_score_ = np.nan
        # each tree is represented by a tuple of (tree object, OOB index)
        self.trees = []

    def fit(self, X, y):
        """
        Given an (X, y) training set, fit all n_estimators trees to different,
        bootstrapped versions of the training data.  Keep track of the indices of
        the OOB records for each tree.  After fitting all of the trees in the forest,
        compute the OOB validation score estimate and store as self.oob_score_, to
        mimic sklearn.
        """
        for _ in range(self.n_estimators):
            bootstrapped_index = resample(range(len(X)), replace=True)
            oob_index = set(range(len(X))).difference(set(bootstrapped_index))

            X_bts, y_bts = X[bootstrapped_index].copy(
            ), y[bootstrapped_index].copy()
            tree = RegressionTree621(self.max_features, self.min_samples_leaf)
            tree.fit(X_bts, y_bts)

            self.trees.append((tree, oob_index))

        if self.oob_score:
            preds = []
            y_true = []
            for i, obs in enumerate(zip(X, y)):
                x_obs, y_obs = obs
                leaves = [
                    tree.root.leaf(x_obs)
                    for tree, oob_index in self.trees
                    if i in oob_index
                ]
                if leaves:
                    y_pred = np.sum(
                        [leaf.n * leaf.prediction for leaf in leaves]
                    ) / np.sum([leaf.n for leaf in leaves])
                    preds.append(y_pred)
                    y_true.append(y_obs)

            self.oob_score_ = r2_score(y_true, preds)

    def predict(self, X_test) -> np.ndarray:
        """
        Given a 2D nxp array with one or more records, compute the weighted average
        prediction from all trees in this forest. Weight each trees prediction by
        the number of observations in the leaf making that prediction.  Return a 1D vector
        with the predictions for each input record of X_test.
        """
        preds = np.zeros(X_test.shape[0])
        for i, x_test in enumerate(X_test):
            leaves = [tree.root.leaf(x_test) for tree, _ in self.trees]
            y_pred = np.sum([leaf.n * leaf.prediction for leaf in leaves]) / np.sum(
                [leaf.n for leaf in leaves]
            )
            preds[i] = y_pred

        return preds

    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the prediction for each record and then compute R^2 on that and y_test.
        """
        return r2_score(y_test, self.predict(X_test))


class RandomForestClassifier621:
    def __init__(
        self, n_estimators=10, min_samples_leaf=3, max_features=0.3, oob_score=False
    ):
        # super().__init__(n_estimators, oob_score=oob_score)
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.oob_score = oob_score
        self.oob_score_ = np.nan
        # each tree is represented by a tuple of (tree object, OOB index)
        self.trees = []
        self.n_classes = 0

    def fit(self, X, y):
        """
        Given an (X, y) training set, fit all n_estimators trees to different,
        bootstrapped versions of the training data.  Keep track of the indices of
        the OOB records for each tree.  After fitting all of the trees in the forest,
        compute the OOB validation score estimate and store as self.oob_score_, to
        mimic sklearn.
        """
        self.n_classes = len(np.unique(y))
        for _ in range(self.n_estimators):
            bootstrapped_index = resample(range(len(X)), replace=True)
            oob_index = set(range(len(X))).difference(set(bootstrapped_index))

            X_bts, y_bts = X[bootstrapped_index].copy(
            ), y[bootstrapped_index].copy()
            tree = ClassifierTree621(self.max_features, self.min_samples_leaf)
            tree.fit(X_bts, y_bts)

            self.trees.append((tree, oob_index))

        if self.oob_score:
            preds = []
            y_true = []
            for i, obs in enumerate(zip(X, y)):
                x_obs, y_obs = obs
                leaves = [
                    tree.root.leaf(x_obs)
                    for tree, oob_index in self.trees
                    if i in oob_index
                ]
                if leaves:
                    class_count = np.zeros(self.n_classes, dtype=np.int64)
                    for leaf in leaves:
                        class_, counts = np.unique(leaf.y, return_counts=True)
                        for k, v in zip(class_, counts):
                            class_count[k] += v

                    y_pred = np.argmax(class_count)
                    preds.append(y_pred)
                    y_true.append(y_obs)

            self.oob_score_ = accuracy_score(y_true, preds)

    def predict(self, X_test) -> np.ndarray:
        """
        Given a 2D nxp array with one or more records, compute the weighted average
        prediction from all trees in this forest. Weight each trees prediction by
        the number of observations in the leaf making that prediction.  Return a 1D vector
        with the predictions for each input record of X_test.
        """
        preds = np.zeros(X_test.shape[0])
        for i, x_test in enumerate(X_test):
            leaves = [tree.root.leaf(x_test) for tree, _ in self.trees]
            class_count = np.zeros(self.n_classes, dtype=np.int64)
            for leaf in leaves:
                class_, counts = np.unique(leaf.y, return_counts=True)
                for k, v in zip(class_, counts):
                    class_count[k] += v

            y_pred = np.argmax(class_count)
            preds[i] = y_pred

        return preds

    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the prediction for each record and then compute R^2 on that and y_test.
        """
        return accuracy_score(y_test, self.predict(X_test))
