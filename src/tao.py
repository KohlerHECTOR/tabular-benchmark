import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from typing import List, Tuple, Optional

class TreeNode:
    def __init__(self, depth: int = 0):
        self.feature_idx: Optional[int] = None
        self.threshold: Optional[float] = None
        self.left: Optional['TreeNode'] = None
        self.right: Optional['TreeNode'] = None
        self.depth: int = depth
        self.prediction: Optional[float] = None
        self.is_leaf: bool = True

class TAOClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        max_depth: int = 3,
        min_samples_split: int = 2,
        max_iter: int = 100,
        tol: float = 1e-4
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_iter = max_iter
        self.tol = tol

    def _initialize_tree(self, X: np.ndarray, y: np.ndarray) -> TreeNode:
        """Initialize the tree structure with random splits."""
        def _grow_tree(depth: int, X: np.ndarray, y: np.ndarray) -> TreeNode:
            node = TreeNode(depth=depth)
            
            if depth >= self.max_depth or len(y) < self.min_samples_split:
                node.prediction = np.mean(y)
                return node

            # Random initialization
            node.is_leaf = False
            node.feature_idx = np.random.randint(X.shape[1])
            node.threshold = np.random.uniform(
                X[:, node.feature_idx].min(),
                X[:, node.feature_idx].max()
            )

            # Split data
            mask = X[:, node.feature_idx] <= node.threshold
            if np.all(mask) or np.all(~mask):
                node.is_leaf = True
                node.prediction = np.mean(y)
                return node

            node.left = _grow_tree(depth + 1, X[mask], y[mask])
            node.right = _grow_tree(depth + 1, X[~mask], y[~mask])
            
            return node

        return _grow_tree(0, X, y)

    def _optimize_node(self, node: TreeNode, X: np.ndarray, y: np.ndarray) -> float:
        """Optimize split threshold and feature for a single node."""
        if node.is_leaf:
            old_pred = node.prediction
            node.prediction = np.mean(y)  # For leaf nodes, just update prediction
            return abs(old_pred - node.prediction)

        best_error = float('inf')
        best_feature = node.feature_idx
        best_threshold = node.threshold
        
        # Try all features
        for feature in range(X.shape[1]):
            sorted_idx = np.argsort(X[:, feature])
            sorted_X = X[sorted_idx]
            sorted_y = y[sorted_idx]
            
            # Try potential thresholds
            for i in range(1, len(X)):
                threshold = (sorted_X[i-1, feature] + sorted_X[i, feature]) / 2
                
                # Split data based on threshold
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                # Skip if split creates empty nodes
                if np.sum(left_mask) < 1 or np.sum(right_mask) < 1:
                    continue
                
                # Calculate predictions for each side
                left_pred = np.mean(y[left_mask])
                right_pred = np.mean(y[right_mask])
                
                # Calculate MSE for this split
                predictions = np.where(left_mask, left_pred, right_pred)
                error = np.mean((y - predictions) ** 2)
                
                # Update best split if this one is better
                if error < best_error:
                    best_error = error
                    best_feature = feature
                    best_threshold = threshold

        change = abs(node.threshold - best_threshold)
        node.feature_idx = best_feature
        node.threshold = best_threshold
        
        # Recursively optimize children
        mask = X[:, node.feature_idx] <= node.threshold
        if np.any(mask) and np.any(~mask):
            self._optimize_node(node.left, X[mask], y[mask])
            self._optimize_node(node.right, X[~mask], y[~mask])
            
        return change

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'TAOClassifier':
        """Fit the TAO classifier."""
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        
        # Convert to binary labels {0, 1}
        y_binary = (y == self.classes_[1]).astype(float)
        
        # Initialize tree
        self.tree_ = self._initialize_tree(X, y_binary)
        
        # Iterative optimization
        for _ in range(self.max_iter):
            max_change = self._optimize_node(self.tree_, X, y_binary)
            if max_change < self.tol:
                break
                
        return self

    def _predict_single(self, node: TreeNode, x: np.ndarray) -> float:
        """Predict for a single sample."""
        if node.is_leaf:
            return node.prediction
            
        if x[node.feature_idx] <= node.threshold:
            return self._predict_single(node.left, x)
        return self._predict_single(node.right, x)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        check_is_fitted(self)
        X = check_array(X)
        
        probas = np.array([self._predict_single(self.tree_, x) for x in X])
        return np.vstack([1 - probas, probas]).T

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]
