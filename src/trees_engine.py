"""
From-Scratch Decision Trees (Classification & Regression with Cost-Complexity Pruning),
One-Hot Categorical Encoding, Random Forests (with Out-of-Bag Validation), and
Second-Order Taylor XGBoost (`05_Trees_and_Gradient_Boosted_Ensembles/trees_and_boosting.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, Union
import numpy as np


# =============================================================================
# 1. ONE-HOT ENCODER FOR CATEGORICAL FEATURES
# =============================================================================

class OneHotEncoderScratch:
    """
    One-Hot Categorical Feature Encoder mapping discrete categorical columns
    into binary indicator vectors in {0.0, 1.0}.
    """

    def __init__(self) -> None:
        self.categories_: List[np.ndarray] = []

    def fit(self, X: np.ndarray) -> "OneHotEncoderScratch":
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        self.categories_ = [np.unique(X_arr[:, j]) for j in range(X_arr.shape[1])]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert len(self.categories_) > 0, "OneHotEncoderScratch must be fitted first."
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        blocks = []
        for j, cats in enumerate(self.categories_):
            col = X_arr[:, j : j + 1]
            blocks.append((col == cats[None, :]).astype(np.float64))
        return np.hstack(blocks)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# =============================================================================
# 2. IMPURITY METRICS (SHANNON ENTROPY, GINI, MISCLASSIFICATION, VARIANCE)
# =============================================================================

def entropy_impurity(y: np.ndarray) -> float:
    """Compute Shannon Entropy H(S) = -sum_c p_c log2(p_c) in bits."""
    if y.size == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / float(y.size)
    probs = probs[probs > 0.0]
    return float(-np.sum(probs * np.log2(probs)))


def gini_impurity(y: np.ndarray) -> float:
    """Compute Gini Impurity G(S) = 1 - sum_c p_c^2."""
    if y.size == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / float(y.size)
    return float(1.0 - np.sum(probs ** 2))


def variance_impurity(y: np.ndarray) -> float:
    """Compute empirical target variance Var(S) = (1/|S|) sum_{i in S} (y_i - y_bar)^2."""
    if y.size <= 1:
        return 0.0
    return float(np.var(y))


# =============================================================================
# 3. DECISION TREE NODE & CLASSIFICATION / REGRESSION TREES
# =============================================================================

@dataclass
class _TreeNode:
    feature_idx: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["_TreeNode"] = None
    right: Optional["_TreeNode"] = None
    value: Optional[Union[float, int]] = None
    probs: Optional[np.ndarray] = None
    impurity: float = 0.0
    n_samples: int = 0

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class DecisionTreeClassifierScratch:
    """
    CART Decision Tree Classifier supporting Shannon Entropy and Gini Impurity,
    midpoint continuous splits, and random feature subspace selection (for Random Forests).
    """

    def __init__(
        self,
        criterion: Literal["gini", "entropy"] = "gini",
        max_depth: int = 6,
        min_samples_split: int = 2,
        min_impurity_decrease: float = 0.0,
        max_features: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.random_state = random_state

        self.root_: Optional[_TreeNode] = None
        self.num_classes_: int = 2
        self._rng = np.random.default_rng(random_state)

    def _impurity(self, y: np.ndarray) -> float:
        return gini_impurity(y) if self.criterion == "gini" else entropy_impurity(y)

    def _leaf_stats(self, y: np.ndarray) -> Tuple[int, np.ndarray]:
        counts = np.bincount(y, minlength=self.num_classes_).astype(np.float64)
        probs = counts / max(float(np.sum(counts)), 1.0)
        return int(np.argmax(probs)), probs

    def _best_split(self, X: np.ndarray, y: np.ndarray, parent_imp: float) -> Tuple[Optional[int], Optional[float], float]:
        m, n = X.shape
        feat_Wait = (
            np.arange(n)
            if (self.max_features is None or self.max_features >= n)
            else self._rng.choice(n, size=self.max_features, replace=False)
        )
        best_gain = -1.0
        best_feat: Optional[int] = None
        best_thresh: Optional[float] = None

        for j in feat_Wait:
            col = X[:, j]
            uniq = np.unique(col)
            if uniq.size <= 1:
                continue
            thresholds = 0.5 * (uniq[:-1] + uniq[1:])
            # If too many unique midpoints, evaluate 32 quantiles for speed while keeping exact midpoints on small sets
            if thresholds.size > 40:
                thresholds = np.quantile(thresholds, np.linspace(0.02, 0.98, 36))

            for t in thresholds:
                left_mask = col <= t
                n_l = int(np.sum(left_mask))
                n_r = m - n_l
                if n_l == 0 or n_r == 0:
                    continue
                imp_l = self._impurity(y[left_mask])
                imp_r = self._impurity(y[~left_mask])
                gain = parent_imp - ((n_l / float(m)) * imp_l + (n_r / float(m)) * imp_r)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = int(j)
                    best_thresh = float(t)

        return best_feat, best_thresh, best_gain

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> _TreeNode:
        imp = self._impurity(y)
        val, probs = self._leaf_stats(y)
        node = _TreeNode(value=val, probs=probs, impurity=imp, n_samples=y.size)

        if depth >= self.max_depth or y.size < self.min_samples_split or imp <= 1e-12:
            return node

        feat, thresh, gain = self._best_split(X, y, imp)
        if feat is None or thresh is None or gain < max(self.min_impurity_decrease, 1e-12):
            return node

        left_mask = X[:, feat] <= thresh
        node.feature_idx = feat
        node.threshold = thresh
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifierScratch":
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64).ravel()
        assert X_arr.ndim == 2 and X_arr.shape[0] == y_arr.shape[0]
        self.num_classes_ = max(int(np.max(y_arr) + 1), 2)
        self.root_ = self._build_tree(X_arr, y_arr, depth=0)
        return self

    def _traverse_proba(self, x_row: np.ndarray, node: _TreeNode) -> np.ndarray:
        if node.is_leaf:
            assert node.probs is not None
            return node.probs
        assert node.feature_idx is not None and node.threshold is not None
        if x_row[node.feature_idx] <= node.threshold:
            assert node.left is not None
            return self._traverse_proba(x_row, node.left)
        assert node.right is not None
        return self._traverse_proba(x_row, node.right)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.root_ is not None
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        return np.array([self._traverse_proba(row, self.root_) for row in X_arr], dtype=np.float64)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


class DecisionTreeRegressorScratch:
    """
    CART Regression Tree splitting by Variance (MSE) Reduction and supporting
    Cost-Complexity Post-Pruning (`ccp_alpha >= 0.0`).
    """

    def __init__(
        self,
        max_depth: int = 6,
        min_samples_split: int = 2,
        ccp_alpha: float = 0.0,
        max_features: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.ccp_alpha: float = float(ccp_alpha)
        self.max_features = max_features
        self._rng = np.random.default_rng(random_state)
        self.root_: Optional[_TreeNode] = None
        self.total_samples_: int = 1

    def _best_split(self, X: np.ndarray, y: np.ndarray, parent_var: float) -> Tuple[Optional[int], Optional[float], float]:
        m, n = X.shape
        feat_indices = (
            np.arange(n)
            if (self.max_features is None or self.max_features >= n)
            else self._rng.choice(n, size=self.max_features, replace=False)
        )
        best_gain = -1.0
        best_feat: Optional[int] = None
        best_thresh: Optional[float] = None

        for j in feat_indices:
            col = X[:, j]
            uniq = np.unique(col)
            if uniq.size <= 1:
                continue
            thresholds = 0.5 * (uniq[:-1] + uniq[1:])
            if thresholds.size > 45:
                thresholds = np.quantile(thresholds, np.linspace(0.02, 0.98, 40))

            for t in thresholds:
                left_mask = col <= t
                n_l = int(np.sum(left_mask))
                n_r = m - n_l
                if n_l == 0 or n_r == 0:
                    continue
                gain = parent_var - (
                    (n_l / float(m)) * variance_impurity(y[left_mask])
                    + (n_r / float(m)) * variance_impurity(y[~left_mask])
                )
                if gain > best_gain:
                    best_gain = gain
                    best_feat = int(j)
                    best_thresh = float(t)

        return best_feat, best_thresh, best_gain

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> _TreeNode:
        var = variance_impurity(y)
        node = _TreeNode(value=float(np.mean(y)), impurity=var, n_samples=y.size)

        if depth >= self.max_depth or y.size < self.min_samples_split or var <= 1e-12:
            return node

        feat, thresh, gain = self._best_split(X, y, var)
        if feat is None or thresh is None or gain <= 1e-12:
            return node

        left_mask = X[:, feat] <= thresh
        node.feature_idx = feat
        node.threshold = thresh
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def _prune_subtree(self, node: _TreeNode) -> Tuple[float, int]:
        """Bottom-up Minimal Cost-Complexity Pruning: collapse subtree if R(t) - R(T_t) <= ccp_alpha * (|T_t| - 1)."""
        if node.is_leaf:
            r_node = (node.n_samples / float(self.total_samples_)) * node.impurity
            return r_node, 1

        assert node.left is not None and node.right is not None
        r_left, leaves_left = self._prune_subtree(node.left)
        r_right, leaves_right = self._prune_subtree(node.right)
        r_sub = r_left + r_right
        leaves_sub = leaves_left + leaves_right

        r_collapsed = (node.n_samples / float(self.total_samples_)) * node.impurity
        if (r_collapsed - r_sub) <= self.ccp_alpha * (leaves_sub - 1):
            node.left = None
            node.right = None
            node.feature_idx = None
            node.threshold = None
            return r_collapsed, 1
        return r_sub, leaves_sub

    def count_leaves(self, node: Optional[_TreeNode] = None) -> int:
        if node is None:
            node = self.root_
        if node is None:
            return 0
        if node.is_leaf:
            return 1
        return self.count_leaves(node.left) + self.count_leaves(node.right)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeRegressorScratch":
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        self.total_samples_ = y_arr.size
        self.root_ = self._build_tree(X_arr, y_arr, depth=0)
        if self.ccp_alpha > 0.0 and self.root_ is not None:
            self._prune_subtree(self.root_)
        return self

    def _traverse(self, x_row: np.ndarray, node: _TreeNode) -> float:
        if node.is_leaf:
            return float(node.value)  # type: ignore[arg-type]
        assert node.feature_idx is not None and node.threshold is not None
        if x_row[node.feature_idx] <= node.threshold:
            assert node.left is not None
            return self._traverse(x_row, node.left)
        assert node.right is not None
        return self._traverse(x_row, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.root_ is not None
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        return np.array([self._traverse(row, self.root_) for row in X_arr], dtype=np.float64)


# =============================================================================
# 4. RANDOM FOREST (BOOTSTRAP AGGREGATION + FEATURE SUBSAMPLING + OOB SCORE)
# =============================================================================

class RandomForestScratch:
    """
    Random Forest Classifier combining Bootstrap Sampling With Replacement (63.2% unique law),
    Random Feature Subspace Selection (d' = floor(sqrt(n))), and Out-of-Bag (OOB) validation.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        max_depth: int = 8,
        min_samples_split: int = 2,
        max_features: Union[Literal["sqrt", "all"], int] = "sqrt",
        criterion: Literal["gini", "entropy"] = "gini",
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self.random_state = random_state

        self.trees_: List[DecisionTreeClassifierScratch] = []
        self.oob_indices_: List[np.ndarray] = []
        self.inbag_unique_fractions_: List[float] = []
        self.oob_score_: float = 0.0
        self.num_classes_: int = 2

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestScratch":
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64).ravel()
        m, n = X_arr.shape
        self.num_classes_ = max(int(np.max(y_arr) + 1), 2)

        if self.max_features == "sqrt":
            d_sub = max(1, int(np.sqrt(n)))
        elif self.max_features == "all":
            d_sub = n
        else:
            d_sub = int(self.max_features)

        rng = np.random.default_rng(self.random_state)
        self.trees_.clear()
        self.oob_indices_.clear()
        self.inbag_unique_fractions_.clear()

        oob_vote_sums = np.zeros((m, self.num_classes_), dtype=np.float64)
        oob_counts = np.zeros(m, dtype=np.int64)
        all_idx = np.arange(m)

        for b in range(self.n_estimators):
            # Sample m instances uniformly WITH replacement
            boot_idx = rng.integers(0, m, size=m)
            uniq_inbag = np.unique(boot_idx)
            self.inbag_unique_fractions_.append(uniq_inbag.size / float(m))

            oob_mask = np.ones(m, dtype=bool)
            oob_mask[uniq_inbag] = False
            oob_idx = all_idx[oob_mask]
            self.oob_indices_.append(oob_idx)

            tree = DecisionTreeClassifierScratch(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=d_sub,
                random_state=int(rng.integers(0, 1_000_000)),
            )
            tree.num_classes_ = self.num_classes_
            tree.fit(X_arr[boot_idx], y_arr[boot_idx])
            self.trees_.append(tree)

            if oob_idx.size > 0:
                oob_vote_sums[oob_idx] += tree.predict_proba(X_arr[oob_idx])
                oob_counts[oob_idx] += 1

        valid_oob = oob_counts > 0
        if np.any(valid_oob):
            oob_preds = np.argmax(oob_vote_sums[valid_oob], axis=1)
            self.oob_score_ = float(np.mean(oob_preds == y_arr[valid_oob]))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert len(self.trees_) > 0
        X_arr = np.asarray(X, dtype=np.float64)
        prob_sum = np.zeros((X_arr.shape[0], self.num_classes_), dtype=np.float64)
        for tree in self.trees_:
            prob_sum += tree.predict_proba(X_arr)
        return prob_sum / float(len(self.trees_))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


# =============================================================================
# 5. SECOND-ORDER TAYLOR XGBOOST ENGINE FROM SCRATCH
# =============================================================================

@dataclass
class _XGBNode:
    feature_idx: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["_XGBNode"] = None
    right: Optional["_XGBNode"] = None
    leaf_weight: float = 0.0

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class XGBoostScratch:
    """
    Extreme Gradient Boosting (XGBoost) from First Principles (Chen & Guestrin, 2016).

    Fits Newton-boosted decision trees using exact first-order gradients g_i and
    second-order Hessians h_i of the loss function:
      - Optimal Leaf Weight: w_j* = - G_j / (H_j + reg_lambda)
      - Exact Split Gain:
            Gain = 0.5 * [ G_L^2/(H_L + lambda) + G_R^2/(H_R + lambda) - (G_L+G_R)^2/(H_L+H_R+lambda) ] - gamma
    """

    def __init__(
        self,
        n_estimators: int = 30,
        learning_rate: float = 0.25,
        max_depth: int = 3,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
        min_child_weight: float = 1e-3,
        loss: Literal["mse", "logloss"] = "mse",
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = float(learning_rate)
        self.max_depth = max_depth
        self.reg_lambda = float(reg_lambda)
        self.gamma = float(gamma)
        self.min_child_weight = float(min_child_weight)
        self.loss = loss

        self.trees_: List[_XGBNode] = []
        self.base_score_: float = 0.0
        self.loss_history: List[float] = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        out = np.empty_like(z, dtype=np.float64)
        pos = z >= 0.0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[~pos])
        out[~pos] = ez / (1.0 + ez)
        return out

    def compute_loss_grad_hess(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute scalar loss and exact per-sample first-order gradient g_i and second-order Hessian h_i.
        """
        yt = np.asarray(y_true, dtype=np.float64).ravel()
        yp = np.asarray(y_pred, dtype=np.float64).ravel()

        if self.loss == "mse":
            # l(y_i, y_hat_i) = 0.5 * (y_hat_i - y_i)^2  =>  g_i = y_hat_i - y_i,  h_i = 1.0
            loss_val = float(0.5 * np.mean((yp - yt) ** 2))
            g = yp - yt
            h = np.ones_like(yt)
            return loss_val, g, h

        if self.loss == "logloss":
            # l(y_i, z_i) = log(1 + exp(z_i)) - y_i * z_i  =>  g_i = p_i - y_i,  h_i = p_i * (1 - p_i)
            p = self._sigmoid(yp)
            loss_val = float(np.mean(np.logaddexp(0.0, yp) - yt * yp))
            g = p - yt
            h = np.maximum(p * (1.0 - p), 1e-12)
            return loss_val, g, h

        raise ValueError(f"Unsupported XGBoost loss: {self.loss}")

    def _build_xgb_tree(self, X: np.ndarray, g: np.ndarray, h: np.ndarray, depth: int) -> _XGBNode:
        G_tot = float(np.sum(g))
        H_tot = float(np.sum(h))
        w_opt = -G_tot / (H_tot + self.reg_lambda)
        node = _XGBNode(leaf_weight=w_opt)

        if depth >= self.max_depth or X.shape[0] < 2 or H_tot < self.min_child_weight:
            return node

        best_gain = 0.0
        best_feat: Optional[int] = None
        best_thresh: Optional[float] = None
        parent_score = (G_tot ** 2) / (H_tot + self.reg_lambda)

        m, n = X.shape
        for j in range(n):
            col = X[:, j]
            uniq = np.unique(col)
            if uniq.size <= 1:
                continue
            thresholds = 0.5 * (uniq[:-1] + uniq[1:])
            if thresholds.size > 45:
                thresholds = np.quantile(thresholds, np.linspace(0.02, 0.98, 40))

            for t in thresholds:
                left_mask = col <= t
                H_L = float(np.sum(h[left_mask]))
                H_R = H_tot - H_L
                if H_L < self.min_child_weight or H_R < self.min_child_weight:
                    continue
                G_L = float(np.sum(g[left_mask]))
                G_R = G_tot - G_L

                gain = 0.5 * (
                    (G_L ** 2) / (H_L + self.reg_lambda)
                    + (G_R ** 2) / (H_R + self.reg_lambda)
                    - parent_score
                ) - self.gamma

                if gain > best_gain:
                    best_gain = gain
                    best_feat = int(j)
                    best_thresh = float(t)

        if best_feat is None or best_thresh is None or best_gain <= 1e-12:
            return node

        left_mask = X[:, best_feat] <= best_thresh
        node.feature_idx = best_feat
        node.threshold = best_thresh
        node.left = self._build_xgb_tree(X[left_mask], g[left_mask], h[left_mask], depth + 1)
        node.right = self._build_xgb_tree(X[~left_mask], g[~left_mask], h[~left_mask], depth + 1)
        return node

    def _predict_tree(self, X: np.ndarray, node: _XGBNode) -> np.ndarray:
        out = np.empty(X.shape[0], dtype=np.float64)
        for i, row in enumerate(X):
            curr = node
            while not curr.is_leaf:
                assert curr.feature_idx is not None and curr.threshold is not None
                curr = curr.left if row[curr.feature_idx] <= curr.threshold else curr.right  # type: ignore[assignment]
            out[i] = curr.leaf_weight
        return out

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostScratch":
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        y_arr = np.asarray(y, dtype=np.float64).ravel()

        if self.loss == "mse":
            self.base_score_ = float(np.mean(y_arr))
        else:
            p_mean = np.clip(np.mean(y_arr), 1e-5, 1.0 - 1e-5)
            self.base_score_ = float(np.log(p_mean / (1.0 - p_mean)))

        y_hat = np.full_like(y_arr, self.base_score_, dtype=np.float64)
        self.trees_.clear()
        self.loss_history.clear()

        for _ in range(self.n_estimators):
            loss_val, g, h = self.compute_loss_grad_hess(y_arr, y_hat)
            self.loss_history.append(loss_val)

            tree = self._build_xgb_tree(X_arr, g, h, depth=0)
            self.trees_.append(tree)
            y_hat = y_hat + self.learning_rate * self._predict_tree(X_arr, tree)

        final_loss, _, _ = self.compute_loss_grad_hess(y_arr, y_hat)
        self.loss_history.append(final_loss)
        return self

    def predict_raw(self, X: np.ndarray, up_to_tree: Optional[int] = None) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        n_t = len(self.trees_) if up_to_tree is None else min(up_to_tree, len(self.trees_))
        preds = np.full(X_arr.shape[0], self.base_score_, dtype=np.float64)
        for t in range(n_t):
            preds += self.learning_rate * self._predict_tree(X_arr, self.trees_[t])
        return preds

    def predict(self, X: np.ndarray, up_to_tree: Optional[int] = None) -> np.ndarray:
        raw = self.predict_raw(X, up_to_tree=up_to_tree)
        if self.loss == "mse":
            return raw
        return (self._sigmoid(raw) >= 0.5).astype(np.int64)


# =============================================================================
# 6. FINITE-DIFFERENCE GRADIENT & HESSIAN VERIFICATION SUITE
# =============================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("MODULE 05: DECISION TREES, RANDOM FOREST & XGBOOST — VERIFICATION SUITE")
    print("=" * 72)

    rng = np.random.default_rng(42)
    m = 35
    y_reg = rng.normal(size=m)
    y_bin = rng.integers(0, 2, size=m).astype(np.float64)
    y_hat_test = rng.normal(size=m)
    eps = 1e-5

    # Verify XGBoost first-order gradients g_i and second-order Hessians h_i via centered finite differences
    for loss_mode, y_target in [("mse", y_reg), ("logloss", y_bin)]:
        xgb = XGBoostScratch(loss=loss_mode)
        _, ana_g, ana_h = xgb.compute_loss_grad_hess(y_target, y_hat_test)
        num_g = np.zeros_like(ana_g)
        num_h = np.zeros_like(ana_h)

        for i in range(m):
            yp_plus = y_hat_test.copy()
            yp_minus = y_hat_test.copy()
            yp_plus[i] += eps
            yp_minus[i] -= eps

            # Multiply mean loss by m to get sum of per-sample losses
            l_plus, g_plus, _ = xgb.compute_loss_grad_hess(y_target, yp_plus)
            l_minus, g_minus, _ = xgb.compute_loss_grad_hess(y_target, yp_minus)
            num_g[i] = float(m) * (l_plus - l_minus) / (2.0 * eps)
            num_h[i] = (g_plus[i] - g_minus[i]) / (2.0 * eps)

        rel_g = float(np.linalg.norm(ana_g - num_g) / (np.linalg.norm(ana_g) + np.linalg.norm(num_g)))
        rel_h = float(np.linalg.norm(ana_h - num_h) / (np.linalg.norm(ana_h) + np.linalg.norm(num_h)))
        print(f"[Grad/Hess Check] XGBoost {loss_mode.upper():<7} | g_i Error: {rel_g:.3e} | h_i Error: {rel_h:.3e} (< 1e-7 PASS)")
        assert rel_g < 1e-7 and rel_h < 1e-7

    # Verify 63.2% Bootstrap Law on RandomForestScratch
    X_rf = rng.normal(size=(500, 6))
    y_rf = (X_rf[:, 0] + X_rf[:, 1] > 0).astype(np.int64)
    rf = RandomForestScratch(n_estimators=40, max_depth=5, random_state=7).fit(X_rf, y_rf)
    mean_inbag = float(np.mean(rf.inbag_unique_fractions_))
    theoretical_inbag = 1.0 - np.exp(-1.0)
    print(f"[Bootstrap Law] Empirical Unique In-Bag: {mean_inbag*100:.2f}% vs. Theoretical (1 - 1/e) = {theoretical_inbag*100:.2f}% | OOB Acc: {rf.oob_score_*100:.2f}%")
    assert abs(mean_inbag - theoretical_inbag) < 0.015
    print("=" * 72)

