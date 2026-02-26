from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from pyotc.otc_backend.graph.utils import get_01_cost, get_sq_cost

ArrayLike = Union[np.ndarray, List[List[Any]]]


# ============================================================
# Utilities
# ============================================================

def _as_numpy_2d(a: ArrayLike, name: str) -> np.ndarray:
    """
    Convert input into a 2D NumPy array.

    This is used to enforce that:
      - transition matrices P are 2D arrays, and
      - node feature matrices X are 2D arrays,
    before any other validation happens.

    Args:
        a (ArrayLike): NumPy array or nested list.
        name (str): Name used in error messages.

    Returns:
        np.ndarray: 2D NumPy array.

    Raises:
        ValueError: If the input is not 2D.
    """
    arr = np.asarray(a)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, but got shape {arr.shape} (ndim={arr.ndim}).")
    return arr


def _is_square(a: np.ndarray) -> bool:
    """
    Check whether an array is a square matrix.

    Args:
        a (np.ndarray): Candidate array.

    Returns:
        bool: True iff a has shape (n, n).
    """
    return a.ndim == 2 and a.shape[0] == a.shape[1]


def _is_row_stochastic(P: np.ndarray, atol: float = 1e-8) -> bool:
    """
    Check whether a matrix is a valid row-stochastic transition matrix.

    A valid transition matrix must satisfy:
      1) entries are nonnegative (up to tolerance),
      2) each row sums to 1 (up to tolerance).

    Args:
        P (np.ndarray): Candidate transition matrix of shape (n, n).
        atol (float): Absolute tolerance for numerical comparisons.

    Returns:
        bool: True if row-stochastic; False otherwise.
    """
    if np.any(P < -atol):
        return False
    return np.allclose(P.sum(axis=1), 1.0, atol=atol)


def _is_irreducible_transition_matrix(P: np.ndarray, atol: float = 1e-12) -> bool:
    """
    Check irreducibility of a Markov chain transition matrix.

    We build the directed support graph where an edge i -> j exists iff P[i, j] > atol.
    The chain is irreducible iff this support graph is strongly connected.

    Implementation: reachability from node 0 in both the graph and its transpose.

    Args:
        P (np.ndarray): Transition matrix of shape (n, n).
        atol (float): Threshold to decide whether an entry counts as a positive edge.

    Returns:
        bool: True if irreducible; False otherwise.

    Notes:
        - This does NOT check aperiodicity.
        - If any row has no outgoing edges under the threshold, we treat it as invalid.
    """
    n = P.shape[0]
    if n == 0:
        return False

    A = (P > atol)
    for i in range(n):
        if not A[i].any():
            return False

    def _reachable_from(start: int, adj: np.ndarray) -> np.ndarray:
        seen = np.zeros(n, dtype=bool)
        stack = [start]
        seen[start] = True
        while stack:
            u = stack.pop()
            for v in np.flatnonzero(adj[u]):
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        return seen

    if not _reachable_from(0, A).all():
        return False
    if not _reachable_from(0, A.T).all():
        return False
    return True


def _infer_feature_column_kinds(X: np.ndarray) -> List[str]:
    """
    Infer per-feature (per-column) type "kind" from a node feature matrix X.

    Per dimension (column), we classify entries as:
      - "numeric": numeric / bool types
      - "string":  str / bytes types
      - "mixed":   mixture of numeric and string (invalid)
      - "other":   unsupported types (invalid)

    Missing entries (None, NaN in object arrays) are ignored for inference. If an
    entire column is missing, we default it to "numeric" (you can change later).

    Args:
        X (np.ndarray): Node feature matrix of shape (n, p).

    Returns:
        List[str]: List of length p; kind per feature dimension.
    """
    kinds: List[str] = []
    _, p = X.shape

    for j in range(p):
        col = X[:, j]

        if np.issubdtype(col.dtype, np.number) or np.issubdtype(col.dtype, np.bool_):
            kinds.append("numeric")
            continue

        is_num = False
        is_str = False
        is_other = False

        for v in col:
            if v is None:
                continue
            try:
                if isinstance(v, float) and np.isnan(v):
                    continue
            except Exception:
                pass

            if isinstance(v, (str, bytes, np.str_, np.bytes_)):
                is_str = True
            elif isinstance(v, (int, float, bool, np.integer, np.floating, np.bool_)):
                is_num = True
            else:
                is_other = True

            if is_other or (is_num and is_str):
                break

        if is_other:
            kinds.append("other")
        elif is_num and is_str:
            kinds.append("mixed")
        elif is_str:
            kinds.append("string")
        elif is_num:
            kinds.append("numeric")
        else:
            kinds.append("numeric")  # all missing -> default numeric

    return kinds


def _check_feature_kinds_consistent(
    X_list: Sequence[np.ndarray],
    p: int,
    *,
    where: str,
) -> List[str]:
    """
    Enforce that feature types are consistent across a list of node feature matrices.

    Rules:
      - For each feature dimension j, all graphs must agree on kind ("numeric" or "string").
      - Any "mixed" or "other" is rejected.
      - Returns the inferred baseline kinds from the first matrix.

    Args:
        X_list (Sequence[np.ndarray]): List of node feature matrices.
        p (int): Expected number of feature dimensions.
        where (str): Split label used in error messages.

    Returns:
        List[str]: Baseline kinds (length p), inferred from X_list[0].

    Raises:
        TypeError: If a column is mixed/other, or kinds disagree across graphs.
    """
    if len(X_list) == 0:
        return ["numeric"] * p

    base = _infer_feature_column_kinds(X_list[0])
    for j, k0 in enumerate(base):
        if k0 in ("mixed", "other"):
            raise TypeError(
                f"[{where}] invalid feature type in first X at column {j}: kind='{k0}'. "
                f"Expected all-numeric or all-string per column."
            )

    for i, X in enumerate(X_list[1:], start=1):
        kinds = _infer_feature_column_kinds(X)
        for j, (k0, kj) in enumerate(zip(base, kinds)):
            if kj in ("mixed", "other"):
                raise TypeError(
                    f"[{where}] invalid feature type in X[{i}] at column {j}: kind='{kj}'. "
                    f"Expected all-numeric or all-string per column."
                )
            if kj != k0:
                raise TypeError(
                    f"[{where}] inconsistent feature types at column {j}: "
                    f"X[0] is '{k0}' but X[{i}] is '{kj}'."
                )

    return base


def _check_train_test_feature_kinds_match(train_kinds: List[str], test_kinds: List[str]) -> None:
    """
    Enforce that train and test splits agree on feature kinds per dimension.

    Args:
        train_kinds (List[str]): Kinds inferred for train features.
        test_kinds (List[str]): Kinds inferred for test features.

    Raises:
        ValueError: If lengths mismatch (should not happen if p is the same).
        TypeError: If any dimension differs (numeric vs string) between splits.
    """
    if len(train_kinds) != len(test_kinds):
        raise ValueError(
            f"Train/test feature kinds length mismatch: {len(train_kinds)} vs {len(test_kinds)}."
        )
    mism = [(j, tk, vk) for j, (tk, vk) in enumerate(zip(train_kinds, test_kinds)) if tk != vk]
    if mism:
        details = ", ".join([f"col {j}: train={tk}, test={vk}" for j, tk, vk in mism[:10]])
        more = "" if len(mism) <= 10 else f" (and {len(mism) - 10} more)"
        raise TypeError(
            "Train/test feature types are inconsistent per dimension. "
            f"Mismatches: {details}{more}."
        )


def _column_as_list(X: np.ndarray, j: int, kind: str) -> List[Any]:
    """
    Extract X[:, j] as a Python list with values cast to a supported type.

    This is used to feed get_01_cost / get_sq_cost which operate over Python lists.

    Args:
        X (np.ndarray): Node feature matrix (n, p).
        j (int): Feature dimension (column index).
        kind (str): "numeric" or "string".

    Returns:
        List[Any]: List of floats (numeric) or strs (string).

    Raises:
        ValueError: If kind is not recognized.
    """
    col = X[:, j]
    if kind == "numeric":
        return [float(v) for v in col]
    if kind == "string":
        return [str(v) for v in col]
    raise ValueError(f"Unknown kind {kind!r}.")


def _len_or_none(x: Optional[Sequence[Any]]) -> Optional[int]:
    """Utility: return len(x) or None if x is None."""
    return None if x is None else len(x)


def _validate_parallel_lists(
    *,
    where: str,
    P_list: Sequence[Any],
    X_list: Sequence[Any],
    y_list: Optional[Sequence[Any]],
    y_required: bool,
) -> None:
    """
    Validate that parallel lists (P, X, y) are length-aligned.

    Rules:
      - len(P_list) must equal len(X_list)
      - if y_required=True, then y_list must be provided and len(y_list) must match
      - if y_required=False and y_list is provided, its length must still match

    Since alignment cannot be verified programmatically, we print a strong warning
    reminding the user to keep ordering consistent.

    Args:
        where (str): Split label ("train" or "test").
        P_list (Sequence[Any]): List of transition matrices.
        X_list (Sequence[Any]): List of node feature matrices.
        y_list (Optional[Sequence[Any]]): List of labels or None.
        y_required (bool): Whether labels are required for the split.

    Raises:
        ValueError: If lengths mismatch or required y_list is missing.
    """
    if len(P_list) != len(X_list):
        raise ValueError(
            f"[{where}] length mismatch: len(P_{where})={len(P_list)} vs len(X_{where})={len(X_list)}."
        )

    if y_required:
        if y_list is None:
            raise ValueError(f"[{where}] y is required (supervised) but y_{where} is None.")
        if len(y_list) != len(P_list):
            raise ValueError(
                f"[{where}] length mismatch: len(y_{where})={len(y_list)} vs len(P_{where})={len(P_list)}."
            )
    else:
        if y_list is not None and len(y_list) != len(P_list):
            raise ValueError(
                f"[{where}] length mismatch: len(y_{where})={len(y_list)} vs len(P_{where})={len(P_list)}."
            )

    print(
        f"[Warning/{where}] Please make sure P, X"
        + (", y" if y_list is not None or y_required else "")
        + f" are sorted/aligned in the SAME order for the {where} split."
    )


# ============================================================
# Data Structures
# ============================================================

@dataclass(frozen=True)
class AttributedNetwork:
    """
    A single attributed network sample.

    Attributes:
        P (np.ndarray): Transition matrix of shape (n, n).
        X (np.ndarray): Node feature matrix of shape (n, p).
        y (Any): Optional network-level label. Required for train if supervised=True.
    """
    P: np.ndarray
    X: np.ndarray
    y: Any = None

    @property
    def n(self) -> int:
        """Number of nodes (states) in this network."""
        return int(self.P.shape[0])


class AttributedNetworkDataset:
    """
    Lightweight dataset wrapper around a list of AttributedNetwork samples.

    Supports:
      - __len__  : number of samples
      - __getitem__: index access
      - __iter__ : iteration over samples

    Optionally enforces labels are present for all samples if require_y=True.
    """

    def __init__(
        self,
        samples: Sequence[AttributedNetwork],
        *,
        name: str,
        require_y: bool = False,
    ) -> None:
        self.name = name
        self.require_y = require_y
        self._samples = list(samples)

        if len(self._samples) == 0:
            raise ValueError(f"[{self.name}] dataset is empty.")

        if self.require_y:
            missing = [i for i, s in enumerate(self._samples) if s.y is None]
            if missing:
                raise ValueError(f"[{self.name}] require_y=True but y missing at indices: {missing}")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> AttributedNetwork:
        return self._samples[idx]

    def __iter__(self):
        return iter(self._samples)


# ============================================================
# DataLoader Structure
# ============================================================

class AttributedNetworkDataLoader:
    """
    DataLoader for attributed network data (transition matrices + node features).

    Validation performed on load:

    1) Transition matrix P checks:
       - P is 2D and square (n x n)
       - row-stochastic: nonnegative entries, row sums equal to 1
       - irreducible: support graph of P is strongly connected

    2) Node feature matrix X checks:
       - X is 2D with shape (n, p) where n matches P.shape[0]
       - feature types are consistent across graphs within each split:
         each dimension is either all numeric or all string (categorical)

    3) Train/test consistency (optional but default True):
       - train and test have the same feature type per dimension

    4) Label rules:
       - supervised=True requires y for every train graph
       - test labels are optional
       - if y exists for both train and test, we enforce that:
           * train labels are all the same Python type
           * test labels are all the same Python type
           * train label type matches test label type

    Cost matrix computation:

      After building the loader, call calculate_cost(...) to compute node-to-node
      cost matrices for each feature dimension between network pairs.

      Cost definition per feature dimension:
        - string features: 0/1 mismatch cost (get_01_cost)
        - numeric features: squared difference cost (get_sq_cost)

      Output format (dimension-first):
        costs[mode][d][(i, j)] -> ndarray cost matrix for feature dim d between network i and j.
    """

    def __init__(
        self,
        *,
        p: int,
        train_data: Sequence[Dict[str, Any]],
        test_data: Optional[Sequence[Dict[str, Any]]] = None,
        supervised: bool = True,
        transition_atol: float = 1e-8,
        irreducible_atol: float = 1e-12,
        enforce_train_test_feature_consistency: bool = True,
    ) -> None:
        """
        Construct a DataLoader from list-of-dicts input.

        Most users should prefer from_lists(...) which accepts parallel lists.

        Args:
            p (int): Feature dimension (number of columns in X).
            train_data (Sequence[Dict[str, Any]]): Each dict must contain:
                - "P": transition matrix (n, n)
                - "X": node feature matrix (n, p)
                - optional "y": label
            test_data (Optional[Sequence[Dict[str, Any]]]): Same format as train_data.
            supervised (bool): If True, require y for all training samples.
            transition_atol (float): Tolerance for row-stochastic check.
            irreducible_atol (float): Threshold for irreducibility graph edges.
            enforce_train_test_feature_consistency (bool): Enforce train/test kind match.

        Raises:
            ValueError / TypeError: For any validation failure.
        """
        if not isinstance(p, int) or p <= 0:
            raise ValueError(f"p must be a positive integer, got {p!r}.")

        self.p = p
        self.supervised = supervised
        self.transition_atol = transition_atol
        self.irreducible_atol = irreducible_atol
        self.enforce_train_test_feature_consistency = enforce_train_test_feature_consistency

        self._feature_kinds: Dict[str, List[str]] = {}

        # NOTE: dimension-first cost storage:
        #   self.costs[mode][d][(i,j)] = ndarray
        self.costs: Dict[str, Dict[int, Dict[Tuple[int, int], np.ndarray]]] = {}

        self.train = self._build_dataset(train_data, where="train", require_y=self.supervised)

        self.test: Optional[AttributedNetworkDataset] = None
        if test_data is not None:
            self.test = self._build_dataset(test_data, where="test", require_y=False)

            if self.enforce_train_test_feature_consistency:
                _check_train_test_feature_kinds_match(
                    self._feature_kinds["train"],
                    self._feature_kinds["test"],
                )

    @classmethod
    def from_lists(
        cls,
        *,
        p: int,
        P_train: Sequence[ArrayLike],
        X_train: Sequence[ArrayLike],
        y_train: Optional[Sequence[Any]] = None,
        P_test: Optional[Sequence[ArrayLike]] = None,
        X_test: Optional[Sequence[ArrayLike]] = None,
        y_test: Optional[Sequence[Any]] = None,
        supervised: bool = True,
        transition_atol: float = 1e-8,
        irreducible_atol: float = 1e-12,
        enforce_train_test_feature_consistency: bool = True,
    ) -> "AttributedNetworkDataLoader":
        """
        Convenience constructor from parallel lists of (P, X, y).

        This avoids having to manually build list-of-dicts. It performs:
          - length checks across P/X/y lists
          - a printed warning about alignment/order
          - optional train/test label type consistency checks

        Args:
            p (int): Feature dimension (number of columns in X).
            P_train (Sequence[ArrayLike]): Training transition matrices.
            X_train (Sequence[ArrayLike]): Training node feature matrices.
            y_train (Optional[Sequence[Any]]): Training labels (required if supervised=True).
            P_test (Optional[Sequence[ArrayLike]]): Test transition matrices.
            X_test (Optional[Sequence[ArrayLike]]): Test node feature matrices.
            y_test (Optional[Sequence[Any]]): Test labels (optional).
            supervised (bool): If True, require y_train and enforce label checks.
            transition_atol (float): Tolerance for row-stochastic check.
            irreducible_atol (float): Threshold for irreducibility graph edges.
            enforce_train_test_feature_consistency (bool): Enforce train/test kind match.

        Returns:
            AttributedNetworkDataLoader: A validated loader instance.

        Raises:
            ValueError: If list lengths mismatch or required lists are missing.
            TypeError: If train/test label types are inconsistent (when both provided).
        """
        _validate_parallel_lists(
            where="train",
            P_list=P_train,
            X_list=X_train,
            y_list=y_train,
            y_required=supervised,
        )

        train_data = []
        for i in range(len(P_train)):
            item = {"P": P_train[i], "X": X_train[i]}
            if y_train is not None:
                item["y"] = y_train[i]
            train_data.append(item)

        test_data = None
        if P_test is not None or X_test is not None or y_test is not None:
            if P_test is None or X_test is None:
                raise ValueError(
                    "[test] To provide a test split, you must provide BOTH P_test and X_test. "
                    f"Got len(P_test)={_len_or_none(P_test)} and len(X_test)={_len_or_none(X_test)}."
                )

            _validate_parallel_lists(
                where="test",
                P_list=P_test,
                X_list=X_test,
                y_list=y_test,
                y_required=False,
            )

            test_data = []
            for i in range(len(P_test)):
                item = {"P": P_test[i], "X": X_test[i]}
                if y_test is not None:
                    item["y"] = y_test[i]
                test_data.append(item)

        # Label type consistency check (train vs test) if test labels are provided.
        if supervised and y_train is not None and y_test is not None:
            if len(y_train) > 0 and len(y_test) > 0:
                train_type = type(y_train[0])
                test_type = type(y_test[0])

                if train_type != test_type:
                    raise TypeError(
                        f"[label check] Train and test label types differ: "
                        f"{train_type} (train) vs {test_type} (test)."
                    )

                if not all(isinstance(y, train_type) for y in y_train):
                    raise TypeError("[label check] Not all train labels share the same type.")
                if not all(isinstance(y, test_type) for y in y_test):
                    raise TypeError("[label check] Not all test labels share the same type.")

        return cls(
            p=p,
            train_data=train_data,
            test_data=test_data,
            supervised=supervised,
            transition_atol=transition_atol,
            irreducible_atol=irreducible_atol,
            enforce_train_test_feature_consistency=enforce_train_test_feature_consistency,
        )

    # ----------------------------
    # API
    # ----------------------------

    def train_loader(self) -> AttributedNetworkDataset:
        """Return the training dataset."""
        return self.train

    def test_loader(self) -> Optional[AttributedNetworkDataset]:
        """Return the test dataset, or None if no test split was provided."""
        return self.test

    def get_feature_kinds(self, split: str = "train") -> List[str]:
        """
        Return the inferred feature kinds ("numeric"/"string") for a split.

        Args:
            split (str): "train" or "test".

        Returns:
            List[str]: Per-dimension kinds, length p.

        Raises:
            KeyError: If split kinds were not stored (e.g., test split missing).
        """
        if split not in self._feature_kinds:
            raise KeyError(f"No feature kinds stored for split={split!r}.")
        return list(self._feature_kinds[split])

    def calculate_cost(
        self,
        *,
        within_train: bool = True,
        train_test: bool = False,
        within_test: bool = False,
        store: bool = True,
        include_self: bool = True,
    ) -> Dict[str, Dict[int, Dict[Tuple[int, int], np.ndarray]]]:
        """
        Compute node-to-node cost matrices between network pairs.

        Modes:
          - within_train: all train-vs-train pairs
          - train_test:   all train-vs-test pairs
          - within_test:  all test-vs-test pairs

        Output (dimension-first):
            result[mode][d][(i, j)] = cost matrix for feature dim d
                                      between network i and network j.

        Args:
            within_train (bool): Compute train-train costs.
            train_test (bool): Compute train-test costs.
            within_test (bool): Compute test-test costs.
            store (bool): If True, merge results into self.costs and return self.costs.
            include_self (bool): If True, include (i, i) for within-split modes.

        Returns:
            Dict[str, Dict[int, Dict[Tuple[int, int], np.ndarray]]]: Cost dictionary.

        Raises:
            ValueError: If test split is missing but train_test/within_test requested.
        """
        if (train_test or within_test) and self.test is None:
            raise ValueError(
                "Test loader is empty (test_data=None), but train_test or within_test was requested."
            )

        result: Dict[str, Dict[int, Dict[Tuple[int, int], np.ndarray]]] = {}

        if within_train:
            result["within_train"] = self._compute_pair_costs(
                A=self.train, A_split="train",
                B=self.train, B_split="train",
                include_self=include_self,
                symmetric=True,
            )

        if train_test:
            assert self.test is not None
            result["train_test"] = self._compute_pair_costs(
                A=self.train, A_split="train",
                B=self.test,  B_split="test",
                include_self=True,
                symmetric=False,
            )

        if within_test:
            assert self.test is not None
            result["within_test"] = self._compute_pair_costs(
                A=self.test, A_split="test",
                B=self.test, B_split="test",
                include_self=include_self,
                symmetric=True,
            )

        if store:
            for k, v in result.items():
                self.costs[k] = v
            return self.costs

        return result

    # ----------------------------
    # Internal: build dataset
    # ----------------------------

    def _build_dataset(
        self,
        data: Sequence[Dict[str, Any]],
        *,
        where: str,
        require_y: bool,
    ) -> AttributedNetworkDataset:
        """
        Build and validate a dataset split from list-of-dicts.

        Args:
            data (Sequence[Dict[str, Any]]): Each dict must include "P" and "X".
            where (str): Split label ("train" or "test").
            require_y (bool): If True, require y for every sample.

        Returns:
            AttributedNetworkDataset: Validated dataset.

        Raises:
            ValueError / TypeError: If any sample violates required constraints.
        """
        if data is None:
            raise ValueError(f"[{where}] data is None.")
        if not isinstance(data, (list, tuple)):
            raise TypeError(f"[{where}] data must be a list/tuple of dicts, got {type(data)}.")

        samples: List[AttributedNetwork] = []
        X_list: List[np.ndarray] = []

        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise TypeError(f"[{where}] item {i} must be a dict with keys P, X, (optional) y.")
            if "P" not in item or "X" not in item:
                raise KeyError(f"[{where}] item {i} must contain keys 'P' and 'X'.")

            P = _as_numpy_2d(item["P"], name=f"[{where}] P[{i}]")
            X = _as_numpy_2d(item["X"], name=f"[{where}] X[{i}]")
            y = item.get("y", None)

            if not _is_square(P):
                raise ValueError(f"[{where}] P[{i}] must be square (n x n), got shape {P.shape}.")
            n = P.shape[0]

            if X.shape != (n, self.p):
                raise ValueError(
                    f"[{where}] X[{i}] must have shape (n, p)=({n}, {self.p}), got {X.shape}."
                )

            if not _is_row_stochastic(P, atol=self.transition_atol):
                raise ValueError(
                    f"[{where}] P[{i}] is not a valid row-stochastic transition matrix "
                    f"(rows must sum to 1 and entries must be nonnegative)."
                )

            if not _is_irreducible_transition_matrix(P, atol=self.irreducible_atol):
                raise ValueError(
                    f"[{where}] P[{i}] is not irreducible (support graph is not strongly connected)."
                )

            if require_y and y is None:
                raise ValueError(f"[{where}] y is required (supervised) but missing at index {i}.")

            samples.append(AttributedNetwork(P=P, X=X, y=y))
            X_list.append(X)

        kinds = _check_feature_kinds_consistent(X_list, self.p, where=where)
        self._feature_kinds[where] = kinds

        return AttributedNetworkDataset(samples, name=where, require_y=require_y)

    # ----------------------------
    # Internal: cost computation
    # ----------------------------

    def _compute_pair_costs(
        self,
        *,
        A: AttributedNetworkDataset,
        A_split: str,
        B: AttributedNetworkDataset,
        B_split: str,
        include_self: bool,
        symmetric: bool,
    ) -> Dict[int, Dict[Tuple[int, int], np.ndarray]]:
        """
        Compute per-dimension node-to-node costs for all network pairs.

        Output format:
            out[d][(i, j)] = cost matrix between feature dimension d of X_i and X_j.

        Args:
            A (AttributedNetworkDataset): Left dataset.
            A_split (str): Split label for left dataset ("train"/"test").
            B (AttributedNetworkDataset): Right dataset.
            B_split (str): Split label for right dataset ("train"/"test").
            include_self (bool): Whether to include pairs (i, i) when A is B.
            symmetric (bool): If True and A is B, only compute upper triangle
                (i <= j if include_self else i < j).

        Returns:
            Dict[int, Dict[Tuple[int, int], np.ndarray]]: Dimension-first cost dict.

        Raises:
            TypeError: If feature kinds differ between splits.
        """
        kinds_A = self.get_feature_kinds(A_split)
        kinds_B = self.get_feature_kinds(B_split)

        if kinds_A != kinds_B:
            raise TypeError(
                f"Feature kinds differ between splits {A_split!r} and {B_split!r}. "
                f"{A_split}: {kinds_A}, {B_split}: {kinds_B}."
            )
        kinds = kinds_A

        out: Dict[int, Dict[Tuple[int, int], np.ndarray]] = {d: {} for d in range(self.p)}

        for i in range(len(A)):
            j_start = 0
            if symmetric and A is B:
                j_start = i if include_self else i + 1

            for j in range(j_start, len(B)):
                if (A is B) and (i == j) and (not include_self):
                    continue

                Xi = A[i].X
                Xj = B[j].X

                for d in range(self.p):
                    kind = kinds[d]
                    v1 = _column_as_list(Xi, d, kind)
                    v2 = _column_as_list(Xj, d, kind)

                    if kind == "string":
                        C = get_01_cost(v1, v2)
                    elif kind == "numeric":
                        C = get_sq_cost(v1, v2)
                    else:
                        raise ValueError(f"Unexpected kind {kind!r} at dim {d}.")

                    out[d][(i, j)] = C

        return out