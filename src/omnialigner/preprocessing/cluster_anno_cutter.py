import json
from multiprocessing import Pool
from typing import List, Sequence, Tuple, Dict, OrderedDict

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree._tree import Tree
from sklearn.datasets import make_classification
from scipy.spatial.distance import pdist
import optuna

LEAF = -1


def leaf_metrics(tree: Tree, X: np.ndarray, dict_init_thresh: Dict[int, float], weight_cov: float = 0.5, weight_purity: float = 0.5, alpha: float = 0.1):
    """
    Evaluate tree coverage, purity, and regularization penalty from initial thresholds.

    Parameters:
        tree        : trained tree
        X           : input features (n_samples, n_features)
        init_thresh : np.ndarray, same shape as tree.threshold
        alpha       : penalty weight

    Returns:
        coverage, purity, reg_penalty, total_score
    """
    
    X = X.astype(np.float32)
    leaf_id = tree.apply(X)
    leaves, _ = np.unique(leaf_id, return_counts=True)

    
    n_leaves_total = np.sum(tree.children_left == LEAF)
    coverage = len(leaves) / n_leaves_total if n_leaves_total > 0 else 0.0

    
    purity_list = []
    for l in leaves:
        idx = (leaf_id == l)
        X_l = X[idx]
        if len(X_l) == 1:
            purity_list.append(1.0)
        else:
            d = pdist(X_l, metric='euclidean').mean()
            purity_list.append(1 - d / np.sqrt(X.shape[1]))
    purity = np.mean(purity_list) if purity_list else 0.0

    
    internal_nodes = np.where(tree.children_left != LEAF)[0]
    threshold_now = tree.threshold[internal_nodes]
    threshold_init = np.array([dict_init_thresh.get(i, 0.3) for i in internal_nodes])
    reg_penalty = np.mean((threshold_now - threshold_init) ** 2)

    
    total_score = weight_cov * coverage + weight_purity * purity - alpha * reg_penalty

    return total_score


def predict_tree(clf: DecisionTreeClassifier, best_params: Dict, nodes: Sequence[Tuple[str, bool]], df_avg: pd.DataFrame) -> Dict[str, str]:
    """
    Predict labels for the average expression matrix using the optimized decision tree.
    
    Args:
        clf: The trained DecisionTreeClassifier with fixed topology.
        best_params: Dictionary of optimized threshold parameters for internal nodes.
        df_avg: DataFrame containing the average expression matrix.
    
    Returns:
        dict_results: Dictionary mapping sample names to predicted labels.
    
    """
    X = df_avg.values  
    for k, v in best_params.items():
        node_id = int(k.split("_")[-1])  
        clf.tree_.threshold[node_id] = v
    
    leaf_ids = clf.apply(X)
    leaf_id_to_label = {
        i: name for i, (name, is_leaf) in enumerate(nodes) if is_leaf
    }
    # print("\nResults:")
    dict_results = {}
    for i, leaf_id in enumerate(leaf_ids):
        label = leaf_id_to_label.get(leaf_id, "Unknown")
        group_name = df_avg.index[i]
        dict_results[group_name] = label
        # print(f"  Sample {i}: leaf {leaf_id}, {group_name} → {label}")

    return dict_results


def build_fake_tree_shell(n_nodes: int, n_features: int) -> DecisionTreeClassifier:
    """
    Auxiliary function to build a fake decision tree shell with at least n_nodes.
    This is used to write a fixed topology tree.

    Args:
        n_nodes: The minimum number of nodes in the tree.
        n_features: The number of features in the dataset.

    Returns:
        clf: A DecisionTreeClassifier with a tree structure that has at least n_nodes.

    """
    depth = 1
    while True:
        X_dummy, y_dummy = make_classification(
            n_samples=2 ** (depth + 1),
            n_features=n_features,
            n_informative=2,             
            n_redundant=0,
            n_classes=2,
            n_clusters_per_class=1,      
            random_state=42
        )

        clf = DecisionTreeClassifier(
            max_depth=depth,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        clf.fit(X_dummy, y_dummy)

        if clf.tree_.node_count >= n_nodes:
            return clf

        depth += 1
        if depth > n_nodes:  
            raise ValueError("Cannot build a tree with the required number of nodes. Please check the parameters.")

def build_fixed_tree(
    X: np.ndarray,
    nodes: Sequence[Tuple[str, bool]],
    feature_names: Sequence[str],
    children_left: np.ndarray,
    children_right: np.ndarray,
    init_thresh: np.ndarray = None,
) -> Tuple[DecisionTreeClassifier, np.ndarray]:
    """
    Construct a fixed topology decision tree classifier.
    This function builds a decision tree classifier with a fixed topology defined by the children_left and children_right arrays.

    Args:
        X: Feature matrix, shape (n_samples, n_features)
        children_left / children_right: Topology arrays, -1 indicates leaf
        nodes: Names and leaf status of each node, e.g. ("CD68", False)
        feature_names: All gene names, used to map node names to column indices
        init_thresh: Initial cut-off for all internal nodes

    Returns:
        clf: sklearn.tree.DecisionTreeClassifier (with fixed topology written in)
        internal_nodes: All internal node indices (for external optimization of their cut-offs)
    """
    n_nodes = len(nodes)
    if init_thresh is not None and len(init_thresh) != X.shape[1]:
        raise ValueError(f"init_thresh must have the same length as the number of features ({X.shape[1]}), but got {len(init_thresh)}")

    name2idx = {name: i for i, name in enumerate(feature_names)}
    clf = build_fake_tree_shell(n_nodes=len(nodes), n_features=X.shape[1])
    tree = clf.tree_
    tree.children_left[:n_nodes] = children_left
    tree.children_right[:n_nodes] = children_right
    internal_nodes = np.where(children_left != LEAF)[0]
    dict_internal_thresh = {}
    for i in internal_nodes:
        name, is_leaf = nodes[i]
        assert not is_leaf, f"Node {i} were supposed to be internal, but is_leaf is True"
        assert name in name2idx, f"Node feature name {name} not in feature_names"
        f_idx = name2idx[name]
        tree.feature[i] = f_idx
        tree.threshold[i] = init_thresh[f_idx] if init_thresh is not None else 0.3
        dict_internal_thresh[i] = tree.threshold[i]
    
    return clf, internal_nodes, dict_internal_thresh

def result_to_df(l_res: List[Dict[str, str]]) -> pd.DataFrame:
    """
    Convert a list of sample-to-label prediction dictionaries into a count matrix.

    Each dict in l_res represents one round of prediction, e.g. from one threshold set.

    Returns:
        DataFrame: rows = sample names, columns = predicted labels, values = counts
    """
    # 1. Collect all unique samples and labels
    all_samples = set()
    all_labels = set()
    for res in l_res:
        all_samples.update(res.keys())
        all_labels.update(res.values())

    l_rows = sorted(all_samples)
    l_cols = sorted(all_labels)

    # 2. Build fast lookup index
    sample2idx = {s: i for i, s in enumerate(l_rows)}
    label2idx = {l: j for j, l in enumerate(l_cols)}

    # 3. Fill matrix
    np_matrix = np.zeros((len(l_rows), len(l_cols)), dtype=np.intp)
    for res in l_res:
        for sample, label in res.items():
            i = sample2idx[sample]
            j = label2idx[label]
            np_matrix[i, j] += 1

    return pd.DataFrame(np_matrix, index=l_rows, columns=l_cols)

def _optimize_tree_worker(args):
    """
    Top-level worker for one tree optimization.
    Args is a tuple:
      (X, n_trials, init_thresh, weight_cov, weight_purity, alpha, df_avg)
    """
    X, n_trials, init_thresh, weight_cov, weight_purity, alpha, df_avg, nodes, feature_names, children_left, children_right = args
    clf, internal_nodes, dict_internal_thresh = build_fixed_tree(
        X,
        nodes=nodes,
        feature_names=feature_names,
        children_left=children_left,
        children_right=children_right,
        init_thresh=init_thresh
    )

    def _obj(trial):
        for nid in internal_nodes:
            clf.tree_.threshold[nid] = trial.suggest_float(f"t_{nid}", 0.0, 1.0)
        score = leaf_metrics(
            clf.tree_, X,
            dict_init_thresh=dict_internal_thresh,
            weight_cov=weight_cov,
            weight_purity=weight_purity,
            alpha=alpha
        )
        return -score

    optuna.logging.disable_default_handler()
    study = optuna.create_study()
    study.optimize(_obj, n_trials=n_trials)
    pred_result = predict_tree(clf, study.best_params, nodes, df_avg)
    return pred_result, study.best_params


def train_cut_model(
        df_avg: pd.DataFrame,
        nodes: Sequence[Tuple[str, bool]],
        feature_names: Sequence[str],
        children_left: Sequence[int],
        children_right: Sequence[int],
        result_params: str = "tree_params.json",
        n_trees=10,
        n_trials=30,
        init_thresh: np.ndarray = None,
        weight_cov: float = 0.5,
        weight_purity: float = 0.5,
        alpha: float = 0.1):
    """

    Search the decision tree thresholds for `sc.pl.Dotplot` results:
    ```
        dotplot = sc.pl.DotPlot(adata, standard_scale="var", **kwargs)
        df_avg = dotplot.dot_color_df
    ```

    Args:
        df_avg: DataFrame containing the average expression matrix.
        nodes: Sequence of tuples (name, is_leaf) for each node in the tree.
        feature_names: List of feature names corresponding to the columns in df_avg.
        children_left: Array of left children indices for each node in the tree.
        children_right: Array of right children indices for each node in the tree.
        n_trees: Number of trees to optimize in parallel.
        n_trials: Number of optimization trials.
        init_thresh: Initial thresholds for internal nodes.
        weight_cov: Weight for coverage in the objective function.
        weight_purity: Weight for purity in the objective function.
        alpha: Regularization penalty weight.
    
    """

    print(df_avg.round(2).to_markdown())
    X = df_avg.values
    if init_thresh is None:
        init_thresh = np.zeros(X.shape[1], dtype=np.float32) + 0.3

    # prepare identical args for each tree
    worker_args = [
        (X, n_trials, init_thresh, weight_cov, weight_purity, alpha, df_avg, nodes, feature_names, children_left, children_right)
        for _ in range(n_trees)
    ]

    # run in parallel
    with Pool() as pool:
        l_results = pool.map(_optimize_tree_worker, worker_args)

    l_pred_dict, l_params = zip(*l_results)
    df_pred = result_to_df(l_pred_dict)
    with open(result_params, "w", encoding="utf-8") as f:
        json.dump(l_params, f, indent=2)
    
    return df_pred


def predict_cut_model(
        df_all: pd.DataFrame,
        file_params: str,
        nodes: Sequence[Tuple[str, bool]],
        feature_names: Sequence[str],
        children_left: Sequence[int],
        children_right: Sequence[int]
    ) -> pd.DataFrame:
    """
    Predict labels for the average expression matrix using the optimized decision tree.
    This function loads the best parameters from a JSON file and uses them to predict labels for the input DataFrame.

    Args:
        df_avg: DataFrame containing the average expression matrix.
        file_params: Path to the JSON file containing the best parameters for the decision tree.
        nodes: Sequence of tuples (name, is_leaf) for each node in the tree.
        feature_names: List of feature names corresponding to the columns in df_avg.
        children_left: Array of left children indices for each node in the tree.
        children_right: Array of right children indices for each node in the tree.

    Returns:
        DataFrame: A DataFrame containing the predicted labels for each sample in df_avg.
    """
    l_best_params = json.load(open(file_params, "r", encoding="utf-8"))
    clf, _, _ = build_fixed_tree(
        df_all.values,
        nodes=nodes,
        feature_names=feature_names,
        children_left=children_left,
        children_right=children_right
    )
    # 并行执行 predict_tree
    with Pool() as pool:
        args = [(clf, params, nodes, df_all) for params in l_best_params]
        l_pred_dict = pool.starmap(predict_tree, args)

    df_pred = result_to_df(l_pred_dict)
    return df_pred


if __name__ == "__main__":
    from test_case import generate_test_matrix, example_marker1
    # from pdac_tree import nodes, children_left, children_right, feature_names
    from omnialigner.configs.pdac_tree import nodes, children_left, children_right, feature_names
    
    df_avg = generate_test_matrix(example_marker1())
    print("Test matrix generated successfully.")

    df_pred = train_cut_model(
        df_avg,
        n_trials=300,
        result_params="tree_params.json",
        nodes=nodes,
        feature_names=feature_names,
        children_left=children_left,
        children_right=children_right
    )
    print(df_pred.round(2).to_markdown())

    print("\n\nPredicting with the same model on the same data:")
    df_pred_all = predict_cut_model(df_avg, "tree_params.json", nodes=nodes, feature_names=feature_names, children_left=children_left, children_right=children_right)
    print(df_pred_all.round(2).to_markdown())