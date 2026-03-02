import numpy as np
from sklearn.tree import _tree

import omnialigner as om
plt = om.pl.plt

feature_names = [
    'CD3',  'Ki67', 'CD4', 'CD20', 'CD163', 'Ecadherin', 'LaminABC', 'PCNA',
    'NaKATPase', 'Keratin', 'CD45', 'CD68', 'FOXP3',  'Vimentin', 'Desmin', 'Ki67_570',
    'CD45RO', 'aSMA','PD1', 'CD8a', 'PDL1', 'CDX2', 'CD31', 'Collagen'
]

LEAF = _tree.TREE_LEAF        # == -1

nodes = [
    ("Keratin",          False),  # 0   ← was PanCK (epithelial/tumor gate)
    ("Cancer cell",      True),   # 1
    ("CD45",             False),  # 2   immune vs stromal
    ("CD68",             False),  # 3   myeloid/macrophage gate
    ("CD163",            False),  # 4
    ("CD163+ Mac",       True),   # 5
    ("CD163- Mac",       True),   # 6
    ("CD3",              False),  # 7   T cell gate
    ("CD8a",             False),  # 8
    ("Tc",               True),   # 9
    ("CD4",              False),  # 10
    ("FOXP3",            False),  # 11
    ("Treg",             True),   # 12
    ("Th",               True),   # 13
    ("DNT",              True),   # 14  # CD3+ CD4- CD8-
    ("CD20",             False),  # 15  B cell gate
    ("B Cell",           True),   # 16
    ("Undefined immune", True),   # 17
    ("CD31",             False),  # 18  endothelial gate
    ("Endothelial",      True),   # 19
    ("aSMA",             False),  # 20  myofibroblast / pericyte-like
    ("Myofibroblast",    True),   # 21
    ("Collagen",         False),  # 22  fibroblast ECM-like
    ("Fibroblast",       True),   # 23
    ("Undefined stromal",True)    # 24
]

# ── 名称映射（可选，用于与旧图/旧marker命名对齐）────────────────────────────
name_mapping = {
    "Keratin": "PanCK",
    "CD3": "CD3E",
    "aSMA": "ACTA2",
    "PD1": "PDCD1",
    "PDL1": "CD274",
    "CD31": "PECAM1",
}

children_right = np.array([
     1, LEAF,  3,  4,  5, LEAF, LEAF,  8,  9, LEAF,
    11, 12, LEAF, LEAF, LEAF, 16, LEAF, LEAF, 19, LEAF,
    21, LEAF, 23, LEAF, LEAF
], dtype=int)

children_left = np.array([
     2, LEAF, 18,  7,  6, LEAF, LEAF, 15, 10, LEAF,
    14, 13, LEAF, LEAF, LEAF, 17, LEAF, LEAF, 20, LEAF,
    22, LEAF, 24, LEAF, LEAF
], dtype=int)

leaf_color_map = {
    # ── Lymphoid ─────────────────────────
    "B Cell"            : "#004488",
    "Tc"                : "#FF0000",
    "Th"                : "#FF6666",
    "Treg"              : "#FF66B3",
    "DNT"               : "#8B2500",
    "NK"                : "#0066BB",

    # ── Myeloid ─────────────────────────
    "CD163+ Mac"        : "#FFD700",
    "CD163- Mac"        : "#009E60",
    "Monocyte"          : "#7DB317",
    "Dendritic Cell"    : "#B03AAC",
    "Neu"               : "#44C2A5",

    # ── Stroma / Structure ──────────────
    "Epithelial"        : "#8B4513",
    "Cancer cell"       : "#00D4FF",
    "Fibroblast"        : "#FFAA00",
    "Vascular"          : "#BB6600",
    "Endothelial"       : "#FFC0CB",
    "Mast cell"         : "#A1DB00",

    # ── Fallback / Other ─────────────────
    "Myofibroblast"     : "#BB6600",  # if you use this leaf name
    "Undefined immune"  : "#EE7700",  # if you use this leaf name
    "Undefined stromal" : "#EE7700",  # if you use this leaf name
    "Undefined"         : "#EE7700",
}


if __name__ == "__main__":
    children_left, children_right = children_right, children_left
    # Build adjacency for plotting
    edges = []
    for idx, (lch, rch) in enumerate(zip(children_left, children_right)):
        if lch != LEAF:
            edges.append((idx, lch))
        if rch != LEAF:
            edges.append((idx, rch))

    # Compute depth of each node
    depth = [0]*len(nodes)
    for parent, child in edges:
        depth[child] = depth[parent] + 1

    # Assign x positions by ordering leaves
    leaf_order = []
    def dfs(n):
        if children_left[n] == LEAF and children_right[n] == LEAF:
            leaf_order.append(n)
        else:
            if children_left[n] != LEAF:
                dfs(children_left[n])
            if children_right[n] != LEAF:
                dfs(children_right[n])
    dfs(0)

    x_pos = {}
    for i, leaf in enumerate(leaf_order):
        x_pos[leaf] = i

    # Now propagate x positions up
    def set_x(n):
        if n in x_pos:
            return x_pos[n]
        xs = []
        if children_left[n] != LEAF:
            xs.append(set_x(children_left[n]))
        if children_right[n] != LEAF:
            xs.append(set_x(children_right[n]))
        x = sum(xs)/len(xs)
        x_pos[n] = x
        return x

    set_x(0)
    
    # Plot
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    for parent, child in edges:
        ax.plot([x_pos[parent], x_pos[child]], [-depth[parent], -depth[child]], 'k-')

    for idx, (label, is_leaf) in enumerate(nodes):
        if is_leaf:
            c = leaf_color_map.get(label, "#808080")
            s = 120
        else:
            c = "#333333"
            s =40

        label_used = name_mapping.get(label, label)
        ax.scatter(x_pos[idx], -depth[idx], c=c, s=s, edgecolor="k", zorder=3)
        ax.text(x_pos[idx], -depth[idx] + 0.25, label_used,
             ha="center", va="bottom", fontsize=8, wrap=True)


    ax.axis('off')
    ax.set_title("Marker-based cell-type decision tree")
    fig.savefig("/cluster/home/bqhu_jh/projects/omni/notebook/fig_celltype.pdf")
    # fig.show()
    # plt.show()
    print("children_left =", children_left)
    print("children_right=", children_right)
