import numpy as np
from sklearn.tree import _tree

import omnialigner as om
plt = om.pl.plt

feature_names = ['CD3E', 'MS4A1', 'FCER2', 'LAMP3', 'CR2', 'GLYCAM1',
                 'CD274', 'ITGAX', 'TPSAB1', 'HLA-DRA', 'CD163', 'CD68',
                'NCAM1', 'MPO', 'ACTA2', 'MKI67', 'PECAM1', 'PANK1', 
                'CD4', 'CD8A', 'FOXP3', 'PDCD1', 'MS4A1', 'IL10'
]

LEAF = _tree.TREE_LEAF        # == -1

nodes = [
    ("PANK1",        False),   # 0   ← was  PanCK
    ("Cancer cell",  True),    # 1
    ("CD68",         False),   # 2
    ("CD163",        False),   # 3
    ("CD163+ Mac",   True),    # 4
    ("CD163- Mac",   True),    # 5
    ("CD3E",         False),   # 6
    ("CD8A",         False),   # 7
    ("Tc",           True),    # 8
    ("CD4",          False),   # 9
    ("FOXP3",        False),   #10
    ("Treg",         True),    #11
    ("Th",           True),    #12
    ("DNT",          True),    #13
    ("MS4A1",         False),  #14  ← was  CD20
    ("B Cell",       True),    #15
    ("MPO",          False),   #16
    ("Neu",          True),    #17
    ("NCAM1",         False),  #18  ← was  CD56
    ("NK",           True),    #19
    ("ITGAX",        False),  #20  ← was CD11C
    ("HLA-DRA",       False),  #21  ← HLA-DR
    ("Dendritic Cell", True),  #22
    ("PECAM1",         False), #23  ← was  CD31
    ("Endothelial",  True),    #24
    ("TPSAB1",       False),   #25  ← was  Tryptase
    ("Mast cell",    True),    #26
    ("ACTA2",        False),   #27  # α-SMA
    ("Fibroblast",   True),    #28
    ("Undefined",    True)     #29
]

# ── 2. 旧名 → 新名 映射 ────────────────────────────────────────────────────
name_mapping = {
    "PANK1" : "PanCK",
    "MS4A1" : "CD20",
    "NCAM1" : "CD56",
    "ITGAX" : "CD11C",
    "HLA-DRA" : "HLA-DR",
    "PECAM1" : "CD31",
    "TPSAB1" : "Tryptase",
}

children_right = np.array([
     1, LEAF,  3,  4, LEAF, LEAF,  7,  8, LEAF, 10,
    11, LEAF, LEAF, LEAF, 15, LEAF, 17, LEAF, 19, LEAF,
    21, 22,  LEAF, 24, LEAF, 26, LEAF, 28, LEAF, LEAF
], dtype=int)

children_left = np.array([
     2, LEAF,  6,  5, LEAF, LEAF, 14,  9, LEAF, 13,
    12, LEAF, LEAF, LEAF, 16, LEAF, 18, LEAF, 20, LEAF,
    23, 23,  LEAF, 25, LEAF, 27, LEAF, 29, LEAF, LEAF
], dtype=int)


leaf_color_map = {
    # ── Lymphoid ─────────────────────────
    "B Cell"          : "#004488",
    "Tc"              : "#FF0000",
    "Th"              : "#FF6666",
    "Treg"            : "#FF66B3",
    "DNT"             : "#8B2500",
    "NK"              : "#0066BB",

    # ── Myeloid ─────────────────────────
    "CD163+ Mac"      : "#FFD700",
    "CD163- Mac"      : "#009E60",
    "Monocyte"        : "#7DB317",
    "Dendritic Cell"  : "#B03AAC",
    "Neu"             : "#44C2A5",

    # ── Stroma / Structure ──────────────
    "Epithelial"      : "#8B4513",
    "Cancer cell"     : "#00D4FF",
    "Fibroblast"      : "#FFAA00",
    "Vascular"        : "#BB6600",
    "Endothelial"     : "#FFC0CB",
    "Mast cell"       : "#A1DB00",
    "Undefined"       : "#EE7700",
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
