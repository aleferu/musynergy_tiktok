#!/usr/bin/env python3


import torch


if __name__ == '__main__':
    # Load
    file_path = "intiktok_pyg/ds/full_hd_0_intiktok.pt"
    data = torch.load(file_path)

    print("### Heterogeneous Graph Overview ###\n")

    # Print all Node Types
    print(f"**Node Types ({len(data.node_types)}):**")
    for node_type, store in data.node_items():
        count = store.num_nodes
        print(f"- {node_type}: {count:,} nodes")

    print("\n" + "-"*30 + "\n")

    # Print all Relationship (Edge) Types
    print(f"**Relationship Types ({len(data.edge_types)}):**")
    for edge_type, store in data.edge_items():
        count = store.num_edges
        print(f"- {edge_type}: {count:,} edges")

    print("\n" + "-"*30 + "\n")

    # Print Metadata (Detailed structure)
    print("**Full Metadata (Nodes and Edges):**")
    node_meta, edge_meta = data.metadata()
    print(f"Nodes: {node_meta}")
    print(f"Edges: {edge_meta}")
