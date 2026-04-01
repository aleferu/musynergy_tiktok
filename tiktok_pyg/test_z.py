#!/usr/bin/env python3

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import HeteroConv, GATConv, SAGEConv, GATv2Conv
from torch_geometric.loader import LinkNeighborLoader
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
import tqdm
import os
import os.path as path
import argparse
import logging
import time
import pickle
import csv


class GNN_NOCAT(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels, artist_channels, track_channels, tag_channels):
        super().__init__()
        self.metadata = metadata
        self.out_channels = out_channels

        self.conv1 = HeteroConv({
            ("artist", "collab_with", "artist"): GATConv((artist_channels, artist_channels), hidden_channels, heads=3, concat=False),
            ("artist", "has_tag_artists", "tag"): SAGEConv((artist_channels, tag_channels), hidden_channels, normalize=True, project=True),
            ("artist", "last_fm_match", "artist"): GATv2Conv((artist_channels, artist_channels), hidden_channels, heads=3, concat=False, edge_dim=1),
            ("artist", "follows", "artist"): GATv2Conv((artist_channels, artist_channels), hidden_channels, heads=3, concat=False, edge_dim=1),
            ("track", "has_tag_tracks", "tag"): SAGEConv((track_channels, tag_channels), hidden_channels, normalize=True, project=True),
            ("artist", "linked_to", "artist"): GATConv((artist_channels, artist_channels), hidden_channels, heads=3, concat=False),
            ("artist", "musically_related_to", "artist"): GATConv((artist_channels, artist_channels), hidden_channels, heads=3, concat=False),
            ("artist", "personally_related_to", "artist"): GATConv((artist_channels, artist_channels), hidden_channels, heads=3, concat=False),
            ("tag", "tags_artists", "artist"): SAGEConv((tag_channels, artist_channels), hidden_channels, normalize=True, project=True),
            ("tag", "tags_tracks", "track"): SAGEConv((tag_channels, track_channels), hidden_channels, normalize=True, project=True),
            ("track", "worked_by", "artist"): SAGEConv((track_channels, artist_channels), hidden_channels, normalize=True, project=True),
            ("artist", "worked_in", "track"): SAGEConv((artist_channels, track_channels), hidden_channels, normalize=True, project=True),
        }, aggr="mean")

        self.conv2 = HeteroConv({
            ("artist", "collab_with", "artist"): GATConv((hidden_channels, hidden_channels), hidden_channels, heads=3, concat=False),
            ("artist", "has_tag_artists", "tag"): SAGEConv((hidden_channels, hidden_channels), hidden_channels, normalize=True, project=True),
            ("artist", "last_fm_match", "artist"): GATv2Conv((hidden_channels, hidden_channels), hidden_channels, heads=3, concat=False, edge_dim=1),
            ("artist", "follows", "artist"): GATv2Conv((hidden_channels, hidden_channels), hidden_channels, heads=3, concat=False, edge_dim=1),
            ("track", "has_tag_tracks", "tag"): SAGEConv((hidden_channels, hidden_channels), hidden_channels, normalize=True, project=True),
            ("artist", "linked_to", "artist"): GATConv((hidden_channels, hidden_channels), hidden_channels, heads=3, concat=False),
            ("artist", "musically_related_to", "artist"): GATConv((hidden_channels, hidden_channels), hidden_channels, heads=3, concat=False),
            ("artist", "personally_related_to", "artist"): GATConv((hidden_channels, hidden_channels), hidden_channels, heads=3, concat=False),
            ("tag", "tags_artists", "artist"): SAGEConv((hidden_channels, hidden_channels), hidden_channels, normalize=True, project=True),
            ("tag", "tags_tracks", "track"): SAGEConv((hidden_channels, hidden_channels), hidden_channels, normalize=True, project=True),
            ("track", "worked_by", "artist"): SAGEConv((hidden_channels, hidden_channels), hidden_channels, normalize=True, project=True),
            ("artist", "worked_in", "track"): SAGEConv((hidden_channels, hidden_channels), hidden_channels, normalize=True, project=True),
        }, aggr="mean")

        self.linear1 = Linear(hidden_channels, hidden_channels * 4)
        self.linear2 = Linear(hidden_channels * 4, out_channels)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        x_dict1 = self.conv1(x_dict, edge_index_dict, edge_attr_dict)
        x_dict2 = self.conv2(x_dict1, edge_index_dict, edge_attr_dict)

        x_artist = self.linear1(x_dict2['artist'])
        x_artist = F.relu(x_artist)
        x_artist = self.linear2(x_artist)

        x_artist = F.normalize(x_artist, p=2, dim=-1)
        x_dict['artist'] = x_artist

        return x_dict


def test(model, loader, criterion, device, train_edges_set, model_name):
    model.eval()
    all_labels = []
    all_probs = []
    times = list()
    test_loss = 0.0
    num_batches = 0
    valid_batches = 0

    with torch.no_grad(), open(f"{model_name}_timings.data", "wb") as out_file:
        for batch in tqdm.tqdm(loader, desc="Testing"):
            num_batches += 1
            sampled_data = batch.to(device)

            custom_edge_attr = {}
            if ("artist", "last_fm_match", "artist") in sampled_data.edge_attr_dict:
                custom_edge_attr[("artist", "last_fm_match", "artist")] = sampled_data.edge_attr_dict[("artist", "last_fm_match", "artist")]
            if ("artist", "follows", "artist") in sampled_data.edge_attr_dict:
                custom_edge_attr[("artist", "follows", "artist")] = sampled_data.edge_attr_dict[("artist", "follows", "artist")]

            # Forward pass with timing
            now = time.time()
            pred_dict = model(sampled_data.x_dict, sampled_data.edge_index_dict, custom_edge_attr)
            new_time = time.time() - now
            times.append(new_time * 1000)

            edge_label_index = sampled_data['artist', 'collab_with', 'artist'].edge_label_index
            edge_label = sampled_data['artist', 'collab_with', 'artist'].edge_label

            # Filter edge list
            filtered_edges = []
            filtered_edge_label = []
            positive_count = 0

            for src, dst, label in zip(edge_label_index[0, :], edge_label_index[1, :], edge_label):
                lookup_edge = (
                    sampled_data['artist'].n_id[src].item(),
                    sampled_data['artist'].n_id[dst].item()
                )
                if lookup_edge in train_edges_set:
                    continue
                label_item = label.item()

                # Balancing logic
                if np.isclose(label_item, 1):
                    filtered_edge_label.append(label_item)
                    filtered_edges.append([src.item(), dst.item()])
                    positive_count += 1
                elif positive_count > 0:
                    filtered_edge_label.append(label_item)
                    filtered_edges.append([src.item(), dst.item()])
                    positive_count -= 1
                else:
                    break

            if len(filtered_edges) == 0:
                continue

            valid_batches += 1

            filtered_edges_tensor = torch.tensor(filtered_edges, dtype=torch.long).t().to(device)
            filtered_labels_tensor = torch.tensor(filtered_edge_label).long().to(device)

            src_emb = pred_dict['artist'][filtered_edges_tensor[0]]
            dst_emb = pred_dict['artist'][filtered_edges_tensor[1]]

            preds = (src_emb * dst_emb).sum(dim=-1)

            loss = criterion(preds, filtered_labels_tensor.float())
            test_loss += loss.item()

            probs = torch.sigmoid(preds)
            all_labels.append(filtered_labels_tensor.cpu())
            all_probs.append(probs.cpu())

        pickle.dump(times, out_file)

    if all_labels:
        all_labels = torch.cat(all_labels)
        all_probs = torch.cat(all_probs)
    else:
        all_labels = torch.tensor([])
        all_probs = torch.tensor([])

    return test_loss / valid_batches if valid_batches > 0 else 0, all_labels, all_probs


def calculate_metrics(labels, probs, threshold):
    if labels.numel() == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0
    preds = (probs > threshold).long()
    cm = confusion_matrix(labels, preds)
    tp = cm[1, 1]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tn = cm[0, 0]
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    roc_auc = roc_auc_score(labels, probs)
    return accuracy, precision, recall, f1, roc_auc, tp, fp, fn, tn


def get_params_from_csv(model_name, year, perc, filepath="tiktok_pyg/results_z.csv"):
    try:
        with open(filepath, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if (row["model"] == model_name and
                        int(row["year"]) == year and
                        float(row["perc"]) == perc and
                        row["done"].strip().lower() == "true"):
                    print(f"Found matching 'done' entry in {filepath}.")
                    return float(row['best_threshold']), float(row['mean_follows']), float(row['std_follows'])
    except FileNotFoundError:
        print(f"Warning: Results file not found at {filepath}")
    except Exception as e:
        print(f"Warning: Could not read parameters from CSV: {e}")
    return None, None, None


def main():
    parser = argparse.ArgumentParser(description="GNN Testing for TikTok Heterodata (Standarized)")

    # Logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    data_folder = "tiktok_pyg/ds/"

    # Dataset params
    parser.add_argument("--year", type=int, default=2021, help="Dataset year")
    parser.add_argument("--perc", type=float, default=0.5, help="Dataset percentage")

    args = parser.parse_args()

    year = args.year
    month = 11
    perc = args.perc if args.perc > 0.0 else 0

    model_name = "nocat_tiktok_z"
    test_hd_filename = f"full_hd_{perc}_tiktok.pt"

    # We also need the training edges to filter them out during testing
    train_collab_with_filename = f"collab_with_{year}_{month}_{perc}_tiktok.pt"

    # Model path
    model_path = f"tiktok_pyg/trained_models/model_{model_name}_{year}_{month}_{perc}.pth"

    print(f"Config: Year={year}, Month={month}, Perc={perc}")
    print(f"Model Path: {model_path}")
    print(f"Test Data: {path.join(data_folder, test_hd_filename)}")

    if not path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return

    if not path.exists(path.join(data_folder, test_hd_filename)):
        print(f"Error: Test data file not found: {path.join(data_folder, test_hd_filename)}")
        return

    if not path.exists(path.join(data_folder, train_collab_with_filename)):
        print(f"Error: Train collab file not found: {path.join(data_folder, train_collab_with_filename)}")
        return

    # Load Params from CSV
    best_threshold, follows_mean, follows_std = get_params_from_csv(model_name, year, perc)

    if best_threshold is None:
        print("Error: Could not find training parameters (threshold, mean, std) in tiktok_pyg/results_z.csv")
        return

    print(f"Loaded params: Threshold={best_threshold}, Mean={follows_mean}, Std={follows_std}")

    # Load Data
    print("Loading test data...")
    data = torch.load(path.join(data_folder, test_hd_filename), weights_only=False)
    data.contiguous()

    # Apply Standardization to Follows
    if ("artist", "follows", "artist") in data.edge_attr_dict:
        edge_attr = data["artist", "follows", "artist"].edge_attr
        print("Applying standardization to test data 'follows' attributes...")
        data["artist", "follows", "artist"].edge_attr = (edge_attr - follows_mean) / follows_std
    else:
        print("Warning: 'follows' relationship not found in test data.")
        return

    # Load Train Edges
    print("Loading train edges...")
    train_collab_with = torch.load(path.join(data_folder, train_collab_with_filename))
    train_edges_set = set(map(tuple, train_collab_with.t().tolist()))

    # Loader config
    compt_tree_size = [25, 20]
    batch_size = 128

    print("Initializing Test Loader...")
    test_loader = LinkNeighborLoader(
        data=data,
        num_neighbors=compt_tree_size,
        neg_sampling_ratio=1,
        edge_label_index=("artist", "collab_with", "artist"),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        persistent_workers=True
    )

    # Model Params
    hidden_channels = 64
    out_channels = 64

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: '{device}'")

    artist_channels = data["artist"].x.size(1)
    track_channels = data["track"].x.size(1)
    tag_channels = data["tag"].x.size(1)
    metadata = data.metadata()

    print("Initializing Model...")
    model = GNN_NOCAT(
        metadata=metadata,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        artist_channels=artist_channels,
        track_channels=track_channels,
        tag_channels=tag_channels
    ).to(device)

    print("Loading model state...")
    model.load_state_dict(torch.load(model_path, weights_only=False))

    criterion = F.binary_cross_entropy_with_logits

    print("Starting Test...")
    test_loss, all_labels, all_probs = test(model, test_loader, criterion, device, train_edges_set, model_name)

    accuracy, precision, recall, f1, roc_auc, tp, fp, fn, tn = calculate_metrics(all_labels, all_probs, best_threshold)

    print(f"Test Metrics:")
    print(f"Loss:        {test_loss:.4f}")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"F1-score:    {f1:.4f}")
    print(f"ROC-AUC:     {roc_auc:.4f}")
    print(f"Confusion Matrix:\n{tp} {fn}\n{fp} {tn}")

if __name__ == '__main__':
    main()
