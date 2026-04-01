#!/usr/bin/env python3


import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import HeteroConv, GATConv, SAGEConv, GATv2Conv
from torch_geometric.loader import LinkNeighborLoader
from torch.utils.data import SubsetRandomSampler
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
import tqdm
import copy
import os
import os.path as path
import csv
import logging
import argparse
import glob


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

        x_artist = self.linear1(x_dict2["artist"])
        x_artist = F.relu(x_artist)
        x_artist = self.linear2(x_artist)

        x_artist = F.normalize(x_artist, p=2, dim=-1)
        x_dict["artist"] = x_artist

        return x_dict


class GNN_NOCAT_NOWEIGHT(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels, artist_channels, track_channels, tag_channels):
        super().__init__()
        self.metadata = metadata
        self.out_channels = out_channels

        self.conv1 = HeteroConv({
            ("artist", "collab_with", "artist"): GATConv((artist_channels, artist_channels), hidden_channels, heads=3, concat=False),
            ("artist", "has_tag_artists", "tag"): SAGEConv((artist_channels, tag_channels), hidden_channels, normalize=True, project=True),
            ("artist", "last_fm_match", "artist"): GATv2Conv((artist_channels, artist_channels), hidden_channels, heads=3, concat=False, edge_dim=1),
            ("artist", "follows", "artist"): GATv2Conv((artist_channels, artist_channels), hidden_channels, heads=3, concat=False),
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
            ("artist", "follows", "artist"): GATv2Conv((hidden_channels, hidden_channels), hidden_channels, heads=3, concat=False),
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
        filtered_edge_attr = dict()
        target_relation = ("artist", "last_fm_match", "artist")

        if target_relation in edge_attr_dict:
            filtered_edge_attr[target_relation] = edge_attr_dict[target_relation]

        x_dict1 = self.conv1(x_dict, edge_index_dict, filtered_edge_attr)
        x_dict2 = self.conv2(x_dict1, edge_index_dict, filtered_edge_attr)

        x_artist = self.linear1(x_dict2["artist"])
        x_artist = F.relu(x_artist)
        x_artist = self.linear2(x_artist)

        x_artist = F.normalize(x_artist, p=2, dim=-1)
        x_dict["artist"] = x_artist

        return x_dict


class GNN_NOCAT_OG(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels, artist_channels, track_channels, tag_channels):
        super().__init__()
        self.metadata = metadata
        self.out_channels = out_channels

        self.conv1 = HeteroConv({
            ("artist", "collab_with", "artist"): GATConv((artist_channels, artist_channels), hidden_channels, heads=3, concat=False),
            ("artist", "has_tag_artists", "tag"): SAGEConv((artist_channels, tag_channels), hidden_channels, normalize=True, project=True),
            ("artist", "last_fm_match", "artist"): GATv2Conv((artist_channels, artist_channels), hidden_channels, heads=3, concat=False, edge_dim=1),
            # ("artist", "follows", "artist"): GATv2Conv((artist_channels, artist_channels), hidden_channels, heads=3, concat=False, edge_dim=1),
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
            # ("artist", "follows", "artist"): GATv2Conv((hidden_channels, hidden_channels), hidden_channels, heads=3, concat=False, edge_dim=1),
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

        x_artist = self.linear1(x_dict2["artist"])
        x_artist = F.relu(x_artist)
        x_artist = self.linear2(x_artist)

        x_artist = F.normalize(x_artist, p=2, dim=-1)
        x_dict["artist"] = x_artist

        return x_dict


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0.0
    num_batches = 0

    # We iterate directly over the loader
    for batch in tqdm.tqdm(loader, desc="Training"):
        sampled_data = batch.to(device)
        optimizer.zero_grad()

        # Create a new edge_attr_dict containing 'last_fm_match' and 'follows' attributes
        # We need to explicitly pass these to the model because we used edge_dim=1
        custom_edge_attr = {}
        if ("artist", "last_fm_match", "artist") in sampled_data.edge_attr_dict:
            custom_edge_attr[("artist", "last_fm_match", "artist")] = sampled_data.edge_attr_dict[("artist", "last_fm_match", "artist")]
        if ("artist", "follows", "artist") in sampled_data.edge_attr_dict:
            custom_edge_attr[("artist", "follows", "artist")] = sampled_data.edge_attr_dict[("artist", "follows", "artist")]

        # Forward pass
        pred_dict = model(sampled_data.x_dict, sampled_data.edge_index_dict, custom_edge_attr)

        edge_label_index = sampled_data['artist', 'collab_with', 'artist'].edge_label_index
        edge_label = sampled_data['artist', 'collab_with', 'artist'].edge_label

        src_emb = pred_dict["artist"][edge_label_index[0]]
        dst_emb = pred_dict["artist"][edge_label_index[1]]

        preds = (src_emb * dst_emb).sum(dim=-1)
        loss = criterion(preds, edge_label.float())

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    return epoch_loss / num_batches if num_batches > 0 else 0


def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    all_labels = []
    all_probs = []
    val_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm.tqdm(loader, desc="Validation"):
            sampled_data = batch.to(device)

            # Create a new edge_attr_dict containing 'last_fm_match' and 'follows' attributes
            custom_edge_attr = {}
            if ("artist", "last_fm_match", "artist") in sampled_data.edge_attr_dict:
                custom_edge_attr[("artist", "last_fm_match", "artist")] = sampled_data.edge_attr_dict[("artist", "last_fm_match", "artist")]
            if ("artist", "follows", "artist") in sampled_data.edge_attr_dict:
                custom_edge_attr[("artist", "follows", "artist")] = sampled_data.edge_attr_dict[("artist", "follows", "artist")]

            # Forward pass
            pred_dict = model(sampled_data.x_dict, sampled_data.edge_index_dict, custom_edge_attr)

            edge_label_index = sampled_data['artist', 'collab_with', 'artist'].edge_label_index
            edge_label = sampled_data['artist', 'collab_with', 'artist'].edge_label

            src_emb = pred_dict["artist"][edge_label_index[0]]
            dst_emb = pred_dict["artist"][edge_label_index[1]]

            preds = (src_emb * dst_emb).sum(dim=-1)
            loss = criterion(preds, edge_label.float())
            val_loss += loss.item()

            probs = torch.sigmoid(preds)
            all_labels.append(edge_label.cpu())
            all_probs.append(probs.cpu())
            num_batches += 1

    all_labels = torch.cat(all_labels) if all_labels else torch.empty(0, dtype=torch.long)
    all_probs = torch.cat(all_probs) if all_probs else torch.empty(0)
    return val_loss / num_batches if num_batches > 0 else 0, all_labels, all_probs


def find_best_threshold(labels, probs):
    best_threshold = 0
    best_f1 = 0
    if labels.numel() > 0:
        for threshold in tqdm.tqdm(np.arange(0.2, 0.91, 0.01), desc="Threshold Search"):
            preds_binary = (probs > threshold).long()
            cm = confusion_matrix(labels, preds_binary)
            tp = cm[1, 1]
            fp = cm[0, 1]
            fn = cm[1, 0]
            precision = 0 if tp == 0 else tp / (tp + fp)
            recall = 0 if tp == 0 else tp / (tp + fn)
            f1 = 0 if precision * recall == 0 else 2 * precision * recall / (precision + recall)
            if f1 > best_f1:
                best_threshold = threshold
                best_f1 = f1
    return best_threshold


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


def save_results(data, filepath="intiktok_pyg/results.csv"):
    header = ["model", "year", "month", "perc", "epoch", "train_loss", "val_loss", "acc", "prec", "rec", "f1", "auc", "tp", "fp", "fn", "tn", "best_threshold", "done"]
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([data.get(key, "") for key in header])


def main():
    parser = argparse.ArgumentParser(description="GNN Training for TikTok Heterodata")

    # Logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Config
    data_folder = "intiktok_pyg/ds/"
    # Ensure models dir exists
    models_dir = "intiktok_pyg/trained_models/"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # Dataset params
    parser.add_argument("--year", type=int, default=2023, help="Dataset year")
    parser.add_argument("--perc", type=float, default=0, help="Dataset percentage")
    parser.add_argument("--original", action="store_true", default=False, help="Use original dataset")

    args = parser.parse_args()

    year = args.year
    month = 11
    perc = args.perc if args.perc > 0.0 else 0

    # Filename matching build_train_heterodata_tiktok.py output
    train_hd_filename = f"train_hd_{year}_{month}_{perc}_intiktok.pt"

    print(f"Config: Year={year}, Month={month}, Perc={perc}")
    print(f"Loading data from {path.join(data_folder, train_hd_filename)}...")

    if not path.exists(path.join(data_folder, train_hd_filename)):
        print(f"Error: Data file not found: {path.join(data_folder, train_hd_filename)}")
        return

    # Load Data
    data = torch.load(path.join(data_folder, train_hd_filename), weights_only=False)
    if args.original:
        keep_indices = list(range(17)) + [21]  # see build_ds_tiktok
        data["artist"].x = data["artist"].x[:, keep_indices]
        print("REMOVED TIKTOK ATTRS FROM ARTISTS")
    data.contiguous()

    # Split
    edge_indices = torch.arange(data["artist", "collab_with", "artist"].edge_index.shape[1])
    num_edges = len(edge_indices)
    perm = torch.randperm(num_edges)
    split_idx = int(0.8 * num_edges)

    train_sampler = SubsetRandomSampler(perm[:split_idx])  # type: ignore
    val_sampler = SubsetRandomSampler(perm[split_idx:])  # type: ignore

    # Loader config
    compt_tree_size = [25, 20]
    batch_size = 128

    print("Initializing Loaders...")
    train_loader = LinkNeighborLoader(
        data=data,
        num_neighbors=compt_tree_size,
        neg_sampling_ratio=1,
        edge_label_index=("artist", "collab_with", "artist"),
        batch_size=batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=4,
        persistent_workers=True
    )

    val_loader = LinkNeighborLoader(
        data=data,
        num_neighbors=compt_tree_size,
        neg_sampling_ratio=1,
        edge_label_index=("artist", "collab_with", "artist"),
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=4,
        persistent_workers=True
    )

    # Training parameters
    model_name = "nocat_inog"
    latest_epoch = 0  # Starting from scratch or set this if loading checkpoint
    hidden_channels = 64
    out_channels = 64
    num_epochs = 1000
    patience = 5
    learning_rate = 1e-4
    weight_decay = 1e-5

    # Device
    device = torch.device("cuda")
    print(f"Device: '{device}'")

    # Get channel sizes from data
    artist_channels = data["artist"].x.size(1)
    track_channels = data["track"].x.size(1)
    tag_channels = data["tag"].x.size(1)
    metadata = data.metadata()

    # Initialize model
    print("Initializing Model...")
    model = GNN_NOCAT_OG(
        metadata=metadata,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        artist_channels=artist_channels,
        track_channels=track_channels,
        tag_channels=tag_channels
    ).to(device)

    # Checkpoint loading
    if latest_epoch > 0:
        checkpoint_path = f"intiktok_pyg/trained_models/model_{model_name}_{year}_{month}_{perc}_{latest_epoch}.pth"
        if path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, weights_only=False))
            print(f"Loaded epoch {latest_epoch}")
        else:
            print(f"Checkpoint not found at {checkpoint_path}")
            exit(1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # type: ignore
    criterion = F.binary_cross_entropy_with_logits

    best_val_f1 = 0.0
    best_threshold = 0
    epochs_no_improve = 0
    best_model_state = None
    train_losses = []
    val_losses = []
    best_epoch = 0

    print("Starting training loop...")

    for epoch in range(num_epochs):
        epoch_num = latest_epoch + epoch + 1

        epoch_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch_num}/{num_epochs + latest_epoch}, Training Loss: {epoch_loss:.4f}")

        val_loss, all_labels, all_probs = evaluate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        best_threshold_epoch = find_best_threshold(all_labels, all_probs)
        print(f"Best threshold for this epoch: {best_threshold_epoch}")

        accuracy, precision, recall, f1, roc_auc, tp, fp, fn, tn = calculate_metrics(all_labels, all_probs, best_threshold_epoch)

        print(f"Validation Metrics - Epoch {epoch_num}:")
        print(f"Loss:        {val_loss:.4f}")
        print(f"Accuracy:    {accuracy:.4f}")
        print(f"Precision:   {precision:.4f}")
        print(f"Recall:      {recall:.4f}")
        print(f"F1-score:    {f1:.4f}")
        print(f"ROC-AUC:     {roc_auc:.4f}")
        print(f"Confusion Matrix:\n{tp} {fn}\n{fp} {tn}")

        result_row = {
            "model": model_name,
            "year": year,
            "month": month,
            "perc": perc,
            "epoch": epoch_num,
            "train_loss": epoch_loss,
            "val_loss": val_loss,
            "acc": accuracy,
            "prec": precision,
            "rec": recall,
            "f1": f1,
            "auc": roc_auc,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "best_threshold": best_threshold_epoch,
            "done": False
        }

        save_results(result_row)

        # Save checkpoint
        torch.save(model.state_dict(), f"{models_dir}model_{model_name}_{year}_{month}_{perc}_{epoch_num}.pth")

        if f1 > best_val_f1:
            best_val_f1 = f1
            best_threshold = best_threshold_epoch
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch_num
        else:
            epochs_no_improve += 1
            print("Epochs without improving:", epochs_no_improve)
            if epochs_no_improve == patience:
                print(f"Early stopping!!!")
                print(f"Best epoch: {best_epoch}")
                # Restore best model
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break

    print("Training complete.")
    print("Best epoch:", best_epoch)
    print("Best validation F1-score:", best_val_f1)
    print("Best threshold:", best_threshold)

    # Remove the older models
    checkpoint_pattern = f"{models_dir}model_{model_name}_{year}_{month}_{perc}_*.pth"
    for checkpoint_file in glob.glob(checkpoint_pattern):
        try:
            os.remove(checkpoint_file)
            print(f"Deleted old checkpoint: {checkpoint_file}")
        except OSError as e:
            print(f"Error deleting {checkpoint_file}: {e}")

    # Save final best model
    torch.save(model.state_dict(), f"{models_dir}model_{model_name}_{year}_{month}_{perc}.pth")


if __name__ == '__main__':
    main()
