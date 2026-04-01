#!/usr/bin/env python3


import torch
from torch_geometric.data import HeteroData
import os.path as path
import pandas as pd
import logging
import pickle
import numpy as np


def get_full_data() -> HeteroData:
    logging.info("Reading the whole dataset")

    data = HeteroData()

    data["artist"].x = torch.load(path.join(data_folder, "artists.pt"), weights_only=True)
    logging.info(f"  Artist tensor shape: {data['artist'].x.shape}")

    data["track"].x = torch.load(path.join(data_folder, "tracks.pt"), weights_only=True)
    logging.info(f"  Track tensor shape: {data['track'].x.shape}")

    data["tag"].x = torch.load(path.join(data_folder, "tags.pt"), weights_only=True)
    logging.info(f"  Tag tensor shape: {data['tag'].x.shape}")


    data["artist", "collab_with", "artist"].edge_index = torch.load(path.join(data_folder, "collab_with.pt"), weights_only=True)
    data["artist", "collab_with", "artist"].edge_attr = torch.load(path.join(data_folder, "collab_with_attr.pt"), weights_only=True)
    logging.info(f"  collab_with index tensor shape: {data['artist', 'collab_with', 'artist'].edge_index.shape}")
    logging.info(f"  collab_with attr tensor shape: {data['artist', 'collab_with', 'artist'].edge_attr.shape}")

    data["artist", "has_tag_artists", "tag"].edge_index = torch.load(path.join(data_folder, "has_tag_artists.pt"), weights_only=True)
    data["track", "has_tag_tracks", "tag"].edge_index = torch.load(path.join(data_folder, "has_tag_tracks.pt"), weights_only=True)
    logging.info(f"  has_tag_artists index tensor shape: {data['artist', 'has_tag_artists', 'tag'].edge_index.shape}")
    logging.info(f"  has_tag_tracks index tensor shape: {data['track', 'has_tag_tracks', 'tag'].edge_index.shape}")

    data["artist", "last_fm_match", "artist"].edge_index = torch.load(path.join(data_folder, "last_fm_match.pt"), weights_only=True)
    data["artist", "last_fm_match", "artist"].edge_attr = torch.load(path.join(data_folder, "last_fm_match_attr.pt"), weights_only=True)
    logging.info(f"  last_fm_match index tensor shape: {data['artist', 'last_fm_match', 'artist'].edge_index.shape}")
    logging.info(f"  last_fm_match attr tensor shape: {data['artist', 'last_fm_match', 'artist'].edge_attr.shape}")

    data["artist", "linked_to", "artist"].edge_index = torch.load(path.join(data_folder, "linked_to.pt"), weights_only=True)
    data["artist", "linked_to", "artist"].edge_attr = torch.load(path.join(data_folder, "linked_to_attr.pt"), weights_only=True)
    logging.info(f"  linked_to index tensor shape: {data['artist', 'linked_to', 'artist'].edge_index.shape}")
    logging.info(f"  linked_to attr tensor shape: {data['artist', 'linked_to', 'artist'].edge_attr.shape}")

    data["artist", "musically_related_to", "artist"].edge_index = torch.load(path.join(data_folder, "musically_related_to.pt"), weights_only=True)
    data["artist", "musically_related_to", "artist"].edge_attr = torch.load(path.join(data_folder, "musically_related_to_attr.pt"), weights_only=True)
    logging.info(f"  musically_related_to index tensor shape: {data['artist', 'musically_related_to', 'artist'].edge_index.shape}")
    logging.info(f"  musically_related_to attr tensor shape: {data['artist', 'musically_related_to', 'artist'].edge_attr.shape}")

    data["artist", "personally_related_to", "artist"].edge_index = torch.load(path.join(data_folder, "personally_related_to.pt"), weights_only=True)
    data["artist", "personally_related_to", "artist"].edge_attr = torch.load(path.join(data_folder, "personally_related_to_attr.pt"), weights_only=True)
    logging.info(f"  personally_related_to index tensor shape: {data['artist', 'personally_related_to', 'artist'].edge_index.shape}")
    logging.info(f"  personally_related_to attr tensor shape: {data['artist', 'personally_related_to', 'artist'].edge_attr.shape}")

    # New 'follows' relationship for TikTok
    data["artist", "follows", "artist"].edge_index = torch.load(path.join(data_folder, "follows.pt"), weights_only=True)
    data["artist", "follows", "artist"].edge_attr = torch.load(path.join(data_folder, "follows_attr.pt"), weights_only=True)
    logging.info(f"  follows index tensor shape: {data['artist', 'follows', 'artist'].edge_index.shape}")
    logging.info(f"  follows attr tensor shape: {data['artist', 'follows', 'artist'].edge_attr.shape}")

    data["tag", "tags_artists", "artist"].edge_index = torch.load(path.join(data_folder, "tags_artists.pt"), weights_only=True)
    data["tag", "tags_track", "track"].edge_index = torch.load(path.join(data_folder, "tags_tracks.pt"), weights_only=True)
    logging.info(f"  tags_artists index tensor shape: {data['tag', 'tags_artists', 'artist'].edge_index.shape}")
    logging.info(f"  tags_tracks index tensor shape: {data['tag', 'tags_track', 'track'].edge_index.shape}")

    data["track", "worked_by", "artist"].edge_index = torch.load(path.join(data_folder, "worked_by.pt"), weights_only=True)
    data["artist", "worked_in", "track"].edge_index = torch.load(path.join(data_folder, "worked_in.pt"), weights_only=True)
    logging.info(f"  worked_by index tensor shape: {data['track', 'worked_by', 'artist'].edge_index.shape}")
    logging.info(f"  worked_in index tensor shape: {data['artist', 'worked_in', "track"].edge_index.shape}")

    if data.validate():
        logging.info("  Full data validation successful!")

    return data


def isolate_artists(data: HeteroData, artists_to_keep: torch.Tensor):
    edge_types = [
        ("artist", "collab_with", "artist"),
        ("artist", "has_tag_artists", "tag"),
        ("track", "has_tag_tracks", "tag"),
        ("artist", "last_fm_match", "artist"),
        ("artist", "linked_to", "artist"),
        ("artist", "musically_related_to", "artist"),
        ("artist", "personally_related_to", "artist"),
        ("artist", "follows", "artist"),  # Added follows
        ("tag", "tags_artists", "artist"),
        ("tag", "tags_track", "track"),
        ("track", "worked_by", "artist"),
        ("artist", "worked_in", "track")
    ]

    for edge_type in edge_types:
        edge_index = data[edge_type].edge_index
        mask = torch.ones(edge_index.shape[1], dtype=torch.bool)
        if edge_type[0] == "artist":
            mask &= torch.isin(edge_index[0], artists_to_keep)
        if edge_type[2] == "artist":
            mask &= torch.isin(edge_index[1], artists_to_keep)

        filtered_edge_index = edge_index[:, mask]
        data[edge_type].edge_index = filtered_edge_index

        if hasattr(data[edge_type], "edge_attr"):
            data[edge_type].edge_attr = data[edge_type].edge_attr[mask]


def clean_data(data: HeteroData):
    logging.info(f"Cleaning data per percentile {percentile}")

    # Data
    artist_popularity = data["artist"].x[:, 10]

    # Threshold obtention
    threshold = torch.quantile(artist_popularity, percentile)
    selected_artists = artist_popularity >= threshold
    selected_artist_ids = torch.nonzero(selected_artists).squeeze()

    logging.info("  Isolating artists")
    isolate_artists(data, selected_artist_ids)

    logging.info("  Extracting subgraph")
    track_mask = torch.zeros(data["track"].x.shape[0], dtype=torch.bool)
    track_mask[
        torch.unique(data["artist", "worked_in", "track"].edge_index[1, :])
    ] = True
    logging.info("  Found %d tracks?", track_mask.sum())
    data = data.subgraph({
        "track": track_mask
    })

    if data.validate():
        logging.info("  Data validation after percentile cleanup successful!")
        logging.info("  Number of tracks: %d", data["track"].x.shape[0])
        logging.info("  Number of collabs: %d", data["artist", "collab_with", "artist"].edge_index.shape[1])


def cut_at_date(data: HeteroData) -> HeteroData:
    logging.info(f"Cutting at year {cut_year} and month {cut_month}")

    logging.info("  Reading CSV")
    df = pd.read_csv("data/year_month_track.csv")
    df["track_ids"] = df["track_ids"].apply(eval)

    logging.info("  Setting up some vars")

    # Tracks involved
    mask = (df["year"] < cut_year) | ((df["year"] == cut_year) & (df["month"] < cut_month))
    train_tracks_neo4j = df[mask]["track_ids"].explode().unique().tolist()  # type: ignore

    # Track map
    with open(path.join(data_folder, "track_map.pkl"), "rb") as in_file:
        track_map = pickle.load(in_file)

    # Masks definition
    # Filter only tracks present in our track_map (since we are on a subset of data)
    train_tracks_pyg_t = torch.tensor([
        track_map[track_id]
        for track_id in train_tracks_neo4j
        if track_id in track_map
    ])
    track_mask = torch.zeros(data["track"].x.size(0), dtype=torch.bool)
    track_mask[train_tracks_pyg_t] = True

    train_artists_pyg = data["artist", "worked_in", "track"].edge_index[0, :][
        torch.isin(data["artist", "worked_in", "track"].edge_index[1, :], train_tracks_pyg_t)
    ]

    # Subgraph definition
    train_data = data.subgraph({
        "track": track_mask
    })

    # Isolate artists
    logging.info("  Isolating artists")
    isolate_artists(train_data, train_artists_pyg)

    # Initial edges to consider
    collab_with_edge_index = train_data["artist", "collab_with", "artist"].edge_index[:, ::2]
    worked_in_edge_index = train_data["artist", "worked_in", "track"].edge_index

    # Unique artists that have collabed
    unique_artists = torch.unique(torch.cat((collab_with_edge_index[0, :], collab_with_edge_index[1, :])))

    # Filtering of worked_in
    mask = torch.isin(
        worked_in_edge_index[0, :],
        unique_artists

    )
    filtered_worked_in_edge_index = worked_in_edge_index[:, mask]

    # Find the indices where each artist starts
    artist_id_sorted = filtered_worked_in_edge_index[0, :]

    change_indices = torch.cat((
        torch.tensor([0]),  # Start from index 0
        torch.where(artist_id_sorted[1:] != artist_id_sorted[:-1])[0] + 1
    ))

    # Get artist IDs at those change points
    artists_at_change_points = artist_id_sorted[change_indices]

    # Create dictionary mapping artist → their track indices
    track_ids = filtered_worked_in_edge_index[1, :]
    artist_tracks_dict = {
        artist.item(): track_ids[start:end]
        for artist, start, end in zip(artists_at_change_points, change_indices, torch.cat((change_indices[1:], torch.tensor([track_ids.shape[0]]))))
    }

    # Collect the new collaboration edges
    logging.info("  Building the new collab_with tensors")
    new_collab_with_edge_index = list()
    new_collab_with_edge_attr = list()
    for a0, a1 in zip(collab_with_edge_index[0, :], collab_with_edge_index[1, :]):
        a0_item = a0.item()
        a1_item = a1.item()
        intersection_len = len(np.intersect1d(artist_tracks_dict[a0_item], artist_tracks_dict[a1_item]))
        if intersection_len > 0:
            new_collab_with_edge_index.append((a0_item, a1_item))
            new_collab_with_edge_index.append((a1_item, a0_item))
            new_collab_with_edge_attr.extend([intersection_len, intersection_len])

    train_data["artist", "collab_with", "artist"].edge_index = torch.tensor(new_collab_with_edge_index).t()
    train_data["artist", "collab_with", "artist"].edge_attr = torch.tensor(new_collab_with_edge_attr).t()

    if train_data.validate():
        logging.info("  Data validation after date cut successful!")
        logging.info("  Number of tracks: %d", train_data["track"].x.shape[0])
        logging.info("  Number of collabs: %d", train_data["artist", "collab_with", "artist"].edge_index.shape[1])

    return train_data


def main():
    full_data = get_full_data()

    if percentile > 0:
        clean_data(full_data)

    logging.info("Saving full graph...")
    full_path = path.join(data_folder, f"full_hd_{percentile}_intiktok.pt")
    torch.save(full_data, full_path)

    if cut_year is not None:
        result = cut_at_date(full_data)

        logging.info("Saving training graph...")

        result_path = path.join(data_folder, f"train_hd_{cut_year}_{cut_month}_{percentile}_intiktok.pt")
        collab_with_path = path.join(data_folder, f"collab_with_{cut_year}_{cut_month}_{percentile}_intiktok.pt")
        torch.save(result, result_path)
        torch.save(result["artist", "collab_with", "artist"].edge_index, collab_with_path)

        logging.info("Done!")


if __name__ == "__main__":
    # Logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    for percentile in [0, 0.5, 0.75, 0.9]:
        for cut_year in [2019, 2021, 2023]:

            cut_month = 11

            logging.info("P: %f", percentile)
            logging.info("Y: %d", cut_year)
            logging.info("M: %d", cut_month)

            assert 0 <= percentile <= 1 and (
                (cut_year is None and cut_month is None)
                or
                (cut_year is not None and cut_month is not None)
            )

            data_folder = "intiktok_pyg/ds/"
            main()
