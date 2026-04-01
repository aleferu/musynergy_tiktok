# intiktok_pyg

Dataset creation and training of the models using only the artists with a successful data extraction from TikTok.

- `build_ds_tiktok.py`: Builds the tensors and mappings for artists with `in_tiktok: true`.
- `build_train_heterodata_tiktok.py`: Builds the HeteroData objects for PyG.
- `train.py`: Trains the models.
- `train_z.py`: Trains the Z models, which use standardization (Z-score normalization) of the TikTok `FOLLOWS` relationship weights.
- `test.py`: Testing script for the standard models.
- `test_z.py`: Testing script for the Z models.
- `train.sh` / `train_z.sh` / `test.sh` / `test_z.sh`: Shell scripts to automate training and testing across different years and percentages.
- `test_results.py`: Does what `test.py` does but also extracts the pairs of artist IDs when a TP is found.
- `improve_examples.py`: Takes the CSV exported by `test_results.py` and adds additional information, such as most popular track name and tags.
