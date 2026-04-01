# tiktok_pyg

Dataset creation and training of the models using all artists with an associated TikTok account.

- `build_ds_tiktok.py`: Builds the tensors and mappings for artists with `tiktok_call: true`.
- `build_train_heterodata_tiktok.py`: Builds the HeteroData objects for PyG.
- `train.py`: Trains the models.
- `train_z.py`: Trains the Z models, which use standardization (Z-score normalization) of the TikTok `FOLLOWS` relationship weights.
- `test.py`: Testing script for the standard models.
- `test_z.py`: Testing script for the Z models.
- `train.sh` / `train_z.sh` / `test.sh` / `test_z.sh`: Shell scripts to automate training and testing across different years and percentages.
