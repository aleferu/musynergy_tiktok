# TikTok's improvements for MUSYNERGY

This repository only contains the needed code to improve MUSYNERGY with TikTok's data.

MUSYNERGY repo: https://github.com/aleferu/musicbrainz

## .env file

A .env file is needed in order to configure the database connection. An example .env file is provided. Modify its contents to satisfy your needs.

## Directories

- `tiktok` extracts data from TikTok's Research API and loads it into Neo4j.
- `tiktok_pyg` trains the TikTok models with all the artists with an associated TikTok account.
- `intiktok_pyg` trains the TikTok models with only the artists with a successful data extraction from TikTok.
- `post_tiktok` has some scripts to compute and graph some stats related to Tiktok.

## Dates

- TikTok data: `2025-10-13` to `2025-11-25`.

## Dataset

`schema.md` is dedicated to explain our dataset. With these two files you can learn about the root graph schema and check out some graphs and statistics.
