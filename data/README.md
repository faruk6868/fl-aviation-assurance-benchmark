# Data directory

Public data only. Place the NASA C-MAPSS turbofan dataset here after downloading.

Use the helper:
```
python scripts/download_cmapss.py --source-url <public_zip_url> --target-dir data/c-mapss
```
The script will unpack the archive into `data/c-mapss/` and keep only the text files (train/test/RUL). The repository does not commit raw data.
