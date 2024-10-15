From root dir, run 

`python3 scripts/establish_part_lookup.py`
`python3 scripts/load_part_select_info.py`

To initialize and populate Chroma vector database. 
If encountering HTTP 403 errors when running load_part_select_info, ensure specified headers match your local machine's.

To start Flask server, run

`python3 flask_app.py`