# About
Create static vector tile caches using Python to instruct PostGIS.
Written in Python 3.7 with psycopg2 as the only dependency.

# Configuration
Configure tile generation (including custom Tile Matrix Sets)
using one compact json file (see `./config_examples`).

Example Tile Matrix Set Definitions are included in `./tms_definitions`

# Usage
Using a very basic CLI:

'''
python ./simply_tiles/main.py "config_examples/tilecache_config.json" "example_cache_name" "path/to/my/output_folder" --max_zoom 15
'''

# Docker
The package also includes an unfinished Dockerfile soon to be completed.