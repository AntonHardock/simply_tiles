# About
Create static vector tile caches using Python to instruct PostGIS.
Developed with Python 3.7 and PostGIS 3.0.

# Configuration
Configure tile generation (including custom Tile Matrix Sets)
using one compact json file (see `./config_examples`).

Example Tile Matrix Set Definitions are included in `./tms_definitions`
If you are interested, check out my TMS Tutorial included as jupyter Notebook.
(Currently only available in German)

EXPERIMENTAL FEATURE:
Geoms will be automatically reprojected to the SRID defined in the TMS. 
Not tested thoroughly, yet. In case of Bugs, reproject the geoms yourself before using this tool.

LIMITATION: 
As of now, only identical attribute name lists are supported for
multiple layer caches. This is due to the current handling of sql querries and will change in future versions.

# Usage
Using a very basic CLI:

```
python ./simply_tiles/main.py "config_examples/tilecache_config.json" "example_cache_name" "path/to/my/output_folder" --max_zoom 15
```

# Docker
The package also includes an unfinished Dockerfile soon to be completed.