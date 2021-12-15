import argparse
from vector_tiler import VectorTiler

# prepare argument parser
parser = argparse.ArgumentParser()
parser.add_argument('config_path', type=str, help="Positional Argument. Configuration of vector tile creation")
parser.add_argument('tileset_name', type=str, help="Positional Argument. Name of the cache")
parser.add_argument('tileset_path', type=str, help="Positional Argument. Output path for tile cache")
parser.add_argument('--max_zoom', type=int, help="Optional Argument. Maximum zoomlevel to be cached.")

# parse arguments
args = parser.parse_args()

# instantiate VectorTiler object
tiler = VectorTiler(args.config_path)

# generate tilecache
tiler.generate_tileset(
    args.tileset_path, 
    args.tileset_name, 
    max_zoom=args.max_zoom
    )