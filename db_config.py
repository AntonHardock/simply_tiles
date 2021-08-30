# postgres database to connect to

DATABASE = {
    'user':     'postgres',
    'password': '###########',
    'host':     'localhost',
    'port':     '5432',
    'database': 'postgres'
    }

# Table to query for MVT data, and columns to
# include in the tiles.
TABLE = {
    'table':       'verwaltungseinheit',
    'geom_column':  'geom',
    'attr_columns': 'gid, name'
    }  

# HTTP server information
HOST = 'localhost'
PORT = 8080

