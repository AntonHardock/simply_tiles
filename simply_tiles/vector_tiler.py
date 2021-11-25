import os
import psycopg2
import json
from pathlib import Path
from tqdm import tqdm 

from simply_tiles.tms import TileMatrixSet


class VectorTiler:

    def __init__(self, config:str):
        
        # parse config 
        with open(config, mode="r") as json_file:
            configs = json.load(json_file)

        # parse config parameters
        with open(configs["TMS_DEFINITION"], mode="r") as json_file:
            self.tms_definition = json.load(json_file)
        
        self.user_bounds = configs.get("USER_BOUNDS", None)
        self.database = configs["DATABASE"]
        self.layers = configs["LAYERS"]

        # extract tms srid that will be used to create pbf tiles
        self.pbf_srid = str(self.tms_definition["srid"])

        # process layer configs, simplifying sql template filling
        self.all_attributes = set()

        for layer in self.layers.values():
            self.all_attributes.update(layer["attributes"])
        self.all_attributes = ", ".join(self.all_attributes)

        for name, layer in self.layers.items():
            layer["pbf_srid"] = self.pbf_srid
            layer["attributes"] = ", ".join(layer["attributes"])
            layer["layer_name"] = name
          
        # derive bounding box of geometries that are supposed to be tiled
        conn = psycopg2.connect(**self.database) 
        with conn.cursor() as cur:
            
            if self.user_bounds: 
                print("BBOX is set to user defined bounds and reprojected if required.")

                if self.user_bounds["srid"] == self.pbf_srid:
                    self.bbox = self.user_bounds

                else:
                    cur.execute(self.query_user_bbox())
                    geojson_bbox = cur.fetchone()[0]
                    self.bbox = self.parse_geojson_bbox(geojson_bbox)
            
            else: 
                print("BBOX is auto-detected across all geoms of all specified layers.")

                cur.execute(self.query_autodetected_bbox())
                geojson_bbox = cur.fetchone()[0]
                self.bbox = self.parse_geojson_bbox(geojson_bbox)
        
        conn.close()   

    #-----------------------------------------------------------------------------------------------
    # methods for retrieving and parsing bounding boxes from PosGIS
    #-----------------------------------------------------------------------------------------------

    def query_user_bbox(self) -> str:
        
        '''Returns SQL query to request a bounding box as GeoJSON.
        The bbox is transformed to the srid as specified in self.pbf_srid
        
        Since the user may provide a bbox in any projection, 
        the bbox edges are densified for safe transformation to other coordinate systems.
        To this end, I follow the MVT Example of Paul Ramsay:
        https://github.com/pramsey/minimal-mvt/blob/8b736e342ada89c5c2c9b1c77bfcbcfde7aa8d82/minimal-mvt.py#L84-L91
        Though I'm not sure if densifying is generally required or if the step is only needed when the initial projection is EPSG 3857'''

        
        bounds = self.user_bounds.copy() #copy to avoid permanent modification of attribute
        DENSIFY_FACTOR = 4
        bounds['seg_size'] = (bounds['xmax'] - bounds['xmin'])/DENSIFY_FACTOR

        env = 'ST_Segmentize(ST_MakeEnvelope({xmin}, {ymin}, {xmax}, {ymax}, {srid}), {seg_size})'.format(**bounds)
        return 'SELECT ST_AsGeoJSON(ST_Transform({0}, {1}), 9, 1)'.format(env, self.pbf_srid)
    

    def query_autodetected_bbox(self) -> str: 
        
        '''Returns SQL query to request a bounding box as GeoJSON.
        The bbox is transformed to the srid as specified in self.pbf_srid

        The bbox is autodetected by creating a union of geometries 
        across all specified layers/tables (union).
        '''

        geom = 'SELECT ST_Transform({table}.{geom}, ' + self.pbf_srid + ') AS geom FROM {table}'
        geoms = [geom.format(**layer) for layer in self.layers.values()]
        geoms = " UNION ".join(geoms)
        
        return '''WITH geoms AS ({}) SELECT ST_AsGeoJSON(ST_Extent(geom), 9, 1) FROM geoms;'''.format(geoms)


    def parse_geojson_bbox(self, geojson_extent:str) -> dict:
        
        '''Returns a bbox as dictionary, parsed from a bbox
        as returned by the PostGIS ST_AsGeoJSON() command.
        '''

        geom_extent = json.loads(geojson_extent)
        bbox = geom_extent["bbox"] #bbox as a list
        names = ["xmin", "ymin", "xmax", "ymax"] #names of the bbox elements in corresponding order
        bbox = {name:point for name, point in zip(names, bbox)}
        bbox['srid'] = self.pbf_srid

        return bbox


    #-----------------------------------------------------------------------------------------------
    # methods to create a single pbf file according to zoom level, x and y
    #-----------------------------------------------------------------------------------------------

    def query_temporary_table(self):
        '''
        https://stackoverflow.com/questions/24243887/create-in-withcte-using-postgresql
        PROBABLY CHANGE SO THAT INTERSECTION HAPPENS BEFORE UNION!
        '''
  
        # template to materialize envelope polygon from bounding box
        env = 'ST_MakeEnvelope({xmin}, {ymin}, {xmax}, {ymax}, {srid})'.format(**self.bbox)

        # template to select a single layer / table
        table = """SELECT 
                    '{layer_name}' AS layer, 
                    ST_Transform({table}.{geom}, {pbf_srid}) AS geom, 
                    {attributes}
                    FROM {table}"""
        
        # template to get the union of all tables / layers
        table_union = [table.format(**layer) for layer in self.layers.values()]
        table_union = " UNION ".join(table_union)
        
        # assembled query string
        return '''CREATE TEMPORARY TABLE temp_table_for_mvt_cache AS

                    WITH
                        table_union AS ({0})
                    SELECT * from table_union
                    WHERE table_union.geom && {1};'''.format(table_union, env)


    def query_tile_from_temp_table(self, env:dict) -> str:
        '''
        SQL query to retrieve a vector tile from a predefined, temporary table.
        Selects geometries that lie within provided tile envelope. 
        '''
        
        env["srid"] = self.pbf_srid
        env = 'ST_MakeEnvelope({xmin}, {ymin}, {xmax}, {ymax}, {srid})'.format(**env)

        layer_template = """
        (
        SELECT '{0}' AS layername, ST_AsMVT(q, '{0}', 4096)   
        FROM (
            SELECT 
                layer, 
                ST_AsMVTGeom(t.geom, bbox.b2d) AS geom, 
                {1}
            FROM temp_table_for_mvt_cache AS t, bbox
            WHERE layer = '{0}' 
            AND t.geom && bbox.geom 
            ) AS q
        )
        """

        layer_union = [layer_template.format(layer["layer_name"], self.all_attributes) for layer in self.layers.values()] 
        layer_union = "\nUNION\n".join(layer_union)

        return """
                WITH 
                    bbox AS (
                        SELECT {0}::box2d AS b2d, 
                        {0} AS geom)
                {1};""".format(env, layer_union)

    #-----------------------------------------------------------------------------------------------
    #  method creating single tiles, iterating through z, x and y
    #-----------------------------------------------------------------------------------------------

    def generate_tileset(self, path:str, tileset_name:str, max_zoom:int = None):

        path = Path(path)
        
        # create tileset folder
        tileset_path = path / tileset_name
        os.mkdir(tileset_path)

        # open db connection 
        conn = psycopg2.connect(**self.database)
        cur = conn.cursor()

        # create temporary table containing all relevant geometries and attributes
        cur.execute(self.query_temporary_table())
        print('Created temp union table from all specified tables')
        print('Geoms deviating from specified pbf_srid are automatically transformed')

        # instantiate TileMatrixSet instructions
        tms = TileMatrixSet(**self.tms_definition)

        # define maximum zoom level
        max_z = max_zoom if max_zoom else tms.max_zoom
        
        # for each zoom level: 
        for z in range(0, max_z + 1): # +1 is needed since range() ends the interval at maximum value - 1
            
            print("Generating tiles for zoom level", z)

            # create zoom level folder
            zoom_level_path = tileset_path / str(z)
            os.mkdir(zoom_level_path)

            # derive tile index limits covering the bbox in current zoom level
            limits = tms.tile_limits(self.bbox, z)

            # derive all grid index combinations from tile limits
            x_min, x_max = limits["tileMinCol"], limits["tileMaxCol"]
            y_min, y_max = limits["tileMinRow"], limits["tileMaxRow"]
            grid_idx_combinations = [(x, y) for x in range(x_min, x_max + 1) for y in range(y_min, y_max + 1)] 
 
            # for each grid id on x axis, create corresponding folder 
            for x in range(x_min, x_max + 1):
                os.mkdir(zoom_level_path / str(x))
            
            # for each x and y combination of grid_ids at current zoom level, create corresponding folders and files
            for x, y in tqdm(grid_idx_combinations):
                
                tile_path = zoom_level_path / str(x)
                tile_name = str(y) + ".pbf"
                
                env = tms.tile_envelope(x,y,z)
                sql = self.query_tile_from_temp_table(env)

                cur.execute(sql)
                results = cur.fetchall()

                results = [row[1] for row in results if len(row[1]) > 0 ] # row[1] contains the binary result as "memoryview" object 

                # proceed if results is not empty
                if results: 

                    with open(tile_path / tile_name, "ab") as f:
                        for pbf in results:
                            f.write(pbf)

        # close db connection
        cur.close()
        conn.close()

        return None

