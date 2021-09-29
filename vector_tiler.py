import os
import psycopg2
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import warnings
from tqdm import tqdm 

#-------------------------------------------------------------------------------------------
# Module constants
#-------------------------------------------------------------------------------------------

EPSG_EXTENT = {
    "3857": {"srid":3857, "xmin":-20037508.342789, "ymin":-20037508.342789, "xmax":20037508.342789, "ymax":20037508.342789},
    "4326": {"srid":4326, "xmin":-180, "ymin":-90, "xmax":180, "ymax":90}
}

#-------------------------------------------------------------------------------------------
# Define Class that generates vector tilesets
#-------------------------------------------------------------------------------------------

class VectorTiler:

    def __init__(self, config_file):
        
        configs = self._read_configs(config_file)
        
        # initialize geometry related attributes
        self.pbf_srid = str(configs["PBF_SRID"])
        self.epsg_extent = EPSG_EXTENT[self.pbf_srid]


        # initialize database related attributes
        self.database = configs["DATABASE"]
        self.layers = configs["LAYERS"]

        # prepare each layer config dict to simplify filling of sql string templates
        self.all_attributes = set()
        for layer in self.layers.values():
            self.all_attributes.update(layer["attributes"])
        self.all_attributes = ", ".join(self.all_attributes)

        for name, layer in self.layers.items():
            layer["pbf_srid"] = self.pbf_srid
            layer["attributes"] = ", ".join(layer["attributes"])
            layer["layername"] = name
  
        # LEGACY (needed for debugging funcitonality, that only works with quadratic crs!!!)
        self.crs_min = self.epsg_extent["xmin"]  
        self.crs_max = self.epsg_extent["xmax"]
          
        # derive bounding boxes
        self.user_bounds = configs.get("USER_BOUNDS", None)
        
        conn = psycopg2.connect(**self.database) 
        with conn.cursor() as cur:
            
            if self.user_bounds: 
                print("BBOX is derived from user defined bounds and reprojected if required.")

                if self.user_bounds["srid"] == self.pbf_srid:
                    self.bbox = self.user_bounds

                else:
                    cur.execute(self._query_user_bbox())
                    geojson_bbox = cur.fetchone()[0]
                    self.bbox = self._parse_geojson_bbox(geojson_bbox)
            
            else: 
                print("BBOX ist auto-detected across all geoms of all specified layers.")

                cur.execute(self._query_autodetected_bbox())
                geojson_bbox = cur.fetchone()[0]
                self.bbox = self._parse_geojson_bbox(geojson_bbox)
        
        conn.close()   


    def _read_configs(self, config_file):
        with open(config_file, mode="r") as json_data_file:
            configs = json.load(json_data_file)
        return configs


    def _query_user_bbox(self) -> str:
        '''Returns bbox derived from user defined bounds (as GeoJSON).
        Bounds need to be declared in EPSG WGS84.
        The edges are densified for safe transformation to other coordinate systems.
        Here, I follow the MVT Example of Paul Ramsay:
        https://github.com/pramsey/minimal-mvt/blob/8b736e342ada89c5c2c9b1c77bfcbcfde7aa8d82/minimal-mvt.py#L84-L91
        Though I'm not sure if densifying is generally required or if the step is only needed with EPSG 3857'''

        
        bounds = self.user_bounds.copy() #copy to avoid permanent modification of attribute
        DENSIFY_FACTOR = 4
        bounds['seg_size'] = (bounds['xmax'] - bounds['xmin'])/DENSIFY_FACTOR

        env = 'ST_Segmentize(ST_MakeEnvelope({xmin}, {ymin}, {xmax}, {ymax}, {srid}), {seg_size})'.format(**bounds)
        return 'SELECT ST_AsGeoJSON(ST_Transform({0}, {1}), 9, 1)'.format(env, self.pbf_srid)
    

    def _query_autodetected_bbox(self) -> str: 
        '''Returns SQL query to request the bounding box (as GeoJSON)
        of geometries across all specified tables (union).
        Geometries are reprojected given specified srid.
        '''

        geom = 'SELECT ST_Transform({table}.{geom}, ' + self.pbf_srid + ') AS geom FROM {table}'
        geoms = [geom.format(**layer) for layer in self.layers.values()]
        geoms = " UNION ".join(geoms)
        
        return '''WITH geoms AS ({}) SELECT ST_AsGeoJSON(ST_Extent(geom), 9, 1) FROM geoms;'''.format(geoms)


    def _parse_geojson_bbox(self, geojson_extent:str) -> dict:
        '''Returns a bbox as dictionary, parsed from the extent
        as returned by "self.query_gejson_extent"
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
                    '{layername}' AS layer, 
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


    def query_tile_from_temp_table(self, z:int, x:int, y:int) -> str:
        '''
        SQL query to retrieve a vector tile from a predefined, temporary table. 
        '''
        
        level_zero_tile_env = 'ST_MakeEnvelope({xmin}, {ymin}, {xmax}, {ymax}, {srid})'.format(**self.epsg_extent)
        
        env = 'ST_TileEnvelope({0}, {1}, {2}, {3})'.format(z, x, y, level_zero_tile_env)
        
        temp_table = """SELECT 
                            layer, 
                            ST_AsMVTGeom(t.geom, bbox.b2d) AS geom, 
                            {0}
                        FROM temp_table_for_mvt_cache AS t, bbox
                        WHERE t.geom && bbox.geom""".format(self.all_attributes)

        return """
                WITH 
                    bbox AS (
                        SELECT {0}::box2d AS b2d, 
                        {0} AS geom),
                    mvt_composite AS (
                        {1}) 
                SELECT ST_AsMVT(mvt_composite.*,'composite') FROM mvt_composite;
                """.format(env, temp_table)


    #-----------------------------------------------------------------------------------------------
    # methods finding all relevant grid ids, given a bounding box and zoom level
    #-----------------------------------------------------------------------------------------------

    def find_axes_cutpoints(self, zoom_level):
        '''Given zoom level and axes extent, finds coordinates at which axis is cut into tiles'''
        n_cutpoints = 2**zoom_level
        cutpoints = np.linspace(self.crs_min, self.crs_max, endpoint=False, num=n_cutpoints)
        cutpoints = np.append(cutpoints, self.crs_max)
        return cutpoints


    def find_grid_intervals(self, cutpoints):
        
        grid_intervals = {}
        
        for axis in ["x", "y"]:
            
            xmin, xmax = self.bbox[axis + "min"], self.bbox[axis + "max"]
            id1 = self._find_grid_id(cutpoints, xmin, axis=axis)
            id2 = self._find_grid_id(cutpoints, xmax, axis=axis)
            grid_intervals[axis] = [id1, id2]
            
        grid_intervals["y"].reverse() #reverse order as ymax marks the start of the tile grid on y axis
        
        return grid_intervals
    
    
    def _find_grid_id(self, cutpoints, c, axis=None):
        ''' given axis cutpoints,
        finds tile grid index containing the coordinate c.
        Different procedures for x and y axis, since tile grid follows y axis from top to bottom
        '''
        
        if axis == "x":
            if c == min(cutpoints):
                grid_id = 0 # smallest cutpoint = minimal grid id on x
            else:
                interval_search = (c > cutpoints[:-1]) & (c <= cutpoints[1:]) 
                grid_id = int(np.where(interval_search)[0]) #np.where returns a tuple where second element is empty, so grab the first ele
                
        elif axis == "y":
            if c == max(cutpoints):
                grid_id = 0 # largest cutpoint = minimal grid id on y
            else:
                cutpoints_flipped = np.flip(cutpoints)
                interval_search = (c < cutpoints_flipped[:-1]) & (c >= cutpoints_flipped[1:])
                grid_id = int(np.where(interval_search)[0])
                
        else:
            raise ValueError("axis parameter must be 'x' or 'y'")
            
        return grid_id

    #-----------------------------------------------------------------------------------------------
    #  method creating single tiles, iterating through z, x and y
    #-----------------------------------------------------------------------------------------------

    def generate_tileset(self, path, tileset_name, zoomlevel_range=(0,15)):

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

        # for each zoom level: 
        zMin, zMax = zoomlevel_range
        for z in range(zMin, zMax + 1):
            
            print("Generating tiles for zoom level", z)

            # create zoom level folder
            zoom_level_path = tileset_path / str(z)
            os.mkdir(zoom_level_path)

            # get cutpoints and grid intervals (min / max)
            cutpoints = self.find_axes_cutpoints(z)
            grid_intervals = self.find_grid_intervals(cutpoints)

            # derive all grid id combinations from grid_id intervals
            x_min, x_max = grid_intervals['x']
            y_min, y_max = grid_intervals['y']
            grid_id_combinations = [(x, y) for x in range(x_min, x_max + 1) for y in range(y_min, y_max + 1)] 
 
            # for each grid id on x axis, create corresponding folder 
            for x in range(x_min, x_max + 1):
                os.mkdir(zoom_level_path / str(x))
            
            # for each x and y combination of grid_ids at current zoom level, create corresponding folders and files
            for x, y in tqdm(grid_id_combinations):
                
                tile_path = zoom_level_path / str(x)
                tile_name = str(y) + ".pbf"
                
                sql = self.query_tile_from_temp_table(z,x,y)

                cur.execute(sql)
                pbf = cur.fetchone()[0]

                if len(pbf) > 1:
                    with open(tile_path / tile_name, "wb") as f:
                        f.write(pbf)

        # close db connection
        cur.close()
        conn.close()

        return None

    #-----------------------------------------------------------------------------------------------
    # methods to visualize tiling of a given zoom level for debugging purposes
    #-----------------------------------------------------------------------------------------------

    def _find_axes_ranges(self, cutpoints, grid_intervals):
        '''Finds min and max coordinate of crs covered by grid_intervals.
        Only needed for visualization with "visualize tile grid"
        '''
        axes_ranges = {}
            
        for k, l in grid_intervals.items():     
            
            start, end = l #unpack start and end of grid interval
            end += 1 #add 1 to end marker to get correct axis value
            
            if k == "x": 
                axes_ranges[k] = [cutpoints[start], cutpoints[end]]
            elif k == "y":
                cutpoints_flipped = np.flip(cutpoints)
                axes_ranges[k] = [cutpoints_flipped[start], cutpoints_flipped[end]]
            else:
                raise ValueError("Key of grid_inverval item has to be either 'x' or 'y'")
        
        return axes_ranges

    def check_tiling(self, zoom_level):
        '''Computes and visualizes the tile grid for the bounding box, given a zoom level
        This can be used for debugging purposes of the tiling mechanism, which is comprised of two functions:
        -find_axes_cutpoints()
        -find_grid_intervals()'''
        
        if zoom_level > 5:
            msg = """Depending on your inputs, results may be too numerous and detailed 
            to be printed and visualized properly. Try a lower zoom level, a value < 5 should work fine."""
            warnings.warn(msg) 
        
        # run tiling mechanism  
        cutpoints = self.find_axes_cutpoints(zoom_level)
        grid_intervals = self.find_grid_intervals(cutpoints)
       
        # axes coordinates (min and max) corresponding the relevant grid_intervals of x and y
        axes_ranges = self._find_axes_ranges(cutpoints, grid_intervals)

        # prepare plot 
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        limits = [self.crs_min, self.crs_max]
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.vlines(0, self.crs_min, self.crs_max, colors='black')
        ax.hlines(0, self.crs_min, self.crs_max, colors='black')
        
        # plot cutpoints
        ax.vlines(cutpoints[1:-1], cutpoints.min(), cutpoints.max(), colors='grey')
        ax.hlines(cutpoints[1:-1], cutpoints.min(), cutpoints.max(), colors='grey')
        
        # plot bounding box (using scatter plot and filling space inbetween)
        xmin, xmax = self.bbox["xmin"], self.bbox["xmax"]
        ymin, ymax = self.bbox["ymin"], self.bbox["ymax"]
        x = [xmin, xmax, xmax, xmin]
        y = [ymax, ymax, ymin, ymin]
        ax.scatter(x,y) 
        ax.fill_between(x, y, color='blue',alpha=0.5)
        
        # fill area of tiles intersecting the axes_ranges    
        intersecting_tiles_bbox = [
        (min(axes_ranges["x"]), max(axes_ranges["y"])), # upper left corner
        (max(axes_ranges["x"]), max(axes_ranges["y"])), # upper right corner
        (max(axes_ranges["x"]), min(axes_ranges["y"])), # lower right corner
        (min(axes_ranges["x"]), min(axes_ranges["y"])) # lower left corner  
        ]
        
        intersecting_tiles_bbox = list(zip(*intersecting_tiles_bbox))
        ax.fill_between(intersecting_tiles_bbox[0], intersecting_tiles_bbox[1], color='grey',alpha=0.35)

        print("grid_intersections:", cutpoints)
        print("relevant grid intervals:", grid_intervals)
        print("corresponding axes values:", axes_ranges)

        plt.show()


