import os
import psycopg2
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import warnings
from tqdm import tqdm 

#-------------------------------------------------------------------------------------------
# Define Class that generates vector tilesets
#-------------------------------------------------------------------------------------------

class VectorTiler:

    '''Assumes quadratic coordinate (x-axis and y-axis are of equal length)'''

    def __init__(self, config_file, crs_max, pbf_srid, densify_factor=4):
        
        
        # initialize database related attributes  
        configs = self._read_configs(config_file)
        self.database = configs["DATABASE"]
        self.layers = configs["LAYERS"]

        # initialize geometry related attributes
        self.crs_min, self.crs_max = crs_max * -1,  crs_max        
        self.pbf_srid = str(pbf_srid)
        self.densify_factor = densify_factor
        self.geom_srid = None
        self.bbox = None

        # derive further attributes from db 
        conn = psycopg2.connect(**self.database) 
        with conn.cursor() as cur:

            # find srid of geom
            # cur.execute(self._sql_geom_srid())
            # self.geom_srid = cur.fetchone()[0]
            # print("Geom has following srid: ", self.geom_srid)
            
            # if self.geom_srid != self.pbf_srid:     
            #     print("Geom will be reprojected to: ", self.pbf_srid, " for each tile requested")

            # # add srid information to self.table to simplify subsequent query string generation
            # self.table["pbf_srid"] = self.pbf_srid
            # self.table["geom_srid"] = self.geom_srid
            
            # get bounding box
            cur.execute(self._query_geojson_extent())
            geojson_bbox = cur.fetchone()[0]
            self.bbox = self._parse_geojson_bbox(geojson_bbox)
        
        conn.close()   


    def _read_configs(self, config_file):
        with open(config_file, mode="r") as json_data_file:
            configs = json.load(json_data_file)
        return configs

    
    def _query_geojson_extent(self) -> str: 
        '''Returns SQL query to request the bounding box (as GeoJSON)
        of geometries across all specified tables (union).
        '''

        geom = '''SELECT {table}.{geom} AS geom FROM {table} '''
        geoms = [geom.format(**layer) for layer in self.layers.values()]
        geoms = " UNION ".join(geoms)
        
        return '''WITH geoms AS ({}) SELECT ST_AsGeoJSON(ST_Extent(geom), 9, 1) FROM geoms;'''.format(geoms)


    def _parse_geojson_bbox(self, geojson_extent:str) -> dict:
        '''Returns a bbox as dictionary, parsed from the extent
        as returned by "self.query_gejson_extent"'''

        geom_extent = json.loads(geojson_extent)
        bbox = geom_extent["bbox"] #bbox as a list
        names = ["xmin", "ymin", "xmax", "ymax"] #names of the bbox elements in corresponding order
        bbox = {name:point for name, point in zip(names, bbox)}

        return bbox


    def _sql_geom_srid(self):
        return 'SELECT ST_SRID({geom_column}) FROM {table};'.format(**self.table)

    #-----------------------------------------------------------------------------------------------
    # methods to create a single pbf file according to zoom level, x and y
    #-----------------------------------------------------------------------------------------------

    def calculate_tile_envelope(self, z:int, x:int, y:int) -> dict:
    
        '''
        "Calculate geographic tile bounds from tile coordinates.
        XYZ tile coordinates are in "image space" so origin is
        top-left, not bottom right"
        (Paul Ramsay: https://github.com/pramsey/minimal-mvt)
        '''

        world_crs_size = self.crs_max - self.crs_min # Width of world in CRS - assumes crs_min to be negative!
        world_tile_size = 2 ** z  # Width in tiles
        tile_crs_size = world_crs_size / world_tile_size # Tile width in CRS

        env = dict()
        env['pbf_srid'] = self.pbf_srid
        env['xmin'] = self.crs_min + tile_crs_size * x
        env['xmax'] = self.crs_min + tile_crs_size * (x + 1)
        env['ymin'] = self.crs_max - tile_crs_size * (y + 1)
        env['ymax'] = self.crs_max - tile_crs_size * (y)

        return env


    def _sql_envelope(self, env:dict) -> str:
        
        '''Returns a SQL query fragment. 
        It constructs a rectangle polygon geometry based on tile envelope parameters.
        The edges are densified, meaning that the polygon will have more coordinates
        than needed to define the envelope. This is done so that
        the envelope "can be safely converted to other coordinate systems"
        (see Paul Ramsay: https://github.com/pramsey/minimal-mvt)
        '''
        
        env['seg_size'] = (env['xmax'] - env['xmin']) / self.densify_factor
        return 'ST_Segmentize(ST_MakeEnvelope({xmin}, {ymin}, {xmax}, {ymax}, {pbf_srid}), {seg_size})'.format(**env)
        

    def _sql_mvt_layer(self, layer:dict) -> str:
        '''Returns a SQL query fragment. 
        It defines a single layer, consisting of the MVT-encoded geom 
        together with optional attributes.
        '''

        layer = layer.copy() # avoids permanent modification of the input
        attribute_list = layer["attributes"]

        if attribute_list: 

            layer["attributes"] = ", ".join(attribute_list)

            return '''SELECT ST_AsMVTGeom({table}.{geom},bbox.b2d) AS geom, {attributes}
            FROM {table}, bbox
            WHERE {table}.{geom} && bbox.geom'''.format(**layer)

        else:

            return '''SELECT ST_AsMVTGeom({table}.{geom},bbox.b2d) AS geom
            FROM {table}, bbox
            WHERE {table}.{geom} && bbox.geom'''.format(**layer)
            

    def _sql_mvtcomposite(self, layers:dict):
        '''Returns a SQL query fragment.
        The fragment combines single layer queries into one sql result set.
        '''

        layers = [self._sql_mvt_layer(layer) for layer in self.layers.values()]
        return " \nUNION \n".join(layers)


    def query_tile(self, env:dict):
        ''' 
        Assembles full sql query to request a vector tile 
        using the union of all provided tables and given the envelope parameters
        Thanks to Shahzad Bacha for demonstrating the required query logic:
        https://medium.com/@shahzadbacha.gis/composite-mvt-tiles-with-postgis-4b30d6c9f510
        '''
        
        env = self._sql_envelope(env)
        mvtcomposite = self._sql_mvtcomposite(self.layers)
        
        return '''
                WITH 
                    bbox AS (SELECT {0}::box2d AS b2d, {0} AS geom),
                    mvtcomposite AS ({1}) 
                SELECT ST_AsMVT(mvtcomposite.*,'composite') FROM mvtcomposite ;
                '''.format(env, mvtcomposite)
    
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
            conn = psycopg2.connect(**self.database)
            cur = conn.cursor()

            for x, y in tqdm(grid_id_combinations):
                
                tile_path = zoom_level_path / str(x)
                tile_name = str(y) + ".pbf"
                
                env = self.calculate_tile_envelope(z, x, y)
                sql = self.query_tile(env)

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


