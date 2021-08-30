import os
import psycopg2
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import warnings
from tqdm import tqdm 

#-------------------------------------------------------------------------------------------
# Parse database configurations
#-------------------------------------------------------------------------------------------
with open("db_config.json") as json_data_file:
    configs = json.load(json_data_file)

DATABASE = configs["DATABASE"]
TABLE = configs["TABLE"]
HOST = configs["HOST"]
PORT = configs["PORT"]

#-------------------------------------------------------------------------------------------
# Define Class that generates vector tilesets
#-------------------------------------------------------------------------------------------
class VectorTiler:

    '''Assumes quadratic coordinate (x-axis and y-axis are of equal length)'''

    def __init__(self, crs_max, pbf_srid, densify_factor=4, reproject=True):
        
        # initialize class attributes
        self.crs_min, self.crs_max = crs_max * -1,  crs_max        
        self.pbf_srid = str(pbf_srid)
        self.densify_factor = densify_factor
        self.geom_srid = None
        self.bbox = None
        self.geom_is_reprojected = False

        # create internal version of TABLE to add and modify information
        self.tbl = TABLE.copy()
        self.tbl["pbf_srid"] = self.pbf_srid

        # derive further attributes from db 
        conn = psycopg2.connect(**DATABASE) 
        with conn.cursor() as cur:

            # find srid of geom
            cur.execute(self._sql_geom_srid())
            self.geom_srid = cur.fetchone()[0]

            # reproject geom if pbf srid deviates
            if (reproject) and (self.geom_srid != self.pbf_srid):
                print("Geom has following srid: ", self.geom_srid)
                print("A reprojected copy was created. It will be used for tiling and deleted after completion.")
                cur.execute(self._sql_reproject_geom())
                self.tbl['geom_column'] = TABLE['geom_column'] + '_' + self.pbf_srid + '_temp'
                self.geom_is_reprojected = True
                
            # get bounding box
            # CAUTION, IF REPROJECTED THEN THE OTHER GEOM SHALL BE USED!
            cur.execute(self._sql_geojson_bbox())
            geojson_bbox = cur.fetchone()[0]
            self.bbox = self._parse_geojson_bbox(geojson_bbox)
        
        conn.close()      


    def _sql_geojson_bbox(self):
        return 'SELECT ST_AsGeoJSON(ST_Envelope({geom_column})) FROM {table};'.format(**self.tbl)


    def _parse_geojson_bbox(self, geojson_bbox):
       
        bbox = json.loads(geojson_bbox)
        coords = bbox["coordinates"][0][:-1] #contains coordinates in following order: "xmin,ymin", "xmin,ymax", "xmax,ymax", "xmax,ymin"
        coords = np.array(coords) #construct coordinate matrix to simplify extraction of min and max coordinates, first column is x, second is y
        bbox = {
            "xmin": coords[:, 0].min(), 
            "xmax": coords[:, 0].max(),
            "ymin": coords[:, 1].min(),
            "ymax": coords[:, 1].max()  
        }
        return bbox


    def _sql_geom_srid(self):
        return 'SELECT ST_SRID({geom_column}) FROM {table};'.format(**self.tbl)


    def _sql_reproject_geom(self):
        return '''
            ALTER TABLE {table}
                DROP COLUMN IF EXISTS {geom_column}_{pbf_srid}_temp,
                ADD COLUMN {geom_column}_{pbf_srid}_temp geometry(Geometry, {pbf_srid});
                            
            UPDATE {table} SET 
                {geom_column}_{pbf_srid}_temp = ST_Transform({geom_column}, {pbf_srid});
            '''.format(**self.tbl)
    
    #-----------------------------------------------------------------------------------------------
    # functions to create a single pbf file according to zoom level, x and y
    #-----------------------------------------------------------------------------------------------

    def tile_to_envelope(self, z, x, y):
            
        # Width of world in CRS
        world_crs_size = self.crs_max - self.crs_min #assumes crs_min to be negative!
        # Width in tiles
        world_tile_size = 2 ** z
        # Tile width in CRS
        tile_crs_size = world_crs_size / world_tile_size
        # Calculate geographic bounds from tile coordinates
        # XYZ tile coordinates are in "image space" so origin is
        # top-left, not bottom right
        env = dict()
        env['pbf_srid'] = self.pbf_srid
        env['xmin'] = self.crs_min + tile_crs_size * x
        env['xmax'] = self.crs_min + tile_crs_size * (x + 1)
        env['ymin'] = self.crs_max - tile_crs_size * (y + 1)
        env['ymax'] = self.crs_max - tile_crs_size * (y)
        return env
   
    def sql_envelope(self, env):
        ''' Translates tile envelope to sql query, which includes the following steps:
        1) Materialize the bounds
        2) Select the relevant geometry and clip to MVT bounds
        3) Convert to MVT format
        '''
        
        tbl = self.tbl.copy()
        tbl['env'] = self._sql_envelope_to_bounds(env)
        
        return  """
            WITH 
            bounds AS (
                SELECT {env} AS geom, 
                    {env}::box2d AS b2d
            ),
            mvtgeom AS (
                SELECT ST_AsMVTGeom(t.{geom_column}, bounds.b2d) AS geom, 
                    {attr_columns}
                FROM {table} t, bounds
                WHERE ST_Intersects(t.{geom_column}, bounds.geom)
            ) 
            SELECT ST_AsMVT(mvtgeom.*) FROM mvtgeom
        """.format(**tbl)

    def _sql_envelope_to_bounds(self, env):
        '''Add explanation of author what densifying means and why it is neccessary
        This is only a partial sql statement, that is inserted in another one, so not ; at the end'''
        
        env['seg_size'] = (env['xmax'] - env['xmin']) / self.densify_factor

        return '''
            ST_Segmentize(ST_MakeEnvelope({xmin}, {ymin}, {xmax}, {ymax}, {pbf_srid}), {seg_size})
            '''.format(**env)
    
    #-----------------------------------------------------------------------------------------------
    # functions finding all relevant grid ids, given a bounding box and zoom level
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
    # Outlines of callback function iterating through z, x and y
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
            conn = psycopg2.connect(**DATABASE)
            cur = conn.cursor()

            for x, y in tqdm(grid_id_combinations):
                
                tile_path = zoom_level_path / str(x)
                tile_name = str(y) + ".pbf"
                
                env = self.tile_to_envelope(z, x, y)
                sql = self.sql_envelope(env)

                cur.execute(sql)
                pbf = cur.fetchone()[0]

                with open(tile_path / tile_name, "wb") as f:
                    f.write(pbf)

        # delete reprojected geom from source db  
        if self.geom_is_reprojected:
            
            sql = '''
            ALTER TABLE {table} 
                DROP COLUMN {geom_column}_{pbf_srid}_temp;'''.format(**self.tbl)
            cur.execute(sql)
            
        # close db connection
        cur.close()
        conn.close()

        return None

    #-----------------------------------------------------------------------------------------------
    # Additional functions to visualize tiling in a given Tile Matrix Set for debugging purposes
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

