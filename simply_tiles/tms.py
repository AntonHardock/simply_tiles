import math

#-------------------------------------------------------------------------
# Pseudocode as translated from trex
# https://github.com/t-rex-tileserver/t-rex/blob/master/tile-grid/src/grid.rs
# USES only 
#-------------------------------------------------------------------------

METERS_PER_DEGREE = 6378137.0 * 2.0 * math.pi / 360.0

METERS_PER_FEET = 0.3048

STANDARDIZED_PIXEL_SIZE = 0.28 # in mm, assumed to be identical for x and y axis

EPSILON = 0.0000001 #constant used to avoid precision issues with floating point calculations

ORIGIN = "top_left" #all calculations are based on the assumption that top left is the origin of tile matrix


class TileMatrixSet():
    
    def __init__(self, title:str, identifier:str, width:int = None, height:int = None, 
                 extent:dict = None, srid:int = None, units:str = None, 
                 cell_sizes:list = None, origin:str="top_left"):
        
        self.title = title
        self.identifier = identifier
        self.srid = srid # crs used by provided tile matrix set definition
        self.units = units # measuring unit used by given crs
        self.extent = extent # "global bbox" across all tile matrices / zoomlevels, in crs units
        self.tile_width = width # number of pixels along X axis dividing a tile into "cells"
        self.tile_height = height # number of pixels along Y axis dividing a tile into "cells"
        self.cell_sizes = cell_sizes # list of cell sizes for each tile matrix / zoomlevel, in crs units
        self.origin = origin # the extent's corner used as origin for counting tiles in terms of zero based indices
        if self.origin != "top_left":
            raise ValueError("Origin has to be 'top_left' in this implementation")
        
        self.n_levels = len(cell_sizes) # derive number of predefined zoomlevels / tile matrices
        self.max_zoom = self.n_levels - 1 # derive max zoom, taking into account zero based indexing of zoomlevels


    def scale_denominator(self, zoom_level:int) -> float:
        """Derive a general scale denominator from provided cell_sizes (also called reslution levels)
        To do this, a physical pixel size has to be assumed.
        Here, according to the OGC TMS Specifiaction, a standardized pixel size is used 
        Reference: https://www.ogc.org/standards/se
        
        Since this "standard pixel" is defined in the metrical system (millimeters),
        a given cell_size first has to be recalculated in meters (if that't not already the case)
        Then, the cell_size is simply divided by the "standardized pixel size" (also recalculated in meters).
        The result is a scale without a particular unit. 
        """
        
        if self.units == "meters":
            cell_size_in_meters = self.cell_sizes[zoom_level]
        elif self.units == "degrees":
            cell_size_in_meters = self.cell_sizes[zoom_level] * METERS_PER_DEGREE
        elif self.units == "feet":
            cell_size_in_meters = self.cell_sizes[zoom_level] * METERS_PER_FEET
        else:
            raise ValueError("Unit needs to be one of meters, degrees or feet")
        
        scale =  cell_size_in_meters / (STANDARDIZED_PIXEL_SIZE / 1000)

        return scale


    def tile_limits(self, bbox:dict, zoom_level:int) -> dict:

        """Given the predefined TMS extent, 
        derive the first and last tile index (along x and x axis each) 
        covering a particular bounding box in the given tile matrix / zoomlevel.
        
        Direct implementation of TMS 2.0 Pseudocode:
        https://docs.opengeospatial.org/DRAFTS/17-083r3.html#from-bbox-to-tile-indices
        """

        # raise error if bbox goes beyond the defined extent
        a = (bbox["xmin"] < self.extent["xmin"])
        b = (bbox["ymin"] < self.extent["ymin"])
        c = (bbox["xmax"] > self.extent["xmax"])
        d = (bbox["ymax"] > self.extent["ymax"])
            
        if a or b or c or d:
            raise ValueError("Bounding Box lies beyond the TMS extent")
       
        # reference existing variables to match TMS Spec terminology
        bbox_xmin, bbox_xmax = bbox["xmin"], bbox["xmax"]
        bbox_ymin, bbox_ymax = bbox["ymin"], bbox["ymax"]
        tile_matrix_xmin, tile_matrix_ymax = self.extent["xmin"], self.extent["ymax"] 

        # derive tile width and length in crs units
        cell_size = self.cell_sizes[zoom_level]
        tilespan_x, tilespan_y = (self.tile_width * cell_size), (self.tile_height * cell_size)

        # calculate tile limits
        limits = {
            "tileMinCol": math.floor((bbox_xmin - tile_matrix_xmin) / tilespan_x + EPSILON),
            "tileMaxCol": math.floor((bbox_xmax - tile_matrix_xmin) / tilespan_x - EPSILON),
            "tileMinRow": math.floor((tile_matrix_ymax - bbox_ymax) / tilespan_y + EPSILON),
            "tileMaxRow": math.floor((tile_matrix_ymax - bbox_ymin) / tilespan_y - EPSILON)
        }

        # derive total number of tiles along x and y axis
        limits["matrixWidth"] = limits["tileMaxCol"] - limits["tileMinCol"] + 1
        limits["matrixHeight"] = limits["tileMaxRow"] - limits["tileMinRow"] + 1

        return limits


    def tile_envelope(self, tile_col:int, tile_row:int, zoom_level:int) -> dict:
        
        """Returns bbox of a tile in crs units, given tile indices along x and y 
        (tile_col and tile_row) as well as tile matrix id / zoomlevel.
        Here, the bbox is coined 'tile envelope' to delinieate conceptually
        from the bbox with the target geometries and the bbox of the whole TMS
        (extent)"""

        # reference existing variables to match TMS Spec terminology
        tile_matrix_minx, tile_matrix_maxy = self.extent["xmin"], self.extent["ymax"] 

        # derive tile width and length in crs units
        cell_size = self.cell_sizes[zoom_level]
        tilespan_x, tilespan_y = (self.tile_width * cell_size), (self.tile_height * cell_size)

        # calculate envelope
        envelope = {
            "xmin": tile_col * tilespan_x + tile_matrix_minx,
            "ymin": tile_matrix_maxy - (tile_row + 1) * tilespan_y,
            "xmax": (tile_col + 1) * tilespan_x + tile_matrix_minx,
            "ymax": tile_matrix_maxy - tile_row * tilespan_y
        }

        return envelope

#-------------------------------------------------------------------------
# Testing
#-------------------------------------------------------------------------

if __name__ == "__main__":

    # parse config params
    import json

    with open('data/TrexCustomTMS.json', mode="r") as json_data_file:
        tms_definition = json.load(json_data_file)

    tms = TileMatrixSet(**tms_definition)

    #calculate extent width and height
    extent_width = tms.extent["xmax"] - tms.extent["xmin"]
    extent_height = tms.extent["ymax"] - tms.extent["ymin"]
    print(extent_width, extent_height) 

    x = 0
    y = 0
    z = 2

    print(tms.cell_sizes[z])
    print(tms.scale_denominator(z))
    print(tms.tile_limits(tms.extent, z))

    env = tms.tile_envelope(x, y, z)
    env_width = env["xmax"] - env["xmin"]
    env_height = env["ymax"] - env["ymin"]
    print(env_width, env_height)
    print(env_width / 256)
    print(tms.extent)



