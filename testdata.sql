CREATE TABLE test (
	id BIGSERIAL NOT NULL PRIMARY KEY,
	bezeichnung VARCHAR(100) NOT NULL,
	x_manual NUMERIC NOT NULL,
	y_manual NUMERIC NOT NULL
);

INSERT INTO test (id, bezeichnung, x_manual, y_manual) VALUES (1, 'Punkt A', 568519.83, 5964170.19);
INSERT INTO test (id, bezeichnung, x_manual, y_manual) VALUES (2, 'Punkt B', 566535.46, 5933478.54);
INSERT INTO test (id, bezeichnung, x_manual, y_manual) VALUES (3, 'Punkt C', 606090.64, 5936124.37);

SELECT AddGeometryColumn('public','test','geom',25832,'POINT',2);

UPDATE public.test 
    SET geom = ST_SetSRID(
        ST_MakePoint(
            x_manual::DOUBLE PRECISION,
            y_manual::DOUBLE PRECISION
        ), 25832);