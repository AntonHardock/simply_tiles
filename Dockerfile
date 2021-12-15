# Debian 11 based python image - more lightweight in combination with our PostGIS image which is also based off a Debain Distro
FROM python:3.7.12-bullseye

# set working directory inside container
WORKDIR /simply_tiles

# install dependencies
RUN pip install psycopg2==2.8.3 tqdm==4.50.2

# copy source code
COPY simply_tiles .

# copy tms definitions
COPY tms_definitions .

