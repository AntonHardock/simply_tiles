# Debian 11 based python image - more lightweight in combination with our PostGIS image which is also based off a Debain Distro
FROM python:3.7.12-bullseye

# set working directory inside container
WORKDIR /simply_tiles

# install the only dependency needed
RUN pip install psycopg2==2.8.3

# copy source code
COPY simply_tiles .

# copy tms definitions
COPY tms_definitions .

