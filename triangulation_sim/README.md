# Triangulation Simulation


## CSV Format

### sensor.csv
name |  sigma_a  | sigma_b | Coordinates type | (x, lat) | (y, long) | (z, alt)
-----|-----------|---------|------------------|----------|-----------|----------
GOES-16|1|1|ellipsoidal|0|75.2|35786.65
GOES-17|1|1|ellipsoidal|0|137.2|35785.7

These are the coordinates of two GOES satellites in ellipsoidal coordinates

### obs.csv
name | Coordinates type | (x, lat) | (y, long) | (z, alt)
-----|------------------|----------|-----------|----------
0|ellipsoidal|0|106.2|0

This is an observation point on the equater halfway between GOES-16 and GOES-17

### Notes
- Coordinate type are either cartesian (x,y,z) or ellipsoidal (lat, long, alt)
- sigma_a and sigma_b are the uncertainties of the sensor's LOS w.r.t. its major and minor axes respectively. Units are in degrees.


## Outputs

### get_sensors
Returns a list of sensor dictionaries from the input .csv file. Each dictionary has the following attributes:
- name: string ID of sensor
- sig_a: float value of sigma used in normal distribution. Used to calculate uncertainty offset of major axis.
- sig_b: float value of sigma used in normal distribution. Used to calculate uncertainty offset of minor axis.
- cartesian: np.array position in (x, y, z)
- ellipsoidal: np.array position in (lat, long, alt)
- los_vector: np.array unit vector pointing towards the origin (center of earth)

### get_observations
Returns a list of observation dictionaries from the input .csv file. Each dictionary has the following attributes:
- name: string ID of observation
- cartesian: np.array position in (x, y, z)
- ellipsoidal: np.array position in (lat, long, alt)


### get_pointing_vectors
Returns two 2D-np.arrays: 
- Ground-truth pointing vectors
- Pointing vectors with uncertainties applied.
- The shape of each 2D array is (# of sensors, 3)
