import numpy as np
import argparse
import csv
from numpy.linalg import norm
from numpy.random import normal

# Radius of the Earth
R = 6378.1


def get_cartesian(e_coord):
    lat, long, alt = e_coord

    r = R + alt

    x = r * np.cos(long*np.pi/180)
    y = r * np.sin(long*np.pi/180)
    z = r * np.sin(lat*np.pi/180)

    c_coord = np.array([x,y,z])

    return c_coord


def get_ellipsoidal(c_coord):
    x, y, z = c_coord

    lat = np.arctan(z/np.sqrt(x**2 + y**2))*180/np.pi
    long = np.arctan(y/x)*180/np.pi
    alt = np.sqrt(x**2 + y**2 + z**2) - R

    e_coord = np.array([lat, long, alt])

    return e_coord

def get_los_vector(c_coord):
    # Satellites are always pointing towards center of earth: (x,y,z) = (0,0,0)
    los_vector = -c_coord / norm(c_coord)

    return los_vector


def get_sensors(sensor_file):
    sensors = []
    with open(sensor_file) as f:

        csv_reader = csv.reader(f, delimiter=',')

        # Create dictionary for each sensor
        for row in csv_reader:
            s = {}
            s['name'] = row[0]
            s['sig_a'] = float(row[1]) # in radians
            s['sig_b'] = float(row[2])

            if row[3].startswith('ellips'):
                s['ellipsoidal'] = np.array([float(row[4]), float(row[5]), float(row[6])])
                s['cartesian'] = get_cartesian(s['ellipsoidal'])
            else:
                s['cartesian'] = np.array([float(row[4]), float(row[5]), float(row[6])])
                s['ellipsoidal'] = get_ellipsoidal(s['cartesian'])

            s['los_vector'] = get_los_vector(s['cartesian'])

            sensors.append(s)

    return sensors


def get_observations(obs_file):
    obs = []
    with open(obs_file) as f:
        o = {}

        csv_reader = csv.reader(f, delimiter=',')

        # Create dictionary for each sensor
        for row in csv_reader:
            o['name'] = row[0]

            if row[1].startswith('ellips'):
                o['ellipsoidal'] = (float(row[2]), float(row[3]), float(row[4]))
                o['cartesian'] = get_cartesian(o['ellipsoidal'])
            else:
                o['cartesian'] = (float(row[2]), float(row[3]), float(row[4]))
                o['ellipsoidal'] = get_ellipsoidal(o['cartesian'])

            obs.append(o)


    return obs


def get_pointing_vectors(sensors, obs_point):
    # First calculate the ground-truth pointing vector and then apply uncertainty
    N = len(sensors)
    pvectors = np.zeros([N, 3])
    gt_pvectors = np.zeros([N,3])

    for i,s in enumerate(sensors):
        # GT pointing vector
        gt_pv = (obs_point - s['cartesian'])
        gt_pv = gt_pv / norm(gt_pv)

        # Apply uncertainties from sensor's LOS to pointing vector
        pv = apply_uncertainty(gt_pv, s['sig_a'], s['sig_b'])

        pvectors[i,:] = pv
        gt_pvectors[i,:] = gt_pv

    return pvectors, gt_pvectors


def apply_uncertainty(gt_pv, sig_a, sig_b):
    x, y, z = gt_pv

    r = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(y, x)   # Minor axis
    theta = np.arccos(z/r) # Major axis

    # Apply uncertainties to major and minor axis. Convert degrees to radians
    new_phi = phi + normal(0, sig_b/180*np.pi)
    new_theta = theta + normal(0, sig_a/180*np.pi)

    # Get new pointing vector in cartesian coordinates
    new_x = r * np.sin(new_theta) * np.cos(new_phi)
    new_y = r * np.sin(new_theta) * np.sin(new_phi)
    new_z = r * np.cos(new_theta)

    pv = np.array([new_x, new_y, new_z])
    pv = pv / norm(pv)

    return pv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--sensor_input', default='sensor.csv')
    parser.add_argument('--observation_input', default='obs.csv')
    args = parser.parse_args()

    sensors = get_sensors(args.sensor_input)
    obs = get_observations(args.observation_input)

    for s in sensors:
        print("SENSOR: {} XYZ: {}".format(s["name"], s['cartesian']))
        
    # Get the pointing vectors of every sensor for each observation point
    for o in obs:
        obs_point = o['cartesian']
        pv, pv_gt = get_pointing_vectors(sensors, obs_point)

        print("Ground Truth Pointing Vectors: \n",  pv_gt)
        print("Pointing Vectors with Uncertainty: \n",  pv)

        print("Abs. Error:", np.sum(np.abs(pv - pv_gt)))






