import math

import numpy as np

# Object transformation -> Camera Transformation
def camera_parameters(euler_angles, translation, distance, up=(0.0, 1.0, 0.0)):
    # Calculate Camera Position
    cam_azim = -euler_angles[0]
    cam_elev = math.pi/2 - euler_angles[1]
    cam_pos_hat = np.array([
        math.cos(cam_azim)*math.sin(cam_elev),
        math.cos(cam_elev),
        math.sin(cam_azim)*math.sin(cam_elev)
    ])
    cam_position = cam_pos_hat * distance - translation

    # Calculate Camera Look-at
    cam_dir = -cam_pos_hat
    cam_look_at = cam_position + cam_dir

    # Calculate Camera Up direction
    up = np.array(up)
    axis = -cam_pos_hat
    cam_up = np.dot(up, axis) * axis +\
             np.cross(axis, up) * math.sin(euler_angles[2]) +\
             np.cross(np.cross(axis, up), axis) * math.cos(euler_angles[2])
    return (cam_position, cam_look_at, cam_up)

def get_rotation_matrix_y(angle):
    cos= np.cos(angle)
    sin= np.sin(angle)
    return np.array(
        [[ cos,   0, sin, 0],
         [   0,   1,   0, 0],
         [-sin,   0, cos, 0],
         [   0,   0,   0, 1]])
