# Deal with rotation filtering: https://math.stackexchange.com/questions/2621677/extended-kalman-filter-equation-for-orientation-quaternion

import numpy as np
import numpy.linalg as npl
import math
import filterpy.kalman as kf
from scipy.spatial.transform import Rotation

import sys
from d3d.abstraction import ObjectTarget3D

##### simple motion models: http://fusion.isif.org/proceedings/fusion08CD/papers/1569107835.pdf

def wrap_angle(theta):
    '''
    Normalize the angle to [-pi, pi]

    :param float theta: angle to be wrapped
    :return: wrapped angle
    :rtype: float
    '''

    return (theta + np.pi) % (2*np.pi) - np.pi

def motion_CV(state, dt):
    '''
    Constant Velocity model

    :param state: original state in format [x, y, vx, vy]
    :type state: np.ndarray or list
    :param float dt: time difference after last update
    :return: updated state
    :rtype: np.ndarray
    '''

    state = np.copy(state)
    state[0] += state[2] * dt
    state[1] += state[3] * dt
    return state

def motion_CTRV(state, dt):
    raise NotImplementedError()

def motion_CTRA(state, dt):
    '''
    Constant Turn-Rate and (longitudinal) Acceleration model.
    This model also assume that velocity is the same with heading angle.
    CV, CTRV can be modeled by assume value equals zero

    :param state: original state in format [x, y, theta, v, a, w]
    :type state: np.ndarray or list, contents are [x, y, vx, vy]
    :param dt: time difference after last update
    :return: updated state
    :rtype: np.ndarray
    '''

    x, y, th, v, a, w = state
    nth = wrap_angle(th + w * dt)
    nv = v + a * dt
    if np.isclose(w, 0):
        nx = x + (nv + v)/2 * np.cos(th) * dt
        ny = y + (nv + v)/2 * np.sin(th) * dt
    else:
        nx = x + ( nv*w*np.sin(nth) + a*np.cos(nth) - v*w*np.sin(th) - a*np.cos(th)) / (w*w)
        ny = y + (-nv*w*np.cos(nth) + a*np.sin(nth) + v*w*np.cos(th) - a*np.sin(th)) / (w*w)

    state = np.copy(state)
    state[:4] = (nx, ny, nth, nv)
    return state

def motion_CSAA(state, dt):
    '''
    Constant Steering Angle and Acceleration model.

    :param state: original state in format [x, y, theta, v, a, c]
                                           [0  1    2    3  4  5]
    :type state: np.ndarray or list, contents are [x, y, vx, vy]
    :param dt: time difference after last update
    :return: updated state
    :rtype: np.ndarray
    '''

    x, y, th, v, a, c = state
    
    gamma1 = (c*v*v)/(4*a) + th
    gamma2 = c*dt*v + c*dt*dt*a - th
    eta = np.sqrt(2*np.pi)*v*c
    zeta1 = (2*a*dt + v)*np.sqrt(c/2*a*np.pi)
    zeta2 = v*np.sqrt(c/2*a*np.pi)
    sz1, cz1 = fresnel(zeta1)
    sz2, cz2 = fresnel(zeta2)
    
    nx = x + (eta * (np.cos(gamma1)*cz1 + np.sin(gamma1)*sz1 - np.cos(gamma1)*cz2 - np.sin(gamma1)*sz2) +
        2*np.sin(gamma2)*np.sqrt(a*c) + 2*np.sin(th)*np.sqrt(a*c)) / 4*np.sqrt(a*c)*c
    ny = y + (eta * (-np.cos(gamma1)*sz1 + np.sin(gamma1)*cz1 - np.sin(gamma1)*cz2 - np.cos(gamma1)*sz2) +
        2*np.cos(gamma2)*np.sqrt(a*c) - 2*np.sin(th)*np.sqrt(a*c)) / 4*np.sqrt(a*c)*c
    nth = wrap_angle(th - c*dt*dt*a/2 - c*dt*v)
    nv = v + a*dt

    state = np.copy(state)
    state[:4] = (nx, ny, nth, nv)
    return state

##### End motion models #####

class PropertyFilter:
    @property
    def dimension(self):
        '''
        Current dimension estimation
        '''
        pass
    @property
    def dimension_var(self):
        '''
        Covariance for current shape estimation
        '''
        pass
    @property
    def classifcation(self):
        '''
        Current classification estimation
        '''
        pass
    @property
    def classifcation_var(self):
        '''
        Covariance for current classification estimation
        '''
        pass

class PoseFilter:
    # TODO: also reports velocity and its variance
    @property
    def position(self):
        '''
        Current position estimation
        '''
        pass
    @property:
    def position_var(self):
        '''
        Covariance for current position estimation
        '''
        pass
    @property
    def orientation(self):
        '''
        Current orientation estimation
        '''
        pass
    @property
    def orientation_var(self):
        '''
        Covariance for current orientation estimation
        '''
        pass
    @property
    def velocity(self):
        '''
        Current linear velocity estimation
        '''
        pass
    @property
    def velocity_var(self):
        '''
        Covariance for current linear velocity estimation
        '''
        pass
    @property
    def angular_velocity(self):
        '''
        Current angular velocity estimation
        '''
        pass
    @property
    def angular_velocity_var(self):
        '''
        Covariance for current angular velocity esimation
        '''
        pass

class Box_KF(PropertyFilter):
    '''
    Use kalman filter (simple bayesian filter) for a box shape estimation.
    Use latest value for classification result
    '''
    def __init__(self, init: ObjectTarget3D, Q=np.eye(3)):
        '''
        :param init: Initial state for the detection
        '''
        self._filter = kf.KalmanFilter(dim_x=3, dim_z=3)

        # initialize matrices
        self._filter.F = self._filter.H = np.eye(3)
        self._filter.Q = np.asarray(Q).reshape(3,3)

        # feed input
        self._filter.x = init.dimension
        self._filter.P = init.dimension_var

        self._saved_tag = init.tag

    def predict(self, dt):
        self._filter.predict(dt=dt)

    def update(self, detection: ObjectTarget3D):
        self._filter.update(detection.dimension, R=detection.dimension_var)

        # Update classification
        # TODO: do bayesian filtering for classification?
        self._saved_tag = init.tag

    @property
    def dimension(self):
        return self._filter.x

    @property
    def dimension_var(self):
        return self._filter.P

    @property
    def classifcation(self):
        return self._classes

    @property
    def classification_var(self):
        raise NotImplementedError()

class Pose_3DOF_UKF_CV:
    '''
    UKF using constant velocity model for pose estimation, assuming 3DoF (x, y, yaw)

    States: [x, y, vx, vy]
            [0  1  2   3 ]
    Observe: [x, y]
             [0  1]
    '''
    def __init__(self, init: ObjectTarget3D, Q=np.eye(4)):
        # create filter
        sigma_points = kf.MerweScaledSigmaPoints(4, alpha=.1, beta=.2, kappa=-1)
        self._filter = kf.UnscentedKalmanFilter(dim_x=4, dim_z=2, dt=None,
            fx=dm.motion_CV, hx=lambda s: s[:2], points=sigma_points
        )
        self._filter.Q = np.asarray(Q).reshape(4, 4)

        # feed initial state
        self._filter.x = np.array([init.position[0], init.position[1], 0, 0])
        self._filter.P = np.copy(self._filter.Q)
        self._filter.P[:2,:2] = init.position_var[:2, :2]

        self._save_z = init.position[2] # TODO: use simple bayesian filter for z values
        self._save_z_var = init.position_var[2, 2]
        self._save_ori = init.orientation # TODO: use simple bayesian filter for orientation
        self._save_ori_var = init.orientation_var

    def predict(self, dt):
        self._filter.predict(dt=dt)

    def update(self, detection: ObjectTarget3D):
        self._save_z = detection.position[2]
        self._save_z_var = detection.position_var[2, 2]
        self._save_ori = detection.orientation
        self._save_ori_var = detection.orientation_var

        self._filter.update(detection.position[:2], R=detection.position_var[:2,:2])
            
    @property
    def position(self):
        return np.array([self._filter.x[0], self._filter.x[1], self._save_z])

    @property
    def position_var(self):
        cov = np.diag([np.inf, np.inf, self._save_z_var])
        cov[:2, :2] = self._filter.P[:2, :2]
        return cov

    @property
    def orientation(self):
        return self._save_ori

    @property
    def orientation_var(self):
        return self._save_ori_var

    @property
    def velocity(self):
        return np.array([self._filter.x[3], self._filter.x[4], 0])

    @property
    def velocity_var(self):
        cov = np.zeros(3, 3)
        cov[:2, :2] = self._filter.P[3:5, 3:5]
        return cov

    @property
    def angular_velocity(self):
        return np.zeros(3)

    @property
    def angular_velocity_var(self):
        return np.zeros((3, 3))

class Pose_3DOF_UKF_CTRV:
    '''
    UKF using constant turning rate and velocity (CTRV) model for pose estimation
    
    States: [x, y, rz, v, w]
            [0  1  2   3  4]
    Observe: [x, y, rz]
             [0  1  2 ]
    '''
    def __init__(self):
        raise NotImplementedError()

class Pose_3DOF_UKF_CTRA:
    '''
    UKF using constant turning rate and acceleration (CTRA) model for pose estimation

    States: [x, y, rz, v, a, w]
            [0  1  2   3  4  5]
    Observe: [x, y, rz]
             [0  1  2 ]
    '''
    def _state_mean(self, sigmas, Wm):
        x = np.average(sigmas, axis=0, weights=Wm)
        s = np.average(np.sin(sigmas[:, 2]), axis=0, weights=Wm)
        c = np.average(np.cos(sigmas[:, 2]), axis=0, weights=Wm)
        x[2] = np.arctan2(s, c)
        return x

    def _state_diff(self, x, y):
        d = x - y
        d[2] = wrap_angle(d[2])
        return d

    def __init__(self, init: ObjectTarget3D, Q=np.eye(6)):
        sigma_points = kf.MerweScaledSigmaPoints(6, alpha=.1, beta=.2, kappa=-1)
        self._filter = kf.UnscentedKalmanFilter(dim_x=6, dim_z=3, dt=None
            fx=dm.motion_CTRA, hx=lambda s: s[:3], points=sigma_points,
            x_mean_fn=self._state_mean, z_mean_fn=self._state_mean,
            residual_x=self._state_diff, residual_z=self._state_diff,
        )
        self._filter.Q = np.asarray(Q).reshape(6, 6)

        # feed initial state
        yaw, pitch, roll = init.orientation.as_euler("ZYX")
        self._filter.x = np.array([init.position[0], init.position[1], yaw, 0, 0, 0])
        self._filter.P = np.copy(self._filter.Q)
        self._filter.P[:2,:2] = init.position_var[:2, :2]
        self._filter.P[2] = init.orientation_var

        self._save_z = init.position[2] 
        self._save_z_var = init.position_var[2, 2]
        self._save_pitch = pitch # TODO: use simple bayesian filter for pitch and roll values
        self._save_roll = roll

    def predict(self, dt):
        self._filter.predict(dt=dt)

    def update(self, detection: ObjectTarget3D):
        yaw, pitch, roll = detection.orientation.as_euler("ZYX")
        self._save_z = detection.position[2]
        self._save_z_var = detection.position_var[2, 2]
        self._save_pitch = pitch
        self._save_roll = roll
        self._save_ori_var = detection.orientation_var

        obsv = np.array([detection.position[0], detection.position[1], yaw])
        R = np.zeros((3, 3))
        R[:2, :2] = detection.position_var
        R[2, 2] = detection.orientation_var
        self._filter.update(obsv, R=R])
        self._filter.x[2] = wrap_angle(self._filter.x[2])

    @property
    def pose(self):
        return np.array([self._filter.x[0], self._filter.x[1], self._save_z])

    @property
    def position_var(self):
        cov = np.diag([np.inf, np.inf, self._save_z_var])
        cov[:2, :2] = self._filter.P[:2, :2]
        return cov

    @property
    def orientation(self):
        return Rotation.from_euler("ZYX",
            [self._filter.x[2], self._save_pitch, self._save_roll])

    @property
    def orientation_var(self);
        return self._save_ori_var

    @property
    def velocity(self):
        return np.array([
            self._filter.x[3] * np.cos(self._filter.x[2]),
            self._filter.x[3] * np.sin(self._filter.x[2]),
            0
        ])

    @property
    def velocity_var(self):
        # here we will return the linearized covariance
        cov = np.zeros(3, 3)
        A = np.array([
            [-self._filter.x[3] * np.cos(self._filter.x[2]), np.cos(self._filter.x[2])],
            [ self._filter.x[3] * np.sin(self._filter.x[2]), np.sin(self._filter.x[2])]
        ])
        cov[:2, :2] = np.dot(A, self._filter.P[2:4, 2:4])
        return cov

    @property
    def angular_velocity(self):
        return np.array([0, 0, self._filter.x[5]])

    @property
    def angular_velocity(self):
        return np.diag([0, 0, self._filter.P[5, 5]])

class Pose_IMM:
    '''
    UKF using IMM (BR + CV + CA + CTRV + CTRA) model for pose estimation
    '''
    def __init__(self):
        raise NotImplementedError()
