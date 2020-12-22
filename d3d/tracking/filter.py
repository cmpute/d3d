# Deal with rotation filtering: https://math.stackexchange.com/questions/2621677/extended-kalman-filter-equation-for-orientation-quaternion

import logging
import math
import sys
from warnings import warn

import filterpy.kalman as kf
import numpy as np
import numpy.linalg as npl
from d3d.abstraction import ObjectTarget3D
from scipy.spatial.transform import Rotation

_logger = logging.getLogger("d3d")


def is_pd(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = npl.cholesky(B)
        return True
    except npl.LinAlgError:
        return False

def nearest_pd(A):
    """Find the nearest positive-definite matrix to input
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].
    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = npl.svd(B, hermitian=True)

    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2

    if is_pd(A3):
        return A3

    spacing = np.spacing(npl.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not is_pd(A3):
        mineig = np.min(np.real(npl.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

##### simple motion models: http://fusion.isif.org/proceedings/fusion08CD/papers/1569107835.pdf

def wrap_angle(theta):
    '''
    Normalize the angle to [-pi, pi)

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
    def classification(self):
        '''
        Current classification estimation
        '''
        pass
    @property
    def classification_var(self):
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
    @property
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
        self._filter.predict()

    def update(self, detection: ObjectTarget3D):
        self._filter.update(detection.dimension, R=detection.dimension_var)

        # Update classification
        # TODO: do bayesian filtering for classification?
        self._saved_tag = detection.tag

    @property
    def dimension(self):
        return self._filter.x

    @property
    def dimension_var(self):
        return self._filter.P

    @property
    def classification(self):
        return self._saved_tag

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
        sigma_points = kf.JulierSigmaPoints(6)
        self._filter = kf.UnscentedKalmanFilter(dim_x=4, dim_z=2, dt=None,
            fx=motion_CV, hx=lambda s: s[:2], points=sigma_points
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
        return np.array([self._filter.x[2], self._filter.x[3], 0])

    @property
    def velocity_var(self):
        cov = np.zeros((3, 3))
        cov[:2, :2] = self._filter.P[2:4, 2:4]
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
        s = np.average(np.sin(sigmas[:, 2]), weights=Wm)
        c = np.average(np.cos(sigmas[:, 2]), weights=Wm)
        x[2] = np.arctan2(s, c)
        return x

    def _state_diff(self, x, y):
        d = x - y
        d[2] = wrap_angle(d[2])
        return d

    def check_valid(self, note):
        if np.any(np.isnan(self._filter.x)):
            raise ValueError("nan occurs in states! (note: %s)" % note)
        if not is_pd(self._filter.P):
            newp = nearest_pd(self._filter.P)
            diff = npl.norm(self._filter.P - newp)
            message = "Covariance matrix is not positive definite, fixed with diff %.3f! (note: %s)" % (diff, note)
            if diff < 10:
                _logger.warning(message)
                warn(message)
            else:
                _logger.error(message)
                raise RuntimeError(message)
            self._filter.P = newp

    def __init__(self, init: ObjectTarget3D, Q=np.eye(6)):
        sigma_points = kf.JulierSigmaPoints(6)
        self._filter = kf.UnscentedKalmanFilter(dim_x=6, dim_z=3, dt=None,
            fx=motion_CTRA, hx=lambda s: s[:3], points=sigma_points,
            x_mean_fn=self._state_mean, z_mean_fn=self._state_mean,
            residual_x=self._state_diff, residual_z=self._state_diff,
        )
        self._filter.Q = np.asarray(Q).reshape(6, 6)

        # feed initial state
        yaw, pitch, roll = init.orientation.as_euler("ZYX")
        self._filter.x = np.array([init.position[0], init.position[1], yaw, 0, 0, 0])
        self._filter.P = np.copy(self._filter.Q)
        self._filter.P[:2, :2] = init.position_var[:2, :2]
        self._filter.P[2, 2] = init.orientation_var

        self._save_z = init.position[2] 
        self._save_z_var = init.position_var[2, 2]
        self._save_pitch = pitch # TODO: use simple bayesian filter for pitch and roll values
        self._save_roll = roll
        self.check_valid("initialize")

    def predict(self, dt):
        self._filter.predict(dt=dt)
        self.check_valid("prediction")

    def update(self, detection: ObjectTarget3D):
        yaw, pitch, roll = detection.orientation.as_euler("ZYX")
        self._save_z = detection.position[2]
        self._save_z_var = detection.position_var[2, 2]
        self._save_pitch = pitch
        self._save_roll = roll
        self._save_ori_var = detection.orientation_var

        obsv = np.array([detection.position[0], detection.position[1], yaw])
        R = np.zeros((3, 3))
        R[:2, :2] = detection.position_var[:2, :2]
        R[2, 2] = detection.orientation_var
        self._filter.update(obsv, R=R)
        self._filter.x[2] = wrap_angle(self._filter.x[2])
        self.check_valid("update")

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
        return Rotation.from_euler("ZYX",
            [self._filter.x[2], self._save_pitch, self._save_roll])

    @property
    def orientation_var(self):
        return self._save_z_var

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
        cov = np.zeros((3, 3))
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
    def angular_velocity_var(self):
        return np.diag([0, 0, self._filter.P[5, 5]])

class Pose_IMM:
    '''
    UKF using IMM (BR + CV + CA + CTRV + CTRA) model for pose estimation
    '''
    def __init__(self):
        raise NotImplementedError()
