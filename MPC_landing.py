
from time import time, sleep
import casadi as ca
import numpy as np
from casadi import sin, cos, pi
import matplotlib.pyplot as plt
import math
import cv2
from picamera2 import Picamera2
import matplotlib.pyplot as plt
import yaml
import numpy as np

from pymavlink import mavutil
from time import time


master = mavutil.mavlink_connection('/dev/serial0', baud=921600)
master.wait_heartbeat()
print("Heartbeat from system (system %u component %u)" % (master.target_system, master.target_component))


# read camera matrix
with open('calibration.yaml') as f:
    loadeddict = yaml.safe_load(f)
mtx = loadeddict.get('camera_matrix')
dist = loadeddict.get('dist_coeff')
mtx = np.array(mtx)
dist = np.array(dist)

# pycam
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": 'BGR888', 'size': (640, 480)},
    buffer_count=1))
# picam2.start()

# aruco
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

frame_count = 0
save_interval = 10
marker_size = 40
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()

# from simulation_code import simulate

# delta_t = 0.05 # sampling time [s]
mass = 1
theta_max = pi/6
theta_min = -theta_max
phi_max = pi/6
phi_min = -phi_max
T_max = 12
T_min = 8
psi_max = pi/4
psi_min = -psi_max

x1 = ca.SX.sym('x1')
x2 = ca.SX.sym('x2')
x3 = ca.SX.sym('x3')
x4 = ca.SX.sym('x4')
x5 = ca.SX.sym('x5')
x6 = ca.SX.sym('x6')
states = ca.vertcat(x1, x2, x3, x4, x5, x6)
n_states = states.numel()

theta = ca.SX.sym('theta')
phi = ca.SX.sym('phi')
psi = ca.SX.sym('psi')
T = ca.SX.sym('T')

controls = ca.vertcat(theta, phi, psi, T)
n_controls = controls.numel()

# setting matrix_weights' variables
Q_x = 1
Q_y = 1
Q_z = 1
R_theta = 1/theta_max
R_phi = 1/phi_max
R_psi = 1/psi_max
R_T = 1/(T_max + 100)

step_horizon = 0.1  # time between steps in seconds
N = 20           # number of look ahead steps
sim_time = 10     # simulation time

# specs

grav = 9.81

P_kalman = np.identity(6)
Q_kalman = 0.001*np.identity(6)
R_kalman = 100*np.identity(6)

def Pose_estimation(img, marker_size, mtx, dist, detector):
    # img = picam2.capture_array(wait=True)
    corners, ids, rejected = detector.detectMarkers(img)
    if ids is not None:
        ret = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, mtx, dist)
        if ret:
            #-- Unpack the output, get only the first
            rvec, tvec = ret[0][0,0,:], ret[1][0,0,:]
            # print("position = ", tvec, "Orientation = ", rvec)
            # cv2.drawFrameAxes(img, mtx, dist, rvec, tvec, 80)
        # rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)
        # for (markerCorner, markerID) in zip(corners, ids):
        #     # extract the marker corners (which are always returned in
        #     # top-left, top-right, bottom-right, and bottom-left order)
        #     corners = markerCorner.reshape((4, 2))
        #     (topLeft, topRight, bottomRight, bottomLeft) = corners

        #     # convert each of the (x, y)-coordinate pairs to integers
        #     topRight = (int(topRight[0]), int(topRight[1]))
        #     bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
        #     bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
        #     topLeft = (int(topLeft[0]), int(topLeft[1]))

        #     # draw the bounding box of the ArUCo detection
        #     cv2.line(img, topLeft, topRight, (0, 255, 0), 2)
        #     cv2.line(img, topRight, bottomRight, (0, 255, 0), 2)
        #     cv2.line(img, bottomRight, bottomLeft, (0, 255, 0), 2)
        #     cv2.line(img, bottomLeft, topLeft, (0, 255, 0), 2)

        #     # compute and draw the center (x, y)-coordinates of the ArUco marker
        #     cX = int((topLeft[0] + bottomRight[0]) / 2.0)
        #     cY = int((topLeft[1] + bottomRight[1]) / 2.0)
        #     cv2.circle(img, (cX, cY), 4, (0, 0, 255), -1)

            # draw the ArUco marker ID on the image
            # cv2.putText(img, str(markerID),
            #             (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
            #             0.5, (0, 255, 0), 2)
            # print("[INFO] ArUco marker ID: {}".format(markerID))

    # display the output image using matplotlib
        ax.clear()
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis('off')  # Hide axes
        plt.pause(0.001)  # Pause to allow the plot to update
    else:
        print("Aruco ID not found")
        rvec = None
        tvec = None
        # rvec = []
        # tvec = []
    return rvec, tvec



def shift_timestep(step_horizon, t0, state_init, u, f):
    f_value = f(state_init, u[:, 0])
    next_state = ca.DM.full(state_init + (step_horizon * f_value))

    t0 = t0 + step_horizon
    u0 = ca.horzcat(
        u[:, 1:],
        ca.reshape(u[:, -1], -1, 1)
    )

    return t0, next_state, u0



# matrix containing all states over all time steps +1 (each column is a state vector)
X = ca.SX.sym('X', n_states, N + 1)

# matrix containing all control actions over all time steps (each column is an action vector)
U = ca.SX.sym('U', n_controls, N)

# coloumn vector for storing initial state and target state
P = ca.SX.sym('P', n_states + n_states)

# state weights matrix (Q_X, Q_Y, Q_THETA)
Q = ca.diagcat(Q_x, Q_y, Q_z)

# controls weights matrix
R = ca.diagcat(R_theta, R_phi, R_psi, R_T)

def getDerivative_fsm(delta_t, m, x1, x2, x3, x4, x5, x6, theta, phi, psi, T):
    grav = 9.81
    r1 = x2
    r2 = -1/m*(cos(phi)*sin(theta)*cos(psi) + sin(phi)*sin(psi))*T
    r3 = x4
    r4 = -1/m*(cos(phi)*sin(theta)*sin(psi) - sin(phi)*cos(psi))*T
    r5 = x6
    r6 = grav - 1/m*cos(phi)*cos(theta)*T
    rhs = ca.vertcat(r1, r2,  r3, r4, r5, r6)
    return rhs

def DM2Arr(dm):
    return np.array(dm.full())

# discretization model (e.g. x2 = f(x1, v, t) = x1 + v * dt)
rot_3d_z = ca.vertcat(
    ca.horzcat(cos(theta), -sin(theta), 0),
    ca.horzcat(sin(theta),  cos(theta), 0),
    ca.horzcat(         0,           0, 1)
)
# Mecanum wheel transfer function which can be found here:
# https://www.researchgate.net/publication/334319114_Model_Predictive_Control_for_a_Mecanum-wheeled_robot_in_Dynamical_Environments

# RHS = states + J @ controls * step_horizon  # Euler discretization
RHS = getDerivative_fsm(step_horizon, mass, x1, x2, x3, x4, x5, x6, theta, phi, psi, T)
# maps controls from [va, vb, vc, vd].T to [vx, vy, omega].T
f = ca.Function('f', [states, controls], [RHS])


cost_fn = 0  # cost function
g = X[:, 0] - P[:n_states]  # constraints in the equation


# runge kutta
for k in range(N):
    st = X[:, k]
    con = U[:, k]
    state_vec = ca.vertcat(st[0] - P[6], st[2] - P[8], st[4] - P[10])
    cost_fn = cost_fn \
        + state_vec.T @ Q @ state_vec \
        + con.T @ R @ con
    st_next = X[:, k+1]
    f_value = f(st, con)
    # k1 = f(st, con)
    # k2 = f(st + step_horizon/2*k1, con)
    # k3 = f(st + step_horizon/2*k2, con)
    # k4 = f(st + step_horizon * k3, con)
    #
    st_next_euler = st + (step_horizon * f_value)
    g = ca.vertcat(g, st_next - st_next_euler)


OPT_variables = ca.vertcat(
    X.reshape((-1, 1)),   # Example: 3x11 ---> 33x1 where 3=states, 11=N+1
    U.reshape((-1, 1))
)
nlp_prob = {
    'f': cost_fn,
    'x': OPT_variables,
    'g': g,
    'p': P
}

opts = {
    'ipopt': {
        'max_iter': 2000,
        'print_level': 0,
        'acceptable_tol': 1e-8,
        'acceptable_obj_change_tol': 1e-6
    },
    'print_time': 0
}

solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

lbx = ca.DM.zeros((n_states*(N+1) + n_controls*N, 1))
ubx = ca.DM.zeros((n_states*(N+1) + n_controls*N, 1))

# lbx[0: n_states*(N+1): n_states] = -ca.inf     # X lower bound
# lbx[1: n_states*(N+1): n_states] = -ca.inf     # Y lower bound
# lbx[2: n_states*(N+1): n_states] = -ca.inf     # theta lower bound
#
# ubx[0: n_states*(N+1): n_states] = ca.inf      # X upper bound
# ubx[1: n_states*(N+1): n_states] = ca.inf      # Y upper bound
# ubx[2: n_states*(N+1): n_states] = ca.inf      # theta upper bound

# lbx[n_states*(N+1):] = v_min                  # v lower bound for all V
# ubx[n_states*(N+1):] = v_max                  # v upper bound for all V

lbx[0: n_states*(N+1): n_states] = -1 #state x lower bound
ubx[0: n_states*(N+1): n_states] = 10 #state x upper bound
lbx[1: n_states*(N+1): n_states] = -2 #state x_dot lower bound
ubx[1: n_states*(N+1): n_states] = 2 #state x_dot upper bound
lbx[2: n_states*(N+1): n_states] = -1 #state y lower bound
ubx[2: n_states*(N+1): n_states] = 10 #state y upper bound
lbx[3: n_states*(N+1): n_states] = -2 #state y_dot lower bound
ubx[3: n_states*(N+1): n_states] = 2 #state y_dot upper bound
lbx[4: n_states*(N+1): n_states] = -11 #state z lower bound
ubx[4: n_states*(N+1): n_states] = 0 #state z upper bound
lbx[5: n_states*(N+1): n_states] = -2 #state z_dot lower bound
ubx[5: n_states*(N+1): n_states] = 2 #state z_dot upper bound

lbx[n_states*(N+1): n_states*(N+1) + n_controls*N: n_controls] = theta_min #theta lower bound
ubx[n_states*(N+1): n_states*(N+1) + n_controls*N: n_controls] = theta_max #theta upper bound
lbx[n_states*(N+1)+1: n_states*(N+1) + n_controls*N: n_controls] = phi_min #phi lower bound
ubx[n_states*(N+1)+1: n_states*(N+1) + n_controls*N: n_controls] = phi_max #phi upper bound
lbx[n_states*(N+1)+2: n_states*(N+1) + n_controls*N: n_controls] = psi_min #psi lower bound
ubx[n_states*(N+1)+2: n_states*(N+1) + n_controls*N: n_controls] = psi_max #psi upper bound
lbx[n_states*(N+1)+3: n_states*(N+1) + n_controls*N: n_controls] = T_min #T lower bound
ubx[n_states*(N+1)+3: n_states*(N+1) + n_controls*N: n_controls] = T_max #T upper bound


args = {
    'lbg': ca.DM.zeros((n_states*(N+1), 1)),  # constraints lower bound
    'ubg': ca.DM.zeros((n_states*(N+1), 1)),  # constraints upper bound
    'lbx': lbx,
    'ubx': ubx
}

t0 = 0


# picam2.stop()

# x_init = 0
# y_init = 0
# z_init = 0
xdot_init = 0
ydot_init = 0
zdot_init = 0


landing_offset = 140 #mm
x_target = 0
y_target = 0
z_target = 0 + landing_offset/1000
xdot_target = 0
ydot_target = 0
zdot_target = 0

state_target = ca.DM([x_target, xdot_target, y_target, xdot_target, z_target, xdot_target])  # target state

# xx = DM(state_init)
t = ca.DM(t0)

u0 = ca.DM.zeros((n_controls, N))  # initial control


mpc_iter = 0
cat_controls = DM2Arr(u0[:, 0])
times = np.array([[0]])

picam2.start()
copter_mode_exe = 0
while copter_mode_exe == 0:
    msg = master.recv_match(type = 'HEARTBEAT', blocking = True)
    mode = mavutil.mode_string_v10(msg)
    print(mode)
    img = picam2.capture_array(wait=True)
    rvec, tvec = Pose_estimation(img, marker_size, mtx, dist, detector)
    if (tvec is not None) : #and (mode == 'GUIDED')
        x_init = tvec[0]/1000
        y_init = tvec[1]/1000
        z_init = tvec[2]/1000
        MPC_flag = 1
        copter_mode_exe = 1
        print("Guided mode initialized - initial position found")
        print("============initial position=====  ", "x=", x_init, "y=", y_init, "z=", z_init)
    else:
        MPC_flag = 0
        print("Guided mode failed/ Initial position not detected")

# print("============initial position=====  ", "x=", x_init, "y=", y_init, "z=", z_init)
init_condition = np.array([x_init, xdot_init, y_init, ydot_init, z_init, zdot_init])
state_init = ca.DM([x_init, xdot_init, y_init, ydot_init, z_init, zdot_init])        # initial state
X0 = ca.repmat(state_init, 1, N+1)         # initial state full
cat_states = DM2Arr(X0)
##############################################################################

xx = init_condition.flatten()

MPC_exit = 0
if __name__ == '__main__':
    main_loop_stime = time()  # return time in sec
    while (ca.norm_2(state_init - state_target) > 0.01) and (mpc_iter * step_horizon < sim_time) and (tvec is not None) and (MPC_exit == 0):
        start_time = time()
        img = picam2.capture_array(wait=True)
        rvec, tvec = Pose_estimation(img, marker_size, mtx, dist, detector)
        if (tvec is not None) and (tvec is not None):
            x_init = tvec[0]/1000
            y_init = tvec[1]/1000
            z_init = tvec[2]/1000
            state_init = ca.DM([x_init, xdot_init, y_init, ydot_init, z_init, zdot_init])        # initial state
        args['p'] = ca.vertcat(
            state_init,    # current state
            state_target   # target state
        )
        # optimization variable current state
        args['x0'] = ca.vertcat(
            ca.reshape(X0, n_states*(N+1), 1),
            ca.reshape(u0, n_controls*N, 1)
        )
        t1 = time()
        sol = solver(
            x0=args['x0'],
            lbx=args['lbx'],
            ubx=args['ubx'],
            lbg=args['lbg'],
            ubg=args['ubg'],
            p=args['p']
        )
        t2 = time()
        timeToSolve = t2 - t1
        print("time to solve NLP:", timeToSolve)
        u = ca.reshape(sol['x'][n_states * (N + 1):], n_controls, N)
        X0 = ca.reshape(sol['x'][: n_states * (N+1)], n_states, N+1)

        cat_states = np.dstack((
            cat_states,
            DM2Arr(X0)
        ))

        cat_controls = np.vstack((
            cat_controls,
            DM2Arr(u[:, 0])
        ))
        t = np.vstack((
            t,
            t0
        ))

        theta_c = u[0, 0]
        phi_c = u[1, 0]
        psi_c = u[2, 0]
        T_c = u[3, 0]

        rhs_eval = getDerivative_fsm(step_horizon, mass, state_init[0], state_init[1], state_init[2], state_init[3],
                                     state_init[4], state_init[5], theta_c, phi_c, psi_c, T_c)

        x_p = state_init + step_horizon * rhs_eval
        P_p_kalman = A * P_kalman * A.T + Q_kalman
        K_kalman = P_p_kalman * H.T * np.linalg.inv(H * P_kalman * H.T + R_kalman)
        z = state_init
        x_est = x_p + np.dot(K_kalman, (z - np.dot(H, x_p)))
        P = P_p_kalman - K_kalman * H * P_p_kalman
        P_kalman = P

        master.mav.command_long_send(master.target_system, master.target_component,
                                    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
        msg = master.recv_match(type = 'COMMAND_ACK', blocking=True)

        master.mav.send(mavutil.mavlink.MAVLink_set_position_target_local_ned_message(
            10, # time in ms (not used)
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED, # reference frame
            int(0b110111111000), #type mask (pos only)
            -x_est[0], x_est[2], x_est[4], x_est[1], x_est[3], x_est[5], 0, 0, 0, 0, 0 #commands
        ))
        # X, Y, Z, Xdot, Ydot, Zdot
        t0, state_init, u0 = shift_timestep(step_horizon, t0, x_est, u, f)


        # t0, state_init, u0 = shift_timestep(step_horizon, t0, state_init, u, f)
        xx = np.vstack((xx, state_init.flatten()))
        if mpc_iter == 0:
            val = DM2Arr(u[:, 0])
            uu = val.flatten()
        else:
            val = DM2Arr(u[:, 0])
            val2 = val.flatten()
            uu = np.vstack((uu, val2))


        # print(X0)
        X0 = ca.horzcat(
            X0[:, 1:],
            ca.reshape(X0[:, -1], -1, 1)
        )

        # xx ...
        t2 = time()
        # print(mpc_iter)
        # print(t2-t1)
        # print('Error = ', ca.norm_2(state_init - state_target))
        times = np.vstack(( times, t2-t1))

        # state_target = ca.DM([x_target, xdot_target, y_target, xdot_target, z_target, xdot_target])  # target state

        mpc_iter = mpc_iter + 1
        elapsed_time = time() - start_time
        time_to_sleep = 0.4 - elapsed_time
        if time_to_sleep > 0:
            sleep(time_to_sleep)
    main_loop_etime = time()
    ss_error = ca.norm_2(state_init - state_target)

    print('\n\n')
    print('Total time: ', main_loop_etime - main_loop_stime)
    print('avg iteration time: ', np.array(times).mean() * 1000, 'ms')
    print('final error: ', ss_error)

    fig1, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6)

    # Plot data on the first subplot
    ax1.plot(t.flatten().T, xx[:, 0])
    ax1.set_ylabel('x')  # Set y-axis label for the first subplot
    ax2.plot(t.flatten().T, xx[:, 1])
    ax2.set_ylabel('xdot')  # Set y-axis label for the first subplot
    ax3.plot(t.flatten().T, xx[:, 2])
    ax3.set_ylabel('y')  # Set y-axis label for the first subplot
    ax4.plot(t.flatten().T, xx[:, 3])
    ax4.set_ylabel('ydot')  # Set y-axis label for the first subplot
    ax5.plot(t.flatten().T, xx[:, 4])
    ax5.set_ylabel('z')  # Set y-axis label for the first subplot
    ax6.plot(t.flatten().T, xx[:, 5])
    ax6.set_ylabel('zdot')  # Set y-axis label for the first subplot
    plt.savefig('Evolution_of_states.png')

    fig2, (bx1, bx2, bx3, bx4) = plt.subplots(4)
    bx1.plot(t[1:].flatten().T, uu[:, 0])
    bx1.set_ylabel('theta_c')  # Set y-axis label for the first subplot
    bx2.plot(t[1:].flatten().T, uu[:, 1])
    bx2.set_ylabel('phi_c')  # Set y-axis label for the first subplot
    bx3.plot(t[1:].flatten().T, uu[:, 2])
    bx3.set_ylabel('psi_c')  # Set y-axis label for the first subplot
    bx4.plot(t[1:].flatten().T, uu[:, 3])
    bx4.set_ylabel('T_c')  # Set y-axis label for the first subplot

    plt.savefig('Evolution_of_controls.png')


