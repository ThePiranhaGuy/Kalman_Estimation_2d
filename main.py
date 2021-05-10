import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import math

x_init = list(map(float, input().split(",")))

calc_path = [[x_init[0]], [x_init[1]]]
meas_path = [[x_init[0]], [x_init[1]]]
t = 1  # time floaterval taken constant

# state transition matrix F
F = np.array([[1, t, 0.5*t*t, 0, 0, 0],
              [0, 1, t, 0, 0, 0],
              [0, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, t, 0.5*t*t],
              [0, 0, 0, 0, 1, t],
              [0, 0, 0, 0, 0, 1]], dtype=np.float64)

# estimate uncertainity matrix
P = np.diag([100,25,0,100,25,0])
# p = diag(var(x),var(x_v),var(x_a),var(y),var(y_v),var(y_a))

# Process noise matrix
Q = np.array([[t**4/4, t**3/3, t**2/2, 0, 0, 0],
              [t**3/2, t**2, t, 0, 0, 0],
              [t**2/2, t, 1, 0, 0, 0],
              [0, 0, 0, t**4/4, t**3/3, t**2/2],
              [0, 0, 0, t**3/2, t**2, t],
              [0, 0, 0, t**2/2, t, 1]], dtype=np.float64)
sigma_a = 10  # random value really. not really measurable due to large noise/errors
Q = Q * (sigma_a**2)
# observation matrix

H = np.array([[1, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0]], dtype=int)

z = np.array([[0],
              [0]], dtype=np.float64)

# measurement COV uncertainity matrix

R = np.diag([0.25, 1000])
# R= diag(sigma_x_m,sigma_y_m)

# begin
x_meas = np.array([[0], [0], [0], [0], [0], [0]])
x_meas[0,0] = x_init[0]
x_meas[3,0] = x_init[1]
sigma2_x = sigma2_y = 0

#predict
x_pred = F.dot(x_meas)
P = np.add(np.dot(np.dot(F, P), F.T), Q)

# iterations
flag = 0
while(flag == 0):
    

    # measure
    try:
        x_meas = list(map(float, input().split(",")))
        temp = x_meas[1]
        x_meas[1] = x_meas[2]
        x_meas[2] = temp
        x_meas.insert(2, x_meas[1]/2*t)
        x_meas.insert(5, x_meas[4]/2*t)
        x_meas = np.array(x_meas)
        x_meas = np.transpose([x_meas])

    except EOFError:
        flag = 1
        break
    z = np.dot(H,x_meas)  
    #update
    k = np.dot(np.dot(P, H.transpose()), inv( np.add(np.dot(np.dot(H, P), H.transpose()), R)))  # kalman gain
    x_est = np.add(x_pred, np.dot(k, np.subtract(z, np.dot(H, x_pred))))
    m = np.subtract(np.identity(6,dtype=np.float64), np.dot(k, H))
    P = np.add(np.dot(np.dot(m, P), m.T),np.dot(np.dot(k, R), k.T))

    sigma2_x = P[0][0]
    sigma2_y = P[3][3]

    # predict
    x_pred = F.dot(x_est)
    P = np.add(np.dot(np.dot(F, P), F.T), Q)


    # store
    calc_path[0].append(x_est[0][0])
    calc_path[1].append(x_est[3][0])
    meas_path[0].append(z[0][0])
    meas_path[1].append(z[1][0])

# plotting
plt.plot(calc_path[0], calc_path[1], label='Estimated Path')
plt.plot(meas_path[0], meas_path[1], label='Measured Path')
plt.legend()
plt.title("Path")
plt.show()
print(f"final pos = {x_est[0,0]},{x_est[3,0]}")
print(f"uncerainity = {math.sqrt(sigma2_x)},{math.sqrt(sigma2_y)}")
