from numpy import *
import time
from filter import *
import matplotlib.pyplot as plt


def integrate(val, dt):
	return array([sum(val[0:i]) for i in range(0, len(val))])*dt

data_num = 2000
mu = 0
sigma = 0.5

a_figure = plt.figure(figsize=(25,15))
a_axes = a_figure.add_subplot(1,1,1)
v_figure = plt.figure(figsize=(25,15))
v_axes = v_figure.add_subplot(1,1,1)
x_figure = plt.figure(figsize=(25,15))
x_axes = x_figure.add_subplot(1,1,1)

x = linspace(0, 2*pi, data_num)
dt = 2*pi / data_num

a_trueval = sin(x)
v_trueval = integrate(a_trueval, dt)
x_trueval = integrate(v_trueval, dt)

random.seed(int(time.time()))
a_measured = a_trueval + random.normal(mu, sigma, data_num)
v_measured = integrate(a_measured, dt)
x_measured = integrate(v_measured, dt)

# Kalman Filter
F = matrix([[1, dt, (dt**2)/2], [0, 1, dt], [0, 0, 1]])

H = matrix([[0.0, 0.0, 1.0]])
Q = matrix([[0,0,0],[0,0,0],[0,0,0.001]])
R = matrix([[0.1]])

x_0 = matrix([[0], [0], [a_measured[0]]])
P_0 = matrix([[0,0,0],[0,0,0],[0,0,sigma**2]])

filter_KF = KF(F, H, Q, R, x_0, P_0)

a_KF = [0.0]*data_num
v_KF = [0.0]*data_num
x_KF = [0.0]*data_num

a_KF[0] = x_0.item(2,0)
v_KF[0] = x_0.item(1,0)
x_KF[0] = x_0.item(0,0)

for i in range(1, data_num):
	_x = filter_KF.update(a_measured[i])
	a_KF[i] = _x.item(2,0)
	v_KF[i] = _x.item(1,0)
	x_KF[i] = _x.item(0,0)

# Low Pass Filter
a_LPF = [0.0]*data_num
filter_LPF = LPF(0.9, a_measured[0])
for i in range(1, data_num):
	a_LPF[i] = filter_LPF.update(a_measured[i])
v_LPF = integrate(a_LPF, dt)
x_LPF = integrate(v_LPF, dt)


a_axes.plot(x, a_trueval, "blue")
a_axes.plot(x, a_measured, "green")
a_axes.plot(x, a_KF, "red")
a_axes.plot(x, a_LPF, "yellow")

v_axes.plot(x, v_trueval, "blue")
v_axes.plot(x, v_measured, "green")
v_axes.plot(x, v_KF, "red")
v_axes.plot(x, v_LPF, "yellow")

x_axes.plot(x, x_trueval, "blue")
x_axes.plot(x, x_measured, "green")
x_axes.plot(x, x_KF, "red")
x_axes.plot(x, x_LPF, "yellow")

plt.show()
