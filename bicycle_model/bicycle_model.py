#!/usr/bin/env python3

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import numpy as np
import numpy.linalg
import argparse
import yaml


class LTISystem:
    """Class representing a LTI system given in state-space representation where A := system matrix, B := input matrix, C := output matrix, D := feedtrough matrix."""
    
    def __init__(self, A, B, C, D):
        """Initializes a LTI system."""

        self.__taylor_coeff = [1.0,
                               1.0,
                               0.5,
                               0.16666666666,
                               0.04166666666,
                               0.00833333333,
                               0.00138888888,
                               0.00019841269,
                               0.00002480158,
                               0.00000275573,
                               0.00000027557]
        x_dim = A.shape[1]
        u_dim = B.shape[1]
        assert(A.shape[0] == x_dim) # domain-dim has to be state-dim
        self.__A = A
        assert(B.shape[0] == x_dim) # domain-dim has to be state-dim
        self.__B = B
        assert(C.shape[1] == x_dim) # codomain-dim has to be state-dim
        self.__C = C
        assert(D.shape[0] == u_dim) # domain-dim has to be control-dim
        assert(D.shape[1] == x_dim) # codomain-dim has to be state-dim
        self.__D = D

    def discretize(self, delta_t, taylor_order=10):
        """Discretizes system matrix and intput matrix up to the 10th taylor order."""

        A_d = np.zeros(self.__A.shape)
        discretization_factor_B = np.zeros(self.__A.shape)
        delta_t_pow = 1.0
        A_pow = np.identity(self.__A.shape[0])
        for i in range(0, taylor_order+1):
            A_d = A_d + (self.__taylor_coeff[i]*A_pow*delta_t_pow)
            delta_t_pow = delta_t*delta_t_pow
            if (i < taylor_order):
                discretization_factor_B = discretization_factor_B + (self.__taylor_coeff[i+1]*A_pow*delta_t_pow)
            A_pow = self.__A*A_pow
        B_d = discretization_factor_B*self.__B

        return A_d, B_d

    def calculate_trajectory(self, A_d, B_d, x, u, num_steps):
        """Calculates the phase space state trajectory given an initial state and control input."""

        x_trajectory = x
        x_i = x
        for i in range(0, num_steps):
            x_i = (A_d*x_i) + (B_d*u)
            x_trajectory = np.hstack((x_trajectory, x_i))

        return x_trajectory

    def calculate_equilibrium(self, u):
        """Calculates the LTI system equilibrium. Returns 'None' if there is no equilibrium or infinitely many."""
        
        x_eq = None
        try:
            x_eq = np.linalg.solve(self.__A, -self.__B*u)
        except numpy.linalg.LinAlgError:
            print("There is no equilibrium or infinitely many!")

        return x_eq

    def calculate_equilibrium_as_function_of_u(self, u_bounds, u_ticks):
        """Calculates the LTI system equilibrium as a function of the control input. Returns 'None' if there is at least once no equilibrium or infinitely many."""

        init = False
        for u in np.arange(u_bounds[0, 0], u_bounds[0, 1], (u_bounds[0, 1]-u_bounds[0, 0])/u_ticks):
            x_eq = self.calculate_equilibrium(u)
            if x_eq is None:
                return None
            if init is False:
                x_eq_of_u = x_eq
                init = True
            else:
                x_eq_of_u = np.hstack((x_eq_of_u, x_eq))

        return x_eq_of_u

    def calculate_transfer_function_poles(self):
        """Calculates the LTI system transfer function."""

        # TODO recheck: does this only hold when system is controllable and observable?
        w, v = np.linalg.eig(self.__A)

        return w

    def generate_vector_field(self, u, xi, xj):
        """Generates the phase space vector field over two state elements."""

        K_u = self.__B*u
        xi_dot = self.__A[0, 0]*xi+self.__A[0, 1]*xj+K_u[0, 0]
        xj_dot = self.__A[1, 0]*xi+self.__A[1, 1]*xj+K_u[1, 0]

        return xi_dot, xj_dot


class BicycleModel(LTISystem):
    """Class representing the bicycle model."""

    def __init__(self, m, v, theta, c_f, c_r, l_f, l_r):
        """Initializes the bicycle model from its system parameters."""

        A = np.matrix([[-(c_f+c_r)/(m*v), -(m*v*v+(c_r*l_r-c_f*l_f))/(m*v*v)],
                      [(c_r*l_r-c_f*l_f)/theta, -(c_r*l_r*l_r+c_f*l_f*l_f)/(theta*v)]])
        B = np.matrix([[c_f/(m*v)], [(c_f*l_f)/theta]])
        C = np.matrix([[1.0, 0.0]])
        D = np.matrix([[0.0, 0.0]])
        LTISystem.__init__(self, A, B, C, D)


class BicycleModelApp(tk.Frame):
    def __init__(self, master, config):
        tk.Frame.__init__(self, master)
        self.master = master

        # set title
        master.title("Bicycle Model")

        # init system parameters
        self.__m = config["m_init"]
        self.__v = config["v_init"]
        self.__theta = config["theta_init"]
        self.__c_f = config["c_f_init"]
        self.__c_r = config["c_r_init"]
        self.__l_f = config["l_f_init"]
        self.__l_r = config["l_r_init"]

        # init state and control input
        self.__x = np.matrix([[config["slip_angle_init"]], [config["yaw_rate_init"]]])
        self.__u = np.matrix([[config["steering_angle_init"]]])

        # init number of prediction steps
        self.__num_steps = int(config["num_steps_init"])
        self.__delta_t = config["delta_t"]

        # init other parameters
        self.__x_bounds = np.matrix([config["slip_angle_bounds"], config["yaw_rate_bounds"]])
        self.__u_bounds = np.matrix([config["steering_angle_bounds"]])
        self.__x_ticks = int(config["x_ticks"])
        self.__u_ticks = int(config["u_ticks"])

        # init figure
        self.__f = plt.Figure(figsize=(15, 5))

        # calculate data for the plots
        self.__xi = None
        self.__xj = None
        self.__xi_dot = None
        self.__xj_dot = None
        self.__time = None
        self.__x_eq = None
        self.__x_eq_of_u = None
        self.__x_trajectory = None
        self.__poles = None
        self.recalculate_plot_data()

        # draw plots on figure
        self.__phase_space_plot = None
        self.__trajectory_plot = None
        self.__trajectory_plot_start = None
        self.__trajectory_plot_end = None
        self.__trajectory_plot_0 = None
        self.__trajectory_plot_1 = None
        self.__laplace_0 = None
        self.__laplace_1 = None
        self.__x_eq_plot = None
        self.__x_eq_of_u_plot = None
        self.create_figure(self.__f)

        # integrate figure in GUI
        self.__plots = FigureCanvasTkAgg(self.__f, master)
        self.__plots.get_tk_widget().pack(side=tk.LEFT)

        # add animation (in order for the plots to be responsive wrt the slider positions)
        self.__anim = FuncAnimation(self.__f, self.refresh_plots, frames=self.__num_steps,
                             interval=1000*self.__delta_t, blit=True, repeat=True)


        # slider for prediction time
        self.create_slider(master, "prediction time [s]", [0, self.__delta_t*self.__num_steps],
                           self.__num_steps, self.adjust_num_steps, self.__delta_t*self.__num_steps)

        # slider for slip angle (state element)
        self.create_slider(master, "slip angle [rad]", [self.__x_bounds[0, 0], self.__x_bounds[0, 1]],
                           self.__x_ticks*10, self.adjust_xi_0, self.__x[0, 0])

        # slider for yaw rate angle (state element)
        self.create_slider(master, "yaw rate [rad/s]", [self.__x_bounds[1, 0], self.__x_bounds[1, 1]],
                           self.__x_ticks*10, self.adjust_xj_0, self.__x[1, 0])

        # slider for steering angle (control input)
        self.create_slider(master, "steering angle [rad]", [self.__u_bounds[0, 0], self.__u_bounds[0, 1]],
                           self.__u_ticks*10, self.adjust_u, self.__u[0, 0])

        # slider for mass (system parameter)
        self.create_slider(master, "mass [kg]", config["m_bounds"], 100, self.adjust_m, self.__m)

        # slider for velocity (system parameter)
        self.create_slider(master, "velocity [m/s]", config["v_bounds"], 100, self.adjust_v, self.__v)

        # slider for moment of inertia (system parameter)
        self.create_slider(master, "moment of inertia [kg m²]", config["theta_bounds"],
                           100, self.adjust_theta, self.__theta)

        # slider for cornering stiffness front (system parameter)
        self.create_slider(master, "cornering stiffness (f) [N/rad]",
                           config["c_f_bounds"], 100, self.adjust_c_f, self.__c_f)

        # slider for cornering stiffness rear (system parameter)
        self.create_slider(master, "cornering stiffness (r) [N/rad]",
                           config["c_r_bounds"], 100, self.adjust_c_r, self.__c_r)

        # slider for front axis to COG (system parameter)
        self.create_slider(master, "front axis to COG [m]",
                           config["l_f_bounds"], 100, self.adjust_l_f, self.__l_f)

        # slider for rear axis to COG (system parameter)
        self.create_slider(master, "rear axis to COG [m]",
                           config["l_r_bounds"], 100, self.adjust_l_r, self.__l_r)

    def recalculate_plot_data(self):
        # init bicycle modle with current parameters
        bicycle_model = BicycleModel(self.__m, self.__v, self.__theta, self.__c_f, self.__c_r, self.__l_f, self.__l_r)

        # equilibrium
        self.__x_eq = bicycle_model.calculate_equilibrium(self.__u)
        self.__x_eq_of_u = bicycle_model.calculate_equilibrium_as_function_of_u(self.__u_bounds, self.__u_ticks)

        # phase space vector field
        self.__xi, self.__xj = np.meshgrid(np.linspace(self.__x_bounds[0, 0], self.__x_bounds[0, 1], self.__x_ticks), np.linspace(
            self.__x_bounds[1, 0], self.__x_bounds[1, 1], self.__x_ticks))
        self.__xi_dot, self.__xj_dot = bicycle_model.generate_vector_field(self.__u, self.__xi, self.__xj)

        # trajectory
        A_d, B_d = bicycle_model.discretize(self.__delta_t)
        self.__time = self.__delta_t * np.arange(0, self.__num_steps+1, 1)
        self.__x_trajectory = bicycle_model.calculate_trajectory(A_d, B_d, self.__x, self.__u, self.__num_steps)

        # laplace of state element step-responses
        # self.__poles = []
        # if self.__x[0] != 0.0 or self.__x[1] != 0.0 or self.__u!= 0.0:
        #     self.__poles = calculate_transfer_function_poles(A).tolist()
        self.__poles = bicycle_model.calculate_transfer_function_poles().tolist()
        if self.__u != 0.0:
            self.__poles.append(0.0+0.0j)

    def create_figure(self, f):
        plt.suptitle("State-Space Representation")
        gs = f.add_gridspec(6, 12)
        ax_phase_space = f.add_subplot(gs[0: 6, 0: 5])
        ax_trajectory_0 = f.add_subplot(gs[0: 3, 5: 9])
        ax_trajectory_1 = f.add_subplot(gs[3: 6, 5: 9])
        ax_laplace_0 = f.add_subplot(gs[0: 3, 9: 12])
        ax_laplace_1 = f.add_subplot(gs[3: 6, 9: 12])

        self.__phase_space_plot = ax_phase_space.quiver(
            self.__xi, self.__xj, self.__xi_dot, self.__xj_dot, angles='xy', scale_units='xy', pivot='mid', zorder=2)
        ax_phase_space.set_xlim((self.__x_bounds[0, 0], self.__x_bounds[0, 1]))
        ax_phase_space.set_ylim((self.__x_bounds[1, 0], self.__x_bounds[1, 1]))
        ax_phase_space.set_xticks(self.__xi[0, :], minor=False)
        ax_phase_space.set_yticks(self.__xj[:, 0], minor=False)
        ax_phase_space.grid(which='major', zorder=1)
        ax_phase_space.set_xlabel('slip angle [rad]')
        ax_phase_space.set_ylabel('yaw rate [rad/s]')

        if self.__x_eq_of_u is not None:
            self.__x_eq_of_u_plot, = ax_phase_space.plot(
                self.__x_eq_of_u[0, :].T, self.__x_eq_of_u[1, :].T, alpha=0.5, color='black')
        else:
            self.__x_eq_of_u_plot, = ax_phase_space.plot([], [], alpha=0.5, color='black')
        self.__trajectory_plot, = ax_phase_space.plot(self.__x_trajectory[0, :].T, self.__x_trajectory[1, :].T, 'b-')
        self.__trajectory_plot_start, = ax_phase_space.plot(self.__x_trajectory[0, 0], self.__x_trajectory[1, 0], 'bo')
        self.__trajectory_plot_end, = ax_phase_space.plot(self.__x_trajectory[0, -1], self.__x_trajectory[1, -1], 'b+')
        if self.__x_eq is not None:
            self.__x_eq_plot, = ax_phase_space.plot(self.__x_eq[0, 0], self.__x_eq[1, 0], 'kx')
        else:
            self.__x_eq_plot, = ax_phase_space.plot([], [], 'kx')

        self.__trajectory_plot_0, = ax_trajectory_0.plot(self.__time, self.__x_trajectory[0, :].T, 'b')
        self.__trajectory_plot_1, = ax_trajectory_1.plot(self.__time, self.__x_trajectory[1, :].T, 'b')
        ax_trajectory_0.grid()
        ax_trajectory_0.set_ylabel('slip angle [rad]')
        ax_trajectory_0.set_xlabel('prediction time [s]')
        ax_trajectory_0.set_ylim((self.__x_bounds[0, 0], self.__x_bounds[0, 1]))
        ax_trajectory_1.grid()
        ax_trajectory_1.set_ylabel('yaw rate [rad/s]')
        ax_trajectory_1.set_xlabel('prediction time [s]')
        ax_trajectory_1.set_ylim((self.__x_bounds[1, 0], self.__x_bounds[1, 1]))

        ax_laplace_0.grid()
        ax_laplace_0.axhline(0, color='black')
        ax_laplace_0.axvline(0, color='black')
        ax_laplace_0.set_xlabel('Re(s)')
        ax_laplace_0.set_ylabel('Im(s) j')
        self.__laplace_0, = ax_laplace_0.plot(np.real(self.__poles), np.imag(self.__poles), 'bx')
        ax_laplace_1.grid()
        ax_laplace_1.axhline(0, color='black')
        ax_laplace_1.axvline(0, color='black')
        ax_laplace_1.set_xlabel('Re(s)')
        ax_laplace_1.set_ylabel('Im(s) j')
        self.__laplace_1, = ax_laplace_1.plot(np.real(self.__poles), np.imag(self.__poles), 'bx')

        f.tight_layout()

    def refresh_plots(self, i):
        # equilibrium
        if self.__x_eq_of_u is not None:
            self.__x_eq_of_u_plot.set_data(self.__x_eq_of_u[0, :].T, self.__x_eq_of_u[1, :].T)
        else:
            self.__x_eq_of_u_plot.set_data([], [])
        if self.__x_eq is not None:
            self.__x_eq_plot.set_data(self.__x_eq[0, 0], self.__x_eq[1, 0])
        else:
            self.__x_eq_plot.set_data([], [])

        # phase space vector field
        self.__phase_space_plot.set_UVC(self.__xi_dot, self.__xj_dot)

        # trajectory
        self.__trajectory_plot.set_data(self.__x_trajectory[0, :].T, self.__x_trajectory[1, :].T)
        self.__trajectory_plot_0.set_data(self.__time, self.__x_trajectory[0, :].T)
        self.__trajectory_plot_1.set_data(self.__time, self.__x_trajectory[1, :].T)
        self.__trajectory_plot_start.set_data(self.__x_trajectory[0, 0], self.__x_trajectory[1, 0])
        self.__trajectory_plot_end.set_data(self.__x_trajectory[0, -1], self.__x_trajectory[1, -1])

        # laplace of state element step-responses
        self.__laplace_0.set_data(np.real(self.__poles), np.imag(self.__poles))
        self.__laplace_1.set_data(np.real(self.__poles), np.imag(self.__poles))

        return (self.__phase_space_plot, self.__trajectory_plot, self.__trajectory_plot_start,
                self.__trajectory_plot_end, self.__trajectory_plot_0, self.__trajectory_plot_1, self.__laplace_0,
                self.__laplace_1, self.__x_eq_plot, self.__x_eq_of_u_plot)

    def create_slider(self, master, sl_name, sl_bounds, sl_steps, sl_cmd, sl_init_val, sl_length=200):
        sl_xj_0 = tk.Scale(master, label=sl_name, from_=sl_bounds[0], to=sl_bounds[1], resolution=(
            sl_bounds[1]-sl_bounds[0])/float(sl_steps), orient=tk.HORIZONTAL, command=sl_cmd, length=sl_length)
        sl_xj_0.set(sl_init_val)
        sl_xj_0.pack()

    def button_method(self):
        # run this when button click to close window
        self.master.destroy()

    def adjust_num_steps(self, val):
        self.__num_steps = int(float(val)/self.__delta_t)
        self.recalculate_plot_data()

    def adjust_xi_0(self, val):
        self.__x[0, 0] = float(val)
        self.recalculate_plot_data()

    def adjust_xj_0(self, val):
        self.__x[1, 0] = float(val)
        self.recalculate_plot_data()

    def adjust_u(self, val):
        self.__u[0, 0] = float(val)
        self.recalculate_plot_data()

    def adjust_m(self, val):
        self.__m = float(val)
        self.recalculate_plot_data()

    def adjust_v(self, val):
        self.__v = float(val)
        self.recalculate_plot_data()

    def adjust_theta(self, val):
        self.__theta = float(val)
        self.recalculate_plot_data()

    def adjust_c_f(self, val):
        self.__c_f = float(val)
        self.recalculate_plot_data()

    def adjust_c_r(self, val):
        self.__c_r = float(val)
        self.recalculate_plot_data()

    def adjust_l_f(self, val):
        self.__l_f = float(val)
        self.recalculate_plot_data()

    def adjust_l_r(self, val):
        self.__l_r = float(val)
        self.recalculate_plot_data()


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="directory to configuration yaml-file")
    args = parser.parse_args()

    # Read YAML file
    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)

    # launch app
    root = tk.Tk()
    root.configure(background='white')
    app = BicycleModelApp(root, config)
    root = tk.mainloop()
