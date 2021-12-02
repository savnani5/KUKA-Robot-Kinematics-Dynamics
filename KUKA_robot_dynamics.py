import math
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

class Kinematics:

    def __init__(self, robot, DH_table, ang_speed, symbols, radius, centre, theta_init, q_init):
        self.robot = robot
        self.DH_table = DH_table
        self.ang_speed = ang_speed
        self.symbols = symbols
        self.radius = radius
        self.centre = centre
        self.theat_init = theta_init
        self.q_init = q_init
        self.no_of_links = len(self.symbols)
        self.all_trans = []

    def __repr__(self):
        return f"Kinematics Class for {self.robot} robot"

    def forward(self):
        """Function to calculate the cumulative tranformation matrices for each joint
        with respect to the origin"""

        cum_trans = sp.eye(4)
        for i in range(7):
            Curr_trans = sp.Matrix([[sp.cos(self.DH_table[i,3]), -sp.sin(self.DH_table[i,3])*sp.cos(self.DH_table[i,0]), sp.sin(self.DH_table[i,3])*sp.sin(self.DH_table[i,0]), self.DH_table[i,1]*sp.cos(self.DH_table[i,3])],
                      [sp.sin(self.DH_table[i,3]), sp.cos(self.DH_table[i,3])*sp.cos(self.DH_table[i,0]), -sp.cos(self.DH_table[i,3])*sp.sin(self.DH_table[i,0]), self.DH_table[i,1]*sp.sin(self.DH_table[i,3])],
                      [0, sp.sin(self.DH_table[i,0]), sp.cos(self.DH_table[i,0]), self.DH_table[i,2]],
                      [0, 0, 0, 1]])
        
            Curr_trans = sp.nsimplify(Curr_trans, tolerance=1e-8, rational=True)
            cum_trans =  cum_trans*Curr_trans
            self.all_trans.append(cum_trans)
        return cum_trans 

    def inverse(self, cum_trans):
        """Function to calculate the Jacobian matrix from the tranformation matrices"""

        z = sp.ones(3, 6)
        o = sp.ones(3, 6)
        col = cum_trans[:-1,-1]
        for i, j in enumerate(col):
            for k, l in enumerate(self.symbols):
                z[i,k] = sp.diff(j, l)
                o[:,k] = self.all_trans[k][:-1,2]
        Jacobian = sp.Matrix.vstack(z, o)
        return Jacobian

    def traj_plot(self, X, Z):
        """Live plot of the end effector position"""

        plt.scatter(X, Z)
        plt.pause(0.01)

class Dynamics(Kinematics):
    
    def __init__(self, robot, DH_table, ang_speed, symbols, radius, centre, theta_init, q_init, mass, end_effector_force):
        super().__init__(robot, DH_table, ang_speed, symbols, radius, centre, theta_init, q_init)
        self.mass = mass
        self.end_effector_force = end_effector_force

    def __repr__(self):
        return f"Dynamics Class for {self.robot} robot"

    def PotentialEnergy(self):
        """Function for calculating the Potential Energies"""

        g = sp.zeros(self.no_of_links,1)
        P = sp.Matrix([0])
        
        link_pos = sp.zeros(3,self.no_of_links+1)
        link_cm = sp.zeros(3,self.no_of_links)
        
        for i in range(1,self.no_of_links+1):
            link_pos[:,i] = self.all_trans[i-1][:3,3]
            link_cm[:,i-1] = link_pos[:,i-1] + (link_pos[:,i] - link_pos[:,i-1])/2
        for i in range(self.no_of_links):
            P = P + self.mass[i] * sp.Matrix([0,0,-9.81]).T * link_cm[:,i]        
        for i in range(self.no_of_links):
            g[i] = sp.diff(P, self.symbols[i])

        return g

    def Joint_Torque(self, Jacobian, PE):
        """Function for calculating the Joint Torques"""

        J_torq = PE - Jacobian.T * self.end_effector_force
        return J_torq

    def torq_plot(self, samples, vals):
        """Function for plotting the Joint Torques"""
        
        time = np.arange(0, samples)
        fig, axis = plt.subplots(1, 6)

        J_torq_list = [[] for i in range(len(vals[0]))]
        
        for i in range(len(vals)):
            for j in range(len(J_torq_list)):
                J_torq_list[j].append(vals[i][j])
                
        for i in range(len(J_torq_list)):   
            axis[i].plot(time, J_torq_list[i])
            axis[i].set_title(f"Joint {i+1} Torque")
        
        fig.tight_layout(pad=0.3)
        plt.show()

def main(samples):

    # Variable initializations
    ttc = 200                       # Time to complete               
    ang_speed = (-2*math.pi)/ttc 
    symbols = sp.symbols('theta1, theta2, theta4, theta5, theta6, theta7')
    radius = 100
    centre = (0,605,680)
    theta_init = 0
    q_init = sp.Matrix([math.radians(90),(math.radians(0)), (math.radians(-90)),(math.radians(0)),(math.radians(0.01)),(math.radians(0))])
    mass = sp.Matrix([3.94, 4.50, 2.46 + 2.61, 3.41, 3.39, 0.354])
    EE_force = sp.Matrix([0,0,0,0,0,-5])

    # DH table for KUKA robot
    DH_table = sp.Matrix([[math.radians(-90), 0, 360, symbols[0]],
                    [math.radians(90), 0, 0, symbols[1]],
                    [math.radians(90), 0, 420, 0],
                    [math.radians(-90), 0, 0, symbols[2]],
                    [math.radians(-90), 0, 399.5, symbols[3]],
                    [math.radians(90), 0, 0, symbols[4]],
                    [0, 0, 205.5, symbols[5]]])

    delta_t = (2*math.pi)/samples
    q_curr = q_init
    theta = theta_init

    # Instance of the Dynamics class
    dynamics = Dynamics('KUKA', DH_table, ang_speed, symbols, radius, centre, theta_init, q_init, mass, EE_force)

    # Calculating the Forward kinematics
    T_final = dynamics.forward()

    # Calculating the inverse kinematics
    Jacobian = dynamics.inverse(T_final)
    # print("Jacobian", Jacobian)

    # Calculating Potential Energy Equations
    PE = dynamics.PotentialEnergy()
    print("g(q): \n", PE)
    
    # Calculating Joint Torque Equations
    J_torq = dynamics.Joint_Torque(Jacobian, PE)
    print("\n\nJoint Torques: \n", J_torq)

    # Figure Initialization
    figure = plt.figure(clear=False)
    
    # Torque Values
    vals = []

    #__________________________Main loop for trajectory tracking_____________________________
    for i in range(samples):
        
        q_prev = q_curr
        theta = delta_t*i
        
        # end effector position
        x = radius * sp.cos(math.radians(theta))
        z = radius * sp.sin(math.radians(theta)) 
        
        # end effector velocity
        vx = ang_speed * radius * sp.cos(theta)
        vz = ang_speed * radius * sp.sin(theta)
        vel_vector = sp.Matrix([vx,0,vz,0,0,0])
        
        # Inverse of jacobian 
        Jacobian = Jacobian.subs([(symbols[0], q_curr[0]),(symbols[1], q_curr[1]), (symbols[2], q_curr[2]),(symbols[3], q_curr[3]),(symbols[4], q_curr[4]),(symbols[5], q_curr[5])])
        J_torq_inv = J_torq.subs([(symbols[0], q_curr[0]),(symbols[1], q_curr[1]), (symbols[2], q_curr[2]),(symbols[3], q_curr[3]),(symbols[4], q_curr[4]),(symbols[5], q_curr[5])])
        
        J_inv = Jacobian.inv(method='LU')
        q_dot = J_inv*vel_vector 
        # print("\nJoint velocity vector", "=="*20, q_dot)
        
        # Joint angles update
        q_curr = q_prev + q_dot*delta_t
        Desired_pos = T_final.subs([(symbols[0], q_curr[0]),(symbols[1], q_curr[1]), (symbols[2], q_curr[2]),(symbols[3], q_curr[3]),(symbols[4], q_curr[4]),(symbols[5], q_curr[5])])
        
        # Plotting the point retrieved from the inverse Jacobian
        dynamics.traj_plot(Desired_pos[0,-1], Desired_pos[2,-1])

        # Storing the joint torque values
        vals.append(J_torq_inv)

    plt.show()

    # plotting the joint torques
    dynamics.torq_plot(samples, vals)
    

if __name__ == "__main__":
    main(50)