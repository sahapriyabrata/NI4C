import numpy as np
from sympy import symbols
from sympy.physics import mechanics
from sympy import Dummy, lambdify

class nLinkPendulum:
    def __init__(self, n, lengths=None, masses=1):
        #-------------------------------------------------
        # Step 1: construct the pendulum model
    
        # Generalized coordinates and velocities
        # (in this case, angular positions & velocities of each mass) 
        q = mechanics.dynamicsymbols('q:{0}'.format(n))
        u = mechanics.dynamicsymbols('u:{0}'.format(n))
        f = mechanics.dynamicsymbols('f:{0}'.format(n))

        # mass and length
        m = symbols('m:{0}'.format(n))
        l = symbols('l:{0}'.format(n))

        # gravity and time symbols
        g, t = symbols('g,t')
    
        #--------------------------------------------------
        # Step 2: build the model using Kane's Method

        # Create pivot point reference frame
        A = mechanics.ReferenceFrame('A')
        P = mechanics.Point('P')
        P.set_vel(A, 0)

        # lists to hold particles, forces, and kinetic ODEs
        # for each pendulum in the chain
        particles = []
        forces = []
        kinetic_odes = []

        for i in range(n):
            # Create a reference frame following the i^th mass
            Ai = A.orientnew('A' + str(i), 'Axis', [q[i], A.z])
            Ai.set_ang_vel(A, u[i] * A.z)

            # Create a point in this reference frame
            Pi = P.locatenew('P' + str(i), l[i] * Ai.x)
            Pi.v2pt_theory(P, A, Ai)

            # Create a new particle of mass m[i] at this point
            Pai = mechanics.Particle('Pa' + str(i), Pi, m[i])
            particles.append(Pai)

            # Set forces & compute kinematic ODE
            forces.append((Pi, m[i] * g * A.x))
        
            # Add external torque:
            forces.append((Ai, f[i] * A.z))
        
            kinetic_odes.append(q[i].diff(t) - u[i])

            P = Pi

        # Generate equations of motion
        KM = mechanics.KanesMethod(A, q_ind=q, u_ind=u,
                                   kd_eqs=kinetic_odes)
        fr, fr_star = KM.kanes_equations(particles, forces)
        
        #-----------------------------------------------------
        # Step 3: numerically evaluate equations
        
        # lengths and masses
        if lengths is None:
            lengths = np.ones(n) / n
        lengths = np.broadcast_to(lengths, n)
        masses = np.broadcast_to(masses, n)

        # Fixed parameters: gravitational constant, lengths, and masses
        parameters = [g] + list(l) + list(m)
        parameter_vals = [9.81] + list(lengths) + list(masses)

        # define symbols for unknown parameters
        unknowns = [Dummy() for i in q + u + f]
        unknown_dict = dict(zip(q + u + f, unknowns))
        kds = KM.kindiffdict()

        # substitute unknown symbols for qdot terms
        mm_sym = KM.mass_matrix_full.subs(kds).subs(unknown_dict)
        fo_sym = KM.forcing_full.subs(kds).subs(unknown_dict)

        # create functions for numerical calculation 
        self.mm_func = lambdify(unknowns + parameters, mm_sym)
        self.fo_func = lambdify(unknowns + parameters, fo_sym)
        
        self.args = parameter_vals

        A, B, _ = KM.linearize(A_and_B=True)
        parameter_dict = dict(zip(parameters, parameter_vals))
        self.A = A.subs(parameter_dict)
        self.B = B.subs(parameter_dict)
        self.state = q + u
    
    # function which computes the derivatives of parameters
    def gradient(self, y, t, u=None):
        if u is None:
            u = np.zeros(y.shape[0]//2)
        vals = np.concatenate((y, u, self.args))
        sol = np.linalg.solve(self.mm_func(*vals), self.fo_func(*vals))
        return np.array(sol).T[0]        
