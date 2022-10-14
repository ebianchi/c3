import numpy as np

def Evaluate_system(x, dt):
    """Systems are approximated linearly as:

        x_{k+1} = A * x_k  +  B * u_k  +  D * lambda_k  +  d
        E * x_k  + F * lambda_k  +  H * u_k  +  c  >=  0

    This linearization is made about the current state, x.  Thus, this
    function takes in the current state, x, and returns the linearization of
    the system's dynamics in the form of A, B, D, d, E, F, H, and c.

    In this example, there are 3 contacts, 2 inputs each of 2 dimensions, and
    5 state positions (x, y, alpha of the block, and distances of each input
    contact from the center of their respective faces).  The inputs are 2
    accelerations of the contact point locations (ddot_l_1 and ddot_l_2) and
    contact normal forces (force_1 and force_2) acting at those points.
    """

    mu1 = 0.1
    mu2 = 0.1
    mu3 = 1
    g = 9.81
    h = 1
    w = 1
    m = 1
    
    # rt is the length between the COM and the corner
    rt = np.sqrt(h*h + w*w)

    # interpret the state vector
    alpha = x[4]    # the angle w.r.t. the ground
    f1 = x[6]       # distance of contact point 1 from center of its side
    f2 = x[8]       # distance of contact point 2 from center of its side
    
    # get the sine and cosine of alpha
    sin = np.sin(alpha)
    sin = sin[0]
    cos = np.cos(alpha)
    cos = cos[0]
    
    # z is the difference in heights of contact point 1 from contact point 2
    # z = z_2 - z_1
    z = w*sin - h*cos
    

    # Calculate A.
    """A is just a double integrator."""
    A = np.array([[1,dt,0, 0,0, 0,0, 0,0, 0],
                  [0, 1,0, 0,0, 0,0, 0,0, 0],
                  [0, 0,1,dt,0, 0,0, 0,0, 0],
                  [0, 0,0, 1,0, 0,0, 0,0, 0],
                  [0, 0,0, 0,1,dt,0, 0,0, 0],
                  [0, 0,0, 0,0, 1,0, 0,0, 0],
                  [0, 0,0, 0,0, 0,1,dt,0, 0],
                  [0, 0,0, 0,0, 0,0, 1,0, 0],
                  [0, 0,0, 0,0, 0,0, 0,1,dt],
                  [0, 0,0, 0,0, 0,0, 0,0, 1]])
    

    # Calculate B.
    """B maps inputs to changes in state.  These values are derived in Bibit's
    sketch.  Remember the inputs are [ddot_l_1, ddot_l_2, force_1, force_2].

    Note:  the below technically needs the mass to appear in the denominator in
    in a few places, but since m=1, this is inconsequential.
    """
    B = np.array([[    0,     0, -cos * dt**2, sin * dt**2],
                  [    0,     0,    -cos * dt,    sin * dt],
                  [    0,     0,  sin * dt**2, cos * dt**2],
                  [    0,     0,     sin * dt,    cos * dt],
                  [    0,     0,  -f1 * dt**2,  f2 * dt**2],
                  [    0,     0,     -f1 * dt,     f2 * dt],
                  [dt**2,     0,            0,           0],
                  [   dt,     0,            0,           0],
                  [    0, dt**2,            0,           0],
                  [    0,    dt,            0,           0]])
    

    # Calculate D.
    """D maps complementarity variable values to changes in state.  For this
    example, there are 10 complementarity variables corresponding to two input
    contacts which are always in contact but can slip in 2 directions or stick
    (2 contacts * 3 modes) and the ground contact which can make and break
    contact, plus slip in 2 directions (4 more modes).

    These complementarity variables are:
        - force_1
        - tangential force at point 1, in -f1 direction
        - tangential force at point 1, in +f1 direction
        - force_2
        - tangential force at point 2, in -f2 direction
        - tangential force at point 2, in +f2 direction
        - lack of normal force at ground (sort of like -y direction force)
        - tangential force at ground, in +x direction
        - tangential force at ground, in -x direction
        - normal force at ground, in +y direction
    """
    D = [[0, dt**2 *sin, -dt**2 *sin, 0, -cos*dt**2 , dt**2 *cos,0,dt**2 ,-dt**2 ,0],
         [0, dt*sin, -dt*sin, 0, -cos*dt, dt*cos,0,dt,-dt,0],
         [0,dt**2 *cos,-dt**2 *cos,0,dt**2 *sin,-dt**2 *sin,0,0,0,dt**2 ],
         [0,dt*cos,-dt*cos,0,dt**2 *sin,-dt*sin,0,0,0,dt],
         [0, -dt**2 *h, dt**2 *h,0,dt**2 *w,-dt**2 *w,0,-dt**2 *cos*w-dt**2 *sin*h,dt**2 *cos*w+dt**2 *sin*h,dt**2 *sin*w-h*cos*dt**2 ],
         [0, -dt*h, dt*h,0,dt*w,-dt*w,0,-dt*cos*w-dt*sin*h,dt*cos*w+dt*sin*h,dt*sin*w-h*cos*dt],
         [0, -dt**2, dt**2, 0,0,0,0,0,0,0],
         [0,    -dt,    dt, 0,0,0,0,0,0,0],
         [0,0,0,0,-dt**2 ,dt**2 ,0,0,0,0],
         [0,0,0,0,-dt,dt,0,0,0,0]]
    D = np.asarray(D)
    

    # Calculate d.
    """The only contribution to d is gravity.  These values are derived in
    Bibit's sketch.
    
    Note:  I think these mass appearances are false, but since m=1, this is
    inconsequential.
    """
    d = np.array([[0],[0],[-m*g * dt**2], [-dt*m*g], [0],[0],[0],[0],[0],[0]])
    

    E = [[0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,-1,0,0],
         [0,0,0,0,0,0,0,1,0,0],
         [0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,-1],
         [0,0,0,0,0,0,0,0,0,1],
         [0,0,0,0,0,0,0,0,0,0],
         [0,-1,0,0,0,-rt,0,0,0,0],
         [0,1,0,0,0,rt,0,0,0,0],
         [0,0,1,dt,-h*sin+w*cos+z,z*dt,0,0,0,0]]
    E = np.asarray(E)
    
    c = np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0],[-h*cos-w*sin-dt**2 *m*g]])
    
    F = [[0,-1,-1,0,0,0,0,0,0,0],
         [1,dt,-dt,0,0,0,0,0,0,0],
         [1,-dt,dt,0,0,0,0,0,0,0],
         [0,0,0,0,-1,-1,0,0,0,0],
         [0,0,0,1,dt,-dt,0,0,0,0],
         [0,0,0,1,-dt,dt,0,0,0,0],
         [0,0,0,0,0,0,0,-1,-1,mu3],
         np.negative([0,dt*sin-rt*dt*h,-dt*sin+rt*dt*h,0,-dt*cos+rt*dt*w,dt*cos-dt*rt*w,-1,dt-rt*dt*cos*w-rt*dt*sin*h,-dt+rt*dt*cos*w+rt*dt*sin*h,sin*w*rt*dt - h*cos*rt*dt]),
         np.negative([0,-dt*sin+rt*dt*h,dt*sin-rt*dt*h,0,dt*cos-rt*dt*w,-dt*cos+rt*dt*w,-1,-dt+rt*dt*cos*w+rt*dt*sin*h,dt-rt*dt*cos*w-rt*dt*sin*h,-sin*w*rt*dt + h*cos*rt*dt]),
         [0,dt**2 *cos-z*dt**2 *h,-dt**2 *cos+z*dt**2 *h,0,dt**2 *sin+z*dt**2 *w,-dt**2 *sin-dt**2 *w*z,0,-z*dt**2 *cos*w-z*dt**2 *sin*h,z*dt**2 *cos*w+z*dt**2 *sin*h,dt**2 +z*dt**2 *sin*w-z*h*cos*dt**2 ]]
    F = np.asarray(F)
    
    x5 = f1
    x5 = x5[0]
    
    x7 = f2
    x7 = x7[0]
    
    H = [[0,0,mu1,0],
         [-dt,0,0,0],
         [dt,0,0,0],
         [0,0,0,mu2],
         [0,-dt,0,0],
         [0,dt,0,0],
         [0,0,0,0],
         np.negative([0,0,-dt*cos-rt*dt*x5,dt*sin+dt*rt*x7]),
         np.negative([0,0,dt*cos+rt*dt*x5,-dt*sin-dt*rt*x7]),
         [0,0,dt**2 *sin-dt**2 *x5*z,dt**2 *cos+dt**2 *x7*z ]]
    H = np.asarray(H)
    
    return A,B,D,d,E,c,F,H

