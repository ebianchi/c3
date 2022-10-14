import osqp
import numpy as np
from scipy import sparse
from scipy.linalg import block_diag
from scipy import linalg
from lemke_lcp import*
import timeit



def QP_function(x0, delta, omega, N,G,A,B,D,d,E,c,F,H):
    
    # system_parameters
    n = 10
    m = 10
    k = 4
    TOT = N*(n+m+k) + n

    
    # cost matrix
    Q = np.eye(n)
    Q[4,4] = 115
    Q[2,2] = 100
    Q[0,0] = 100
    Q[6,6] = 100
    Q[8,8] = 100
    
    S = np.zeros((m,m))
    R = 0.001*np.eye(k)
    QN = Q

    # setup for quadratic cost
    Csetup_initX = np.zeros((n,n))
    Csetup_initLAM = S
    Csetup_initU = R
    Csetup_init = block_diag(Csetup_initX, Csetup_initLAM, Csetup_initU)
    
    Csetup_reg = block_diag(Q,  S, R)
    
    Csetup_end = QN
    
    C = Csetup_init
    
    qf_init = np.zeros((n+m+k,1))
    x_reg = [[0],[0],[np.sqrt(2)],[0],[np.pi/4],[0],[0.9],[0],[0.9],[0]]
    x_reg = np.asarray(x_reg)
    qf_reg = np.vstack(( x_reg, np.zeros((m+k,1))))
    qf = qf_init
    
    for i in range(N-1):
        C = block_diag(C, Csetup_reg)
        qf = np.vstack((qf, qf_reg    ))
    
    C = block_diag(C, Csetup_end)
    qf = np.vstack((qf, x_reg))
    
    #scaling because of OSQP structure
    C = 2*C
    
    qf_add = - qf.T @ C
    
    #add the position it should converge
    
    
    #setup for ADMM cost
    #G = np.eye(n+m+k)
    
    #for cartpole
    #asd = 0.1 * np.eye(n+m+k)
    #asd[n+m+k-1,n+m+k-1] = 0
    Gsetup = G #Burada Gsetup = asd idi
    
    for i in range(N-1):
        Gsetup = block_diag(Gsetup,G)
        
    Gsetup = block_diag(Gsetup, np.zeros((n,n)))
    
    #scaling because of OSQP structure
    Gsetup = 2 * Gsetup
    
    #DEGISECEKLER
    #rho = np.zeros((N*(n+k+m) + n,1))
    #omega = np.zeros((N*(n+k+m) + n,1))
    cc = delta-omega
    #cc[0:4] = np.zeros((4,1))
    
    
    #LINEAR COST (DIVIDE By 2 BECAUSE MULTIPLIED EARLIER)
    q = - cc.T @ Gsetup + qf_add
    
    #QUADRATIC COST
    P = C + Gsetup
    
    #DYNAMIC CONSTRAINTS
    dyn_init1 = np.eye(n)
    dyn_init2 = np.zeros((n, TOT-n))
    
    dyn_init = np.hstack((dyn_init1, dyn_init2))
    
    
    dyn_reg = np.hstack((A, D, B, -np.eye(n)))
    dyn_size = np.size(dyn_reg,1)
    dyn_shift = n+m+k
    
    dyn = np.zeros((N*n, TOT))
    
    for i in range(N):
        dyn[n*i:n*i+n, dyn_shift*i:dyn_shift*i + dyn_size  ] = dyn_reg
        
    dyn = np.vstack((dyn_init, dyn))
    
    eq = x0

    
    for i in range(N):
        eq = np.vstack((eq, -d))
    
    # Create an OSQP object
    prob = osqp.OSQP()
    sP = sparse.csr_matrix(P)
    sdyn = sparse.csr_matrix(dyn) 
    
    # Setup workspace and change alpha parameter
    prob.setup(P = sP, q = q.T, A = sdyn, l = eq, u = eq, verbose = False) #time_limit = 0.1)
    
    starttime = timeit.default_timer()
    
    # Solve problem
    res = prob.solve()
    
    t_diff_qp = timeit.default_timer() - starttime
    
    sol = res.x
    
    return sol, t_diff_qp
