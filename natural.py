import numpy as np
import approximation
import casadi as ca
import scipy.stats
import math
from steihaug import steihaug
from nearestPD import nearestPD

def build_model(J, g, H):
    ui = ca.SX.sym('u', g.shape[0])
    m = J + ca.mtimes(ui.T,g) + ca.mtimes(ca.mtimes(ui.T,H), ui)/2
    return ui, m

def find_step(u0, r, u, m):
    
    nlp = {
        'x': u,
        'f': m,
    }
    
    # tr radius as input bound      
    ubx = u0 + r 
    lbx = u0 - r
    
    opts = {'ipopt.print_level':2, 'print_time':0, 'ipopt.sb': 'yes', 
                'ipopt.honor_original_bounds': 'yes'}
    
    solver = ca.nlpsol("QCQP", "ipopt", nlp, opts)
    sol = solver(x0=u0, ubx=ubx, lbx = lbx)
    
    # print(solver.stats())
    
    solution = sol['x']
    
    # print(f"Step size = {u0 - solution}")
    
    return solution
    
def f(x:np.ndarray) -> float: 
    """
    Rosenbrock's equation
    """
    
    f = 0
    for i in range(x.shape[0] - 1):
        f += 100*(x[i+1] - x[i]**2)**2 + (x[i] - 1)**2
    
    return f

Ne = 10
Nu = 2

# set initial point
u0 = np.array([-1.5, 0.5])
# u0 = np.array([0., 0.])
j0 = f(u0)

# set initial covariance
s0 = np.array([1E-5]*2)
# s0 = np.array([0.01]*2)
S0 = np.diag(s0)

# set hyperparameters
b0 = 0.001
g1 = 0.9
g2 = 5
e1 = 0.25
e2 = 0.75
r0 = 3


rk = r0*1
bk = b0*1
uk = u0*1
sk = s0*1
jk = j0*1
Sk = np.diag(sk)
rho = 1
step = 0
gk = 0
Hk = 0

# main loop
Niter = 30000
for k in range(Niter):
    # step 1: sampling
    np.set_printoptions(precision=10)
    print(f"It:{k}, jk = {jk}, uk = {uk}, rk={rk}, step = {step}, rho={rho}, gk = {gk}, Hk={Hk}, sk={Sk}")
    # print(f"It:{k}, jk = {jk}, uk = {uk}, rk={rk}, step = {step}, rho={rho}")
    
    Sk = nearestPD(Sk)
    
    obj = scipy.stats.multivariate_normal(mean=uk, cov=Sk, allow_singular=True)
    Uk = obj.rvs(size=Ne)
    
    
    # step 2: calculate gradient
    # evaluate
    Jk = f(Uk.T)
    gk = approximation.natural_gradient(J=Jk, j=jk, U=Uk, u=uk)
    Hk = approximation.natural_Hessian(J=Jk, j=jk, U=Uk, u=uk, S=Sk)
    Ck = approximation.natural_covariance(U=Uk, u=uk, S=Sk)
    
    # step 3: step calculation
    ui, mk = build_model(jk, gk, Hk)    
    step = steihaug(gk, Hk, rk)
    uk1 = uk + step
    
    #create function for the model
    mkf = ca.Function(f'model_{k}', [ui], [mk])
    
    # step 4: acceptance of trial point
    # evaluate
    jk1 = f(uk1)
    
    m2 = mkf(uk)
    m1 = mkf(uk1)
    rho = (jk - jk1)/(m2 - m1)
    
    ill_condition = np.logical_and(np.logical_and(math.isclose(jk, jk1), math.isclose(m2,m1)),~np.isfinite(rho)) 
    
    if ill_condition:
        uk = uk*1
    
    else:
        if rho >= e1 : 
            uk_ = np.array(uk)
            uk = np.array(uk1)#[:,0]
            
            jk = jk1*1
            
        else:
            uk = uk*1
        
    # step 5: update the covariance matrix
    Sk = Sk + bk*Hk
    ## TODO: any heuristics to adjust bk so that Sk does not explode?
    
    # trust-region radius update
    if ill_condition:
        rk = np.max([g2*np.linalg.norm(uk - uk_), rk])
        
    else:
        if rho >= e2: 
            # rk = np.max([g2*np.linalg.norm(uk - uk_), rk])
            rk = rk*g2
        elif rho <= e1:
            rk = g1*np.linalg.norm(uk1 - uk)
            # rk = g1*rk
        else:
            # radius unchanged
            pass
        
    if np.linalg.norm(uk - np.array([1., 1.])) < 0.001:
        print(f"SUCCESS. uk = {uk}")
        break
    
    