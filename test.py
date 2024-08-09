import numpy as np
import approximation


# def f(x:np.ndarray) -> float: 
#     """
#     Rosenbrock's equation
#     """
    
#     f = 0
#     for i in range(x.shape[0] - 1):
#         f += 100*(x[i+1] - x[i]**2)**2 + (x[i] - 1)**2
    
#     return f

def f(x:np.ndarray) -> float: 
    """
    Simple equation
    """
    
    f = 0
    for i in range(x.shape[0]):
        f += (x[i] - 1)**2
    
    return f


Ne = 100
Nu = 2
u0 = np.array([0, 0])
s0 = np.array([0.5, 0.5])

s = s0*1
u = u0*1
j = f(u)

# U = approximation.generate_controls(u_curr=u, s=s0, Ne=Ne)
Un = approximation.generate_controls(u_curr=np.array([0., 0.]), s=np.array([1., 1.]), Ne=Ne) #normalized controls
U = u + Un*s

u_curr = np.mean(U, axis=0)

jj = 0
Niter = 50
for kk in range(Niter):

    J = f(U.T)
    
    print(f"It. {kk}, u = {u}, f = {np.mean(J)}, backtracking = {jj}, s={s}")

    g = approximation.approximate_gradient(u_curr=u, J_curr=j, U=U, J=J)
    g = np.matmul(np.diag(s), g)
    gn = g / np.linalg.norm(g)
    # print(f"normalized gradient = {gn}")

    a = 1
    rho = 0.5
    c1 = 1.
    N_linesearch = 100
    
    # save line search results
    J_linesearch = []
    U_linesearch = []
    u_linesearch = []
    
    for jj in range(N_linesearch): #line search with maximum of 5 steps
        u_trial = u - c1*a*gn
        # print(s, s*gn)
        # U_trial = approximation.generate_controls(u_curr=u_trial, s=abs(s*a), Ne=Ne)
        U_trial = u_trial + Un*a
        
        J_trial = f(U_trial.T)
        
        J_bar = np.mean(J_trial)
        
        J_linesearch.append(J_bar)
        U_linesearch.append(U_trial)
        u_linesearch.append(u_trial)
        
        if J_bar < np.mean(J) + c1*a*np.inner(gn,u_trial - u): #wolfe condition
            U = U_trial*1
            u = u_trial*1

            # s = abs(s/rho)
            break
        
        else:
            a = a*rho
            
    if jj == N_linesearch - 1:
        # no better point was found
        # we pick u that gives the best J from the ensemble
        print(f"Back tracking failed")
        # min_index = int(np.where(J_linesearch == np.min(J_linesearch))[0][0])
        # U = U_linesearch[min_index]
        # u = u_linesearch[min_index]
        # s = abs(s*rho)
        
        # s = s*rho
        # U = approximation.generate_controls(u_curr=u, s=s*rho, Ne=Ne)
    
    

