import numpy as np
import scipy.stats

def approximate_gradient(u_curr:np.ndarray, J_curr:float, U: np.ndarray, J: np.ndarray) -> np.ndarray:
    """
    Equation (12) in Fonseca et al. (2016). For each ensemble member m_k
    we have the corresponding u_k and J_k 
    
    U : a vector of perturbed control, N_e x N_u
    J : a vector of result (objective / constraint), b = b(u_k, m_k), 
    """ 
    
    Ne = U.shape[0]
    
    g = 0
    for k in range(Ne): #iterate over the ensemble
        u_k = U[k,:]
        du = u_k - u_curr
        J_k = J[k]
        dJ = J_k - J_curr
        
        g_k = du*dJ
        g += g_k
    
    g = g/Ne
    
    return g

def generate_controls(u_curr:np.ndarray, s:np.ndarray, Ne:int) -> np.ndarray:
    """
    Generate matrix of controls of the size N_e x N_u
    
    u_curr : the mean, N_u
    s : the standard deviation, N_u
    """
    
    U = []
    
    for k in range(Ne):
        u_k = np.random.normal(u_curr, s)
        U.append(u_k)
        
    U = np.array(U)
    
    return U

def natural_gradient(J:np.ndarray, j:float, U:np.ndarray, u:np.ndarray):
    
    """
    Calculate the natural graident, Equation 8 in Zhang 2023
    J normalized with mean
    U : a matrix of perturbed control, N_e x N_u
    u : the mean of the control, N_u
    J : a vector of result (objective / constraint), b = b(u_k, m_k), 
    
    """
    
    N_e = J.shape[0]
    # J_mean = np.mean(J)
    
    g = 0
    for k in range(N_e):
        g += (J[k] - j)*(U[k,:] - u)
        
    g = g/N_e
    
    # g = g/np.linalg.norm(g)
    
    return g

def natural_Hessian(J:np.ndarray, j:float, U:np.ndarray, u:np.ndarray, S:np.ndarray):
    """
    Calculate the natural Hessian, Equation 9 in Zhang 2023
    J normalized with mean
    U : a matrix of perturbed control, N_e x N_u
    u : the mean of the control, N_u
    J : a vector of result (objective / constraint), b = b(u_k, m_k), 
    S : Covariance matrix of the control, N_u x N_u
    """
    
    N_e = J.shape[0]
    # J_mean = np.mean(J)
    
    H = 0
    for k in range(N_e):
        # print((np.outer((U[k,:] - u), (U[k,:] - u)) - S))
        H += (J[k] - j)*(np.outer((U[k,:] - u), (U[k,:] - u)) - S)
    H = H/N_e
    
    # H = H/np.linalg.norm(H)
    
    return H

def natural_covariance(U:np.ndarray, u:np.ndarray, S:np.ndarray):
    """
    Calculate the natural Hessian, Equation 9 in Zhang 2023
    J normalized with mean
    U : a matrix of perturbed control, N_e x N_u
    u : the mean of the control, N_u 
    S : Covariance matrix of the control, N_u x N_u
    """
    
    N_e = u.shape[0]
    
    C = 0
    for k in range(N_e):
        # print((np.outer((U[k,:] - u), (U[k,:] - u)) - S))
        C += (np.outer((U[k,:] - u), (U[k,:] - u)) - S)
    C = C/N_e
    
    return C

# def generate_natural_controls()