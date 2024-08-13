import numpy as np

def steihaug(g:np.ndarray, B:np.ndarray, R:float):

    TOL = 1E-12
    
    p = 0
    z = np.zeros(g.shape)
    r = g
    d = -r
    
    if np.linalg.norm(r) < TOL:
        return p
    
    for j in range(10000):
        if np.matmul(np.matmul(d, B), d) <= 0:
            D = np.sqrt(4*np.matmul(d, z)**2 - 4*np.matmul(d, d)*(np.matmul(z, z) - R**2))
            tau1 = (-2*np.matmul(d, z) + D)/(2*np.matmul(d, d))
            tau2 = (-2*np.matmul(d, z) - D)/(2*np.matmul(d, d))
            
            p1 = z + tau1*d
            p2 = z + tau2*d
            
            if np.matmul(p1, g) + 0.5*np.matmul(np.matmul(p1, B), p1) > np.matmul(p2, g) + 0.5*np.matmul(np.matmul(p2, B), p2):
                return p2
            else:
                return p1
            
        a = np.matmul(r, r)/np.matmul(np.matmul(d, B), d)
        
        z_ = z + a*d
        
        if np.linalg.norm(z_) >= R:
            D = np.sqrt(4*np.matmul(d, z)**2 - 4*np.matmul(d, d)*(np.matmul(z, z) - R**2))
            tau1 = (-2*np.matmul(d, z) + D)/(2*np.matmul(d, d))
            tau2 = (-2*np.matmul(d, z) - D)/(2*np.matmul(d, d))
            
            p1 = z + tau1*d
            p2 = z + tau2*d
            
            if tau1 > 0:
                return p1
            elif tau2 > 0:
                return p2
            
        r_ = r + a*np.matmul(B, d)
        
        if np.linalg.norm(r_) < TOL:
            return z_
        
        b = np.matmul(r_, r_)/np.matmul(r, r)
        
        d_ = -r_ + b*d
        
        # increment
        d = d_*1
        r = r_*1
        z = z_*1
        p = z_*1
    
    
    raise ValueError(f"Steihaug step FAILED")
    # return z + a*d