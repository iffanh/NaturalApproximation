import numpy as np
import scipy.stats
from nearestPD import nearestPD

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


u0 = np.array([-1.5, 0.5])
j0 = f(u0)
s0 = np.array([1e-1]*2)

b1 = 1.0
b2 = 0.1

uk = u0*1
sk = s0*1
jk = j0*1
Sk = np.diag(sk)

Niter = 3000
TOL = 1E-5

Samples = []
Js = []
objs = []
ll = 0
wl = 0
var = 0
for k in range(Niter):
    
    # print(f"It:{k}, jk = {jk}, uk = {uk}, sk={Sk}, number of prev iteration={k - ll}, TOL = {TOL}")
    print(f"It:{k}, jk = {jk}, uk = {uk}, no. of prev it = {k - ll}, ll = {ll}, wl = {wl}, var = {var}")
    

    Sk = nearestPD(Sk)
    obj = scipy.stats.multivariate_normal(mean=uk, cov=Sk, allow_singular=True)
    
    Uk = obj.rvs(size=Ne)
    Jk = f(Uk.T)
    
    pdfk = obj.pdf(Uk)
    
    Samples.append(Uk)
    Js.append(Jk)
    objs.append(obj)
    
    # maximizing previous information
    
    if k > 0:
        var = 0
        w = 0
        wl = 0
        for l in list(range(k))[::-1]:
            # Ul = Samples[l]
            objl = objs[l]
            
            pdfj = objl.pdf(Uk)
            
            wjk = np.max(pdfk/pdfj)**2
            # print(wjk)
            w += wjk
            
            _var = np.abs((k - l)**2 + 2*(k - l) - w)
            # _var = (k - l)**2 + 2*(k - l) - w
            # print(_var)
            
            if _var > var:
                var = _var*1
                ll = l
                wl = w
                
        if wl > (k - ll)**2 + 2*(k - ll):
            ll = k*1
        
    else:
        # first time
        ll = 0
        
    
    # Calculate the gradient

    dmu = 0
    dSigma = 0    
    for j in range(ll, k+1):
        Uj = Samples[j]
        Phi = obj.pdf(Uj)/objs[ll].pdf(Uj)
        Wj = (1/(Ne*(k - ll + 1)))*(Js[k] - jk)*Phi

        for i in range(Ne):
            # dmu += Wj[i]*(Uj[i,:] - uk)
            dmu += np.matmul(np.linalg.pinv((np.outer(Uj[i,:] - uk, Uj[i,:] - uk) - Sk)),(Uj[i,:] - uk)) 
            dSigma += Wj[i]*(np.outer(Uj[i,:] - uk, Uj[i,:] - uk) - Sk) 
    
    # maximizing previous information
    
    # ll = k
        
    
    # # Calculate the gradient

    # dmu = 0
    # dSigma = 0    
    # for j in range(ll, k+1):
    #     Uj = Samples[j]
    #     Phi = obj.pdf(Uj)/objs[ll].pdf(Uj)
    #     Wj = (1/(Ne*(k - ll + 1)))*(Js[k] - jk)*Phi

    #     for i in range(Ne):
    #         # dmu += Wj[i]*(Uj[i,:] - uk)
    #         dmu += np.matmul(np.linalg.pinv((np.outer(Uj[i,:] - uk, Uj[i,:] - uk) - Sk)),(Uj[i,:] - uk)) 
    #         dSigma += Wj[i]*(np.outer(Uj[i,:] - uk, Uj[i,:] - uk) - Sk) 
        
    
    if 10*b2*np.linalg.norm(dSigma) > np.linalg.norm(Sk):
        dSigma = dSigma*np.linalg.norm(Sk)/np.linalg.norm(dSigma)    
    
    # Next step
    uk_ = uk*1
    Sk_ = Sk + b2*dSigma
    
    for i in range(5):
        uk_ = uk + b1*dmu

        jk_ = f(uk_)
        # if jk_ < jk:
        if jk_ - jk < TOL:
            print(f"i, uk - uk_ = {i}, {uk - uk_}")
            uk = uk_
            Sk = Sk_
            jk = jk_
            b1 = 1.0
            b2 = 0.001
            TOL = TOL/2
            break 
        else:
            b1 = b1/2
            # b2 = b2/2
            
    
    if np.linalg.norm(uk - np.array([1., 1.])) < 0.001:
        print(f"SUCCESS. uk = {uk}")
        break
    
    