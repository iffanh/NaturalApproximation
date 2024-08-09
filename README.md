We will try the following ideas:

1. Use quadratic interpolation model where the gradient and the hessian are approximated using the ensemble (see Zhao et al. 2013, Fonseca et al., 2016)

$$m(x + p) = J(x) + g^T p + 1/2 p^T H p$$

In Zhang et al. 2023, the gradient can be computed as (Equation 8 and Equation 9):

$$g \approx D_\mu L = 1/N \sum_{i=1}^N J(X^i) (X^i - \mu)$$
$$H \approx D_\Sigma L = 1/N \sum_{i=1}^N J(X^i) ( (X^i - \mu)(X^i - \mu) - \Sigma )$$

TR-ens algorihm:

Step 0: Initialization
Set $k = 0$. Set maximum number of iterations. Set covariance step size $\beta_0$
Given an initial point x_0 and an initial trust-region radius $\Delta_0$
Let $0 < \eta_1 \leq \eta_2 < 1$, $0 < \gamma_1 < 1$, and $\gamma_2 > 1$

Step 1: Sampling.
Sample N i.i.d {X^i_k} rnadom variables from $N(x|\mu_k , \Sigma_k)$

Step 2: Calculate the gradients
Calculate $D_\mu L$ and $D_\Sigma L$

Step 3: Step calculation.
Let $g_k = D_\mu L$ and $H_k = \D_Sigma L$ 
Solve $p_k$

Step 4: Acceptance of the trial point.
Compute $J(\mu_k + pk)$ and define

$\rho_k = \frac{(J(\mu_k) - J(\mu_k + p_k))}{(m_k(\mu_k) - m_k(\mu_k + p_k))}$
If  $\rho_k \geq \eta_1$, then define $\mu_{k+1} = \mu_k + p_k$
Else $\mu_{k+1} = \mu_k$. Exit.

Step 5: Update the covariance matrix
$\Sigma_{k+1} = \Sigma_k + \beta_k D_\Sigma L$

Step 6: Trust-region radius update
Set 

$\Delta_{k+1} = 
max[\gamma_2 \| p_k \|_k, \Delta_k] if \rho_k \geq \eta_2
\Delta_k if \rho_k \in (\eta_1, \eta_2)
\gamma_1 \| p_k \|_k if \rho_k \leq \eta_1$
