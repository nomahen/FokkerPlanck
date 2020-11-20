Code for both the Fokker-Planck equation and the test simulations is contained in fp_test.ipynb. Running it requires the files field, spatial, and timesteppers.

To run the code for a given set of initial conditions, you will need: 

    grid_x, a uniform non-periodic grid
    grid_y, a uniform non-periodic grid

    IC = a numpy array
	
	domain = Domain([grid_x, grid_y])
    x,y = domain.values()
	p = Field(domain)
    X = FieldSystem([p])

    p.data[:] = np.copy(IC)

    mu, an Array of size domain - contains the drift coefficients (there may be up to two of these) 
    D_arr, an Array of size domain - contains the diffusion coefficients (there may be up to four of these)

    mu_i = [mu, mu] - a list containing the drift coefficient arrays associated with each of the first derivative operators [dx, dy].
    D_ij = [[D_arr, D_arr], - a list containing the diffusion coefficient corresponding to the second derivatives:  [dxx, dxdy, dydx, dyy ]
            [D_arr, D_arr]]
	    
Note: the elements of the mu_i and D_ij lists are allowed to be different from one another.
			
    diff = FokkerPlanck_2D(X,mu_i,D_ij)

    dt, a float
	tmax, a float
	
To run, execute:

    while diff.t < tmax:
        diff.step(dt)
		
We ran three simulations:

1). Wiener process.

2). Ornstein-Uhlenbeck Process

3). 2D Gaussian initial conditions

For the first two simulations, in order to compare to the analytic solutions, we created initial conditions which were constant in the y-direction
and only examined the output with respect to x. We created initial conditions by calculating the analytical results of evolving delta functions,
and then passed the output into our FP-solver as new initial conditions. We tested these for varying resolutions from N = 16 to N = 256, and
 plotted one row in the x-dimension against the results from the analytic solutions. To analyze the convergence of these equations, we calculated
the absolute error with respect to the analytical solution for each resolution.

Finally,  we constructed a two-dimensional Gaussian for the initial conditions, and constructed non-linear drift and diffusion coefficients to test
the Fokker-Planck equation worked in two dimensions. We plotted the results as a heatmap in two dimensions. Since no analytical solution was available, we tested for convergence using the same method as above but comparing the outputs to our highest resolution simulation.
