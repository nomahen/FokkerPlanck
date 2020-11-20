To run the code, you will need: 

    grid_x, a uniform non-periodic grid
    grid_y, a uniform non-periodic grid

    IC = a numpy array
	
	domain = Domain([grid_x, grid_y])
    x,y = domain.values()
	p = Field(domain)
    X = FieldSystem([p])

    p.data[:] = np.copy(IC)

	mu, an Array of size domain - contains the drift coefficients
    D_arr, an Array of size domain - contains the diffusion coefficients

    mu_i = [mu, mu]
    D_ij = [[D_arr, D_arr],
            [D_arr, D_arr]]
			
    diff = FokkerPlanck_2D(X,mu_i,D_ij)

    dt, a float
	tmax, a float
	
To run, execute:

    while diff.t < tmax:
        diff.step(dt)
