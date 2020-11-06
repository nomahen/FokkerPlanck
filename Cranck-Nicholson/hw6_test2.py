
import pytest
import numpy as np
import field
import spatial
import timesteppers
import equations

error_burgers = {(50,0.5):2.5e-3, (50,0.25):2e-3, (50,0.125):2e-3,(100,0.5):2e-4, (100,0.25):5e-5, (100,0.125):3e-5, (200,0.5):4e-5, (200,0.25):1e-5, (200,0.125):2e-6}
@pytest.mark.parametrize('resolution', [50, 100, 200])
@pytest.mark.parametrize('alpha', [0.5, 0.25, 0.125])
def test_viscous_burgers(resolution, alpha):
    grid_x = field.UniformPeriodicGrid(resolution, 20)
    grid_y = field.UniformPeriodicGrid(resolution, 20)
    domain = field.Domain((grid_x, grid_y))
    x, y = domain.values()

    r = np.sqrt((x-10)**2 + (y-10)**2)
    IC = np.exp(-r**2/4)

    u = field.Field(domain)
    v = field.Field(domain)
    X = field.FieldSystem([u, v])
    u.data[:] = IC
    v.data[:] = IC
    nu = 0.1

    burgers_problem = equations.ViscousBurgers2D(X, nu, 8)

    dt = alpha*grid_x.dx

    while burgers_problem.t < 10-1e-5:
        burgers_problem.step(dt)

    solution = np.loadtxt('u_burgers_%i.dat' %resolution)
    error = np.max(np.abs(solution - u.data))

    error_est = error_burgers[(resolution,alpha)]

    assert error < error_est


