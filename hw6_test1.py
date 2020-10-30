
import pytest
import numpy as np
import field
import spatial
import timesteppers
import equations

error_RD = {(50,0.5):3e-3, (50,0.25):2.5e-3, (50,0.125):2.5e-3,(100,0.5):4e-4, (100,0.25):2e-4, (100,0.125):1e-4, (200,0.5):8e-5, (200,0.25):2e-5, (200,0.125):5e-6}
@pytest.mark.parametrize('resolution', [50, 100, 200])
@pytest.mark.parametrize('alpha', [0.5, 0.25, 0.125])
def test_reaction_diffusion(resolution, alpha):
    grid_x = field.UniformPeriodicGrid(resolution, 20)
    grid_y = field.UniformPeriodicGrid(resolution, 20)
    domain = field.Domain((grid_x, grid_y))
    x, y = domain.values()

    IC = np.exp(-(x+(y-10)**2-14)**2/8)*np.exp(-((x-10)**2+(y-10)**2)/10)

    c = field.Field(domain)
    X = field.FieldSystem([c])
    c.data[:] = IC
    D = 1e-2

    dcdx2 = spatial.FiniteDifferenceUniformGrid(2, 8, c, 0)
    dcdy2 = spatial.FiniteDifferenceUniformGrid(2, 8, c, 1)

    rd_problem = equations.ReactionDiffusion2D(X, D, dcdx2, dcdy2)

    dt = alpha*grid_x.dx

    while rd_problem.t < 1-1e-5:
        rd_problem.step(dt)

    solution = np.loadtxt('c_%i.dat' %resolution)
    error = np.max(np.abs(solution - c.data))

    error_est = error_RD[(resolution,alpha)]

    assert error < error_est


