# KL: This is from Naim
@njit("(f8[:, :, :], f4[:, :], f8, f8, i8, i8)")
def _paintCicIntoGrid(

        density_grid, particle_coords, d_xy, d_z, n_grid_xy, n_grid_z

):
    """Reverse interpolate particles onto a Cartesian `density_grid`.

    Note I allow grid dimensions to be different in xy and z directions.

    Typically, people use same dimensions.

    Arguments

    ---------

    density_grid (np.ndarray): 3D grid of shape

        (n_grid_xy, n_grid_xy, n_grid_z). This array is updated in place.

    particle_coords (np.ndarray): 2D array of shape (nparticles, 3)

    d_xy, d_z (float): Grid size in xy and z directions.

    n_grid_xy (int): Number of grid points for x and y directions

    n_grid_z (int): Number of grid points for z direction.

    """

    dis_r = particle_coords / np.array([d_xy, d_xy, d_z])

    idx_r = dis_r.astype(np.int_)

    dis_r -= idx_r

    # Periodic boundary conditions.

    idx_r %= np.array([n_grid_xy, n_grid_xy, n_grid_z], dtype=np.int_)

    for (x, y, z), (dx, dy, dz) in zip(idx_r, dis_r):
        # Next grid points (e.g., x + 1). The following function applies

        # periodic boundary conditions.

        xp = clip_grid_idx(x, n_grid_xy)

        yp = clip_grid_idx(y, n_grid_xy)

        zp = clip_grid_idx(z, n_grid_z)

        density_grid[x, y, z] += (1 - dx) * (1 - dy) * (1 - dz)

        density_grid[x, y, zp] += (1 - dx) * (1 - dy) * dz

        density_grid[x, yp, z] += (1 - dx) * dy * (1 - dz)

        density_grid[xp, y, z] += dx * (1 - dy) * (1 - dz)

        density_grid[x, yp, zp] += (1 - dx) * dy * dz

        density_grid[xp, y, zp] += dx * (1 - dy) * dz

        density_grid[xp, yp, z] += dx * dy * (1 - dz)

        density_grid[xp, yp, zp] += dx * dy * dz
