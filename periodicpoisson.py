import dolfinx
from petsc4py.PETSc import ScalarType, NullSpace
from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import fem, io, mesh, plot, default_scalar_type
from ufl import grad, inner, dot

from dolfinx_mpc import LinearProblem, MultiPointConstraint
import pyvista


msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((-0.5, -0.5), (0.5, 0.5)),
    n=(100, 100),
    cell_type=mesh.CellType.triangle,
)

V = fem.functionspace(msh, ("Lagrange", 2))

tol = 250 * np.finfo(default_scalar_type).resolution

bcs = None # No Dirichlet boundary conditions

# Identifying the boundaries
def periodic_boundary_top(x):
    return np.isclose(x[1], -0.5, atol = tol)

def periodic_relation_top(x):
    out_x = np.zeros_like(x)
    out_x[0] = x[0]
    out_x[1] = -x[1]
    out_x[2] = x[2]
    return out_x

def periodic_boundary_right(x):
    return np.isclose(x[0], -0.5, atol = tol)

def periodic_relation_right(x):
    out_x = np.zeros_like(x)
    out_x[0] = -x[0]
    out_x[1] = x[1]
    out_x[2] = x[2]
    return out_x


# Periodic boundary conditions via mpc
mpc = MultiPointConstraint(V)
mpc.create_periodic_constraint_geometrical(V, periodic_boundary_right,
        periodic_relation_right, bcs)
mpc.create_periodic_constraint_geometrical(V, periodic_boundary_top,
        periodic_relation_top, bcs)
mpc.finalize()

# Formulation of the PDE and solving
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(msh)
f = 2 * ((2 * np.pi)**2) * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
quadrature_degree = 5
dx = ufl.dx(metadata={"quadrature_degree":quadrature_degree}, domain=msh)

a = dot(grad(u), grad(v)) * dx
L = f * v * dx

problem = LinearProblem(a, L, mpc, bcs = [], petsc_options={"ksp_type":
    "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})
solver = problem.solver
nullspace = NullSpace().create(constant=True, comm=MPI.COMM_WORLD)
problem.A.setNullSpace(nullspace)
uh = problem.solve()
assert(solver.getConvergedReason() > 0)

uh_avg = msh.comm.allreduce(fem.assemble_scalar(fem.form(uh*dx)), op = MPI.SUM)
vol = msh.comm.allreduce(fem.assemble_scalar(fem.form(1*dx)), op = MPI.SUM)
max_val = np.max(np.abs(uh.x.array))

uh.x.array[:] -= uh_avg/vol
    

# Errors

# Exact solution, note that we have to shift x[0] and x[1], because our domain
# is not [0,1] \times [0,1], but [-0.5, 0.5} \times [-0.5, 0.5]
exact_sol = lambda x : np.sin(np.pi * 2* (x[0]+0.5)) * np.sin(np.pi * 2* (x[1]+0.5))
u_exact = fem.Function(mpc.function_space, name="u_exact")
u_exact.interpolate( exact_sol )

# L2 error
L2_error = fem.form(ufl.dot(uh - u_exact, uh - u_exact) * dx)
error_local = fem.assemble_scalar(L2_error) # local to rank
error_L2 = np.sqrt(msh.comm.allreduce(error_local, op=MPI.SUM))

# Max error
error_max = np.max(np.abs(u_exact.x.array-uh.x.array))

# Only print the error on one process
if msh.comm.rank == 0:
    print(f"L2 error : {error_L2:.2e}")
    print(f"Max nodal error : {error_max:.2e}")


# Plotting

cells, types, x = plot.vtk_mesh(V)
grid = pyvista.UnstructuredGrid(cells, types, x)
grid.point_data["u"] = uh.x.array.real
grid.set_active_scalars("u")
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)
warped = grid.warp_by_scalar()
plotter.add_mesh(warped)
plotter.show()

