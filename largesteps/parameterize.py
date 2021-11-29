from largesteps.solvers import CholeskySolver, ConjugateGradientSolver, solve
from weakref import WeakKeyDictionary

# Cache for the system solvers
#_cache = WeakKeyDictionary()
_cache = {}

def to_differential(L, v):
    return L @ v

def from_differential(L, u, method='Cholesky'):
    """
    Convert differential coordinates back to Cartesian.

    If this is the first time we call this function on a given matrix L, the
    solver is cached. It will be destroyed once the matrix is garbage collected.

    Parameters
    ----------
    L : torch.spare.Tensor
        (I +l*L) matrix
    u : torch.Tensor
        Differential coordinates
    method : {'Cholesky', 'CG'}
        Solver to use.
    """
    if L not in _cache.keys():
        if method == 'Cholesky':
            solver = CholeskySolver(L)
        elif method == 'CG':
            solver = ConjugateGradientSolver(L)
        else:
            raise ValueError(f"Unknown solver type '{method}'.")

        _cache[L] = solver
    else:
        solver = _cache[L]
        # TODO: make sure that the method hasn't changed here

    return solve(solver, u)
