from largesteps.solvers import CholeskySolver, ConjugateGradientSolver, solve
import weakref

# Cache for the system solvers
_cache = {}

def cache_put(key, value, A):
    # Called when 'A' is garbage collected
    def cleanup_callback(wr):
        del _cache[key]

    wr = weakref.ref(
        A,
        cleanup_callback
    )

    _cache[key] = (value, wr)

def to_differential(L, v):
    """
    Convert vertex coordinates to the differential parameterization.

    Parameters
    ----------
    L : torch.sparse.Tensor
        (I + l*L) matrix
    v : torch.Tensor
        Vertex coordinates
    """
    return L @ v

def from_differential(L, u, method='Cholesky'):
    """
    Convert differential coordinates back to Cartesian.

    If this is the first time we call this function on a given matrix L, the
    solver is cached. It will be destroyed once the matrix is garbage collected.

    Parameters
    ----------
    L : torch.sparse.Tensor
        (I + l*L) matrix
    u : torch.Tensor
        Differential coordinates
    method : {'Cholesky', 'CG'}
        Solver to use.
    """
    key = (id(L), method)
    if key not in _cache.keys():
        if method == 'Cholesky':
            solver = CholeskySolver(L)
        elif method == 'CG':
            solver = ConjugateGradientSolver(L)
        else:
            raise ValueError(f"Unknown solver type '{method}'.")

        cache_put(key, solver, L)
    else:
        solver = _cache[key][0]

    return solve(solver, u)
