from torch.autograd import Function
import numpy as np
import scipy.sparse as sp
import sksparse.cholmod as cholmod
import cupy as cp
import cupyx.scipy.sparse as cps

from cupy._core.dlpack import toDlpack
from cupy._core.dlpack import fromDlpack
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack

def torch_to_cupy(x):
    """
    Convert a PyTorch tensor to a CuPy array.
    """
    return fromDlpack(to_dlpack(x))

def cupy_to_torch(x):
    """
    Convert a CuPy array to a PyTorch tensor.
    """
    return from_dlpack(toDlpack(x))

def prepare(A, transpose, blocking=True, level_info=True):
    import cupy as _cupy
    from cupy.cuda import device as _device
    from cupy_backends.cuda.libs import cusparse as _cusparse
    import cupyx.scipy.sparse

    policy = _cusparse.CUSPARSE_SOLVE_POLICY_USE_LEVEL if level_info \
        else _cusparse.CUSPARSE_SOLVE_POLICY_NO_LEVEL
    algo = 1 if blocking else 0

    transa = _cusparse.CUSPARSE_OPERATION_TRANSPOSE if transpose \
        else _cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE
    transb = _cusparse.CUSPARSE_OPERATION_TRANSPOSE
    fill_mode = _cusparse.CUSPARSE_FILL_MODE_LOWER

    if cupyx.scipy.sparse.isspmatrix_csc(A):
        A = A.T
        transa = 1 - transa
        fill_mode = 1 - fill_mode

    assert cupyx.scipy.sparse.isspmatrix_csr(A)

    handle = _device.get_cusparse_handle()
    info = _cusparse.createCsrsm2Info()
    m = A.shape[0]
    alpha = np.array(1, dtype=np.float32)
    desc = _cusparse.createMatDescr()
    _cusparse.setMatType(desc, _cusparse.CUSPARSE_MATRIX_TYPE_GENERAL)
    _cusparse.setMatIndexBase(desc, _cusparse.CUSPARSE_INDEX_BASE_ZERO)
    _cusparse.setMatFillMode(desc, fill_mode)
    _cusparse.setMatDiagType(desc, _cusparse.CUSPARSE_DIAG_TYPE_NON_UNIT)

    nrhs = 3
    ldb = nrhs

    ws_size = _cusparse.scsrsm2_bufferSizeExt(
        handle, algo, transa, transb, m, nrhs, A.nnz, alpha.ctypes.data,
        desc, A.data.data.ptr, A.indptr.data.ptr, A.indices.data.ptr,
        0, ldb, info, policy)

    ws = _cupy.empty((ws_size,), dtype=np.int8)

    _cusparse.scsrsm2_analysis(
         handle, algo, transa, transb, m, nrhs, A.nnz, alpha.ctypes.data,
         desc, A.data.data.ptr, A.indptr.data.ptr,
         A.indices.data.ptr, 0, ldb, info, policy, ws.data.ptr)

    def solver(b):
        _cusparse.scsrsm2_solve(
            handle, algo, transa, transb, m, nrhs, A.nnz, alpha.ctypes.data,
            desc, A.data.data.ptr, A.indptr.data.ptr, A.indices.data.ptr,
            b.data.ptr, ldb, info, policy, ws.data.ptr)

    return solver

class Solver:
    """
    Sparse linear system solver base class.

    Methods
    -------
    solve
        Method called to solve the linear system in the forward  and backward pass
    """
    def __init__(self, M):
        pass

    def solve(self, b, backward=False):
        """
        Solve the linear system.

        Parameters
        ----------
        b : torch.Tensor
            The right hand side of the system Lx=b
        backward : bool (optional)
            Whether this is the backward or forward solve
        """
        raise NotImplementedError()

class CholeskySolver(Solver):
    """
    Cholesky solver.

    Precomputes the Cholesky decomposition of the system matrix and solves the
    system by back-substitution.
    """
    def __init__(self, M):
        """
        Initialize the solver

        Parameters
        ----------
        M : torch.tensor
            The matrix to decompose. It is assumed to be symmetric positive definite.
        """
        # Convert L to a scipy sparse matrix for factorization
        values = M.values().cpu().numpy()
        rows,cols = M.indices().cpu().numpy()
        M_cpu = sp.csc_matrix((values, (rows, cols)))
        factor = cholmod.cholesky(M_cpu, ordering_method='nesdis', mode='simplicial')
        L, P = factor.L(), factor.P()
        # Invert the permutation
        Pi = np.argsort(P).astype(np.int32)
        # Transfer to GPU as cupy arrays
        self.L = cps.csc_matrix(L.astype(np.float32))
        self.U = self.L.T
        self.P = cp.array(P)
        self.Pi = cp.array(Pi)
        self.solver_1 = prepare(self.L, False, False, True)
        self.solver_2 = prepare(self.L, True, False, True)

    def solve(self, b, backward=False):
        """
        Solve the sparse linear system.
        """
        tmp = torch_to_cupy(b)[self.P]
        self.solver_1(tmp)
        self.solver_2(tmp)
        return cupy_to_torch(tmp[self.Pi])

class ConjugateGradientSolver(Solver):
    """
    Conjugate gradients solver.
    """
    def __init__(self, M, use_guess=False):
        """
        Initialize the solver.

        Parameters
        ----------
        M : torch.sparse_coo_tensor
            Linear system matrix.
        """
        self.guess = None
        self.M = M

    def solve(self, b, backward=False):
        """
        Solve the sparse linear system.

        Parameters
        ----------
        b : torch.Tensor
            The right hand side of the system Ax=b.
        backward : bool
            Whether we are in the backward or the forward pass.
        """
        if not backward and self.guess is not None:
            x = self.guess # Value at previous iteration
        else:
            x = torch.zeros_like(b)

        r = b - self.M @ x
        p = r
        n = 0
        r_norm = r.norm().square()
        while r_norm > 1e-7:
            Ap = self.M @ p
            alpha = r_norm / (p * Ap).sum(dim=0)
            x = x + alpha*p
            r_old = r
            r_old_norm = r_norm
            r = r - alpha*Ap
            r_norm = r.norm().square()
            beta = r_norm / r_old_norm
            p = r + beta*p
        if not backward:
            # Update initial guess for next iteration
            self.guess = x
        #TODO: try guess on backward as well
        return x

class DifferentiableSolve(Function):
    """
    Differentiable function to solve the linear system.

    This simply calls the solve methods implemented by the Solver classes.
    """
    @staticmethod
    def forward(ctx, solver, b):
        ctx.solver = solver
        return solver.solve(b, backward=False)

    @staticmethod
    def backward(ctx, grad_output):
        solver_grad = None # We have to return a gradient per input argument in forward
        b_grad = None
        if ctx.needs_input_grad[1]:
            b_grad = ctx.solver.solve(grad_output, backward=True)
        return (solver_grad, b_grad)

# Alias for DifferentiableSolve function
solve = DifferentiableSolve.apply
