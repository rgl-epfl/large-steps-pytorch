from torch.autograd import Function
import numpy as np
import scipy.sparse as sp
import sksparse.cholmod as cholmod
import cupy as cp
import cupyx.scipy.sparse as cps
import torch

from cupy._core.dlpack import toDlpack
from cupy._core.dlpack import fromDlpack
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack

import cupy as _cupy
from cupy.cuda import device as _device
from cupy_backends.cuda.libs import cusparse as _cusparse
import cupyx.scipy.sparse

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

class Solver:
    """
    Sparse linear system solver base class.
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

    def __del__(self):
        if hasattr(self, 'infos'):
            for info in self.infos:
                _cusparse.destroyCsrsm2Info(info)
        if hasattr(self, 'descs'):
            for desc in self.descs:
                _cusparse.destroyMatDescr(desc)


    def prepare(self, A, transpose, blocking=True, level_info=True):
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
        self.infos.append(info) # will cause memory leak if not freed manully
        m = A.shape[0]
        alpha = np.array(1, dtype=np.float32)
        desc = _cusparse.createMatDescr()
        self.descs.append(desc) # will cause memory leak if not freed manully
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
        self.infos = [] # cusparse allocation records (needs manual freeing in __del__)
        self.descs = [] # cusparse allocation records (needs manual freeing in __del__)
        self.solver_1 = self.prepare(self.L, False, False, True)
        self.solver_2 = self.prepare(self.L, True, False, True)

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
    def __init__(self, M):
        """
        Initialize the solver.

        Parameters
        ----------
        M : torch.sparse_coo_tensor
            Linear system matrix.
        """
        self.guess_fwd = None
        self.guess_bwd = None
        self.M = M

    def solve_axis(self, b, x0):
        """
        Solve a single linear system with Conjugate Gradients.

        Parameters:
        -----------
        b : torch.Tensor
            The right hand side of the system Ax=b.
        x0 : torch.Tensor
            Initial guess for the solution.
        """
        x = x0
        r = self.M @ x - b
        p = -r
        r_norm = r.norm()
        while r_norm > 1e-5:
            Ap = self.M @ p
            r2 = r_norm.square()
            alpha = r2 / (p * Ap).sum(dim=0)
            x = x + alpha*p
            r_old = r
            r_old_norm = r_norm
            r = r + alpha*Ap
            r_norm = r.norm()
            beta = r_norm.square() / r2
            p = -r + beta*p
        return x

    def solve(self, b, backward=False):
        """
        Solve the sparse linear system.

        There is actually one linear system to solve for each axis in b
        (typically x, y and z), and we have to solve each separately with CG.
        Therefore this method calls self.solve_axis for each individual system
        to form the solution.

        Parameters
        ----------
        b : torch.Tensor
            The right hand side of the system Ax=b.
        backward : bool
            Whether we are in the backward or the forward pass.
        """
        if self.guess_fwd is None:
            # Initialize starting guesses in the first run
            self.guess_bwd = torch.zeros_like(b)
            self.guess_fwd = torch.zeros_like(b)

        if backward:
            x0 = self.guess_bwd
        else:
            x0 = self.guess_fwd

        if len(b.shape) != 2:
            raise ValueError(f"Invalid array shape {b.shape} for ConjugateGradientSolver.solve: expected shape (a, b)")

        x = torch.zeros_like(b)
        for axis in range(b.shape[1]):
            # We have to solve for each axis separately for CG to converge
            x[:, axis] = self.solve_axis(b[:, axis], x0[:, axis])

        if backward:
            # Update initial guess for next iteration
            self.guess_bwd = x
        else:
            self.guess_fwd = x

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
