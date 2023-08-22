from torch.autograd import Function
import numpy as np
from cholespy import CholeskySolverF, MatrixType
import torch

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

class CholeskySolver():
    """
    Cholesky solver.

    Precomputes the Cholesky decomposition of the system matrix and solves the
    system by back-substitution.
    """
    def __init__(self, M):
        self.solver = CholeskySolverF(M.shape[0], M.indices()[0], M.indices()[1], M.values(), MatrixType.COO)

    def solve(self, b, backward=False):
        x = torch.zeros_like(b)
        self.solver.solve(b.detach(), x)
        return x

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
            b_grad = ctx.solver.solve(grad_output.contiguous(), backward=True)
        return (solver_grad, b_grad)

# Alias for DifferentiableSolve function
solve = DifferentiableSolve.apply
