from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from mri.operators import WaveletN

    from modopt.opt.proximity import SparseThreshold
    from modopt.opt.linear import Identity

    from benchmark_utils.init_solver import initialize_opt, get_grad_op, OPTIMIZERS


class Solver(BaseSolver):
    """Zero order solution"""
    name = 'CS'

    install_cmd = 'conda'
    sampling_strategy = 'callback'
    requirements = ['pip:python-pysap --install-option="--nosparse2D --only=pysap-mri"', 'pip:modopt']

    # any parameter defined here is accessible as a class attribute
    parameters = {
        "optimizer": ["pogm", "fista", "condat-vu"],
        "wavelet_name": ["haar", "sym8"],
        "nb_scales": [4],
        "lambd": [1e-7, 1e-8, 1e-4],
    }

    def set_objective(
        self,
        kspace_data,
        fourier_op,
        image,
    ):
        # The arguments of this function are the results of the
        # `to_dict` method of the objective.
        # They are customizable.

        print("Setting objective", self.optimizer, self.lambd, self.wavelet_name, self.nb_scales)
        self.kspace_data = kspace_data

        self.linear_op = WaveletN(
            wavelet_name=self.wavelet_name,
            nb_scales=self.nb_scales,
            dim=len(image.shape),
        )

        self.prox_op = SparseThreshold(
            Identity(),
            self.lambd,
            thresh_type="soft"
        )
        self.grad_op = get_grad_op(fourier_op, OPTIMIZERS[self.optimizer], linear_op=self.linear_op)
        # load the kspace data
        self.grad_op._obs_data = kspace_data
        self.grad_formulation = OPTIMIZERS[self.optimizer]
        self.solver = initialize_opt(
            self.optimizer,
            self.grad_op,
            self.linear_op,
            self.prox_op,
            opt_kwargs={"cost":None, "progress":False},
        )

    def run(self, callback):

        print("Running", self.solver.__class__.__name__, self.grad_formulation)

        if self.grad_formulation == "synthesis":
            w = self.linear_op.adj_op(self.solver._x_old)
        else:
            w = self.solver._x_old.copy()

        self.w = np.random.rand(*w.shape).astype(w.dtype)
        self.x = self.solver._x_old.copy()
        while callback():
            self.solver._update()
            self.solver.idx += 1
            if self.grad_formulation == "synthesis":
                w = self.linear_op.adj_op(self.solver._x_new)
            else:
                w = self.solver._x_new
            self.w = w.copy()
            self.x = self.solver._x_new.copy()

    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objective.
        # They are customizable.
        cost_grad = self.grad_op.cost(self.x)
        cost_prox = self.prox_op.cost(self.x)

        return {"beta": self.w,
                "cost_grad": cost_grad,
                "cost_prox": cost_prox,
                }
