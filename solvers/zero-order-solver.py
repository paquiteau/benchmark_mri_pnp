from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import pysap


class Solver(BaseSolver):
    """Zero order solution"""
    name = 'adjoint'

    install_cmd = 'conda'
    requirements = ['cmake', 'pip:python-pysap']

    # any parameter defined here is accessible as a class attribute
    parameters = {}

    def set_objective(self, kspace_data, fourier_op, image):
        # The arguments of this function are the results of the
        # `to_dict` method of the objective.
        # They are customizable.
        self.kspace_data = kspace_data
        self.fourier_op = fourier_op
        self.image = image

    def run(self, n_iter):
        self.w = pysap.Image(
            data=self.fourier_op.adj_op(self.kspace_data),
            metadata=self.image.metadata
        )

    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objective.
        # They are customizable.
        return {"beta": self.w}
