from benchopt import BaseSolver, safe_import_context
from pathlib import Path

with safe_import_context() as import_ctx:
    import os
    from fastmri_recon.models.subclassed_models.ncpdnet import NCPDNet
    import tensorflow as tf
    import numpy as np

    # don't use the full GPU memory
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(device=gpu, enable=True)


proj_dir = Path(__file__).parent.parent


class Solver(BaseSolver):

    name = "ncpdnet"
    install_cmd = "pip"
    sampling_strategy = "run_once"
    requirements = [
        "git+ssh://git@github.com/zaccharieramzi/fastmri-reproducible-benchmark"
    ]

    def set_objective(self, kspace_data, physics, trajectory_name):
        # Convert the kspace data from torch to tf
        self.kspace_data = tf.convert_to_tensor(kspace_data.cpu().numpy()) * 1e6
        self.traj = tf.convert_to_tensor(physics.nufft.samples) * 2 * np.pi
        self.smaps = tf.convert_to_tensor(physics.nufft.smaps)
        self.shape = physics.nufft.shape
        self.dcomp = tf.convert_to_tensor(physics.nufft.density)
        # Create the NCPDNet model
        # The model is created with the same parameters as in the original

        model = NCPDNet(
            multicoil=True,
            im_size=self.shape,
            dcomp=True,
            refine_smaps=True,
            output_shape_spec=True,
        )
        kspace_shape = self.kspace_data.shape
        inputs = [
            tf.zeros([*kspace_shape, 1], dtype=tf.complex64),
            tf.zeros([1, 2, kspace_shape[-1]], dtype=tf.float32),
            tf.zeros([1, *self.smaps.shape], dtype=tf.complex64),
            (self.shape[::-1],),
            (
                tf.constant([*self.shape]),
                tf.ones([1, kspace_shape[-1]], dtype=tf.float32),
            ),
        ]
        model(inputs)

        NCPDNET_PATH = os.environ.get(
            "NCPDNET_PATH", proj_dir / f"ncpdnet_{trajectory_name}_2d.h5"
        )
        model.load_weights(NCPDNET_PATH)

        self.model = model

    def run(self, stopval=None):
        # Run the NCPDNet model on the kspace data
        x = self.model(
            [
                self.kspace_data[..., None],
                tf.transpose(self.traj)[None],
                self.smaps[None],
                (self.shape[::-1],),
                (
                    self.shape,
                    self.dcomp[None],
                ),
            ]
        )
        self.x_estimate = x

    def get_result(self):
        return {
            "x_estimate": self.x_estimate.numpy().squeeze(),
            "cost": 0,
        }
