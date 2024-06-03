from benchopt import BaseDataset, safe_import_context
from pathlib import Path

with safe_import_context() as import_ctx:
    from mrinufft import get_operator
    from mrinufft.io import read_trajectory
    from benchmark_utils.smaps import get_smaps

    from benchmark_utils.spiral_factory import stack_spiral_factory
    from brainweb_dl import get_mri


class Dataset(BaseDataset):
    name = "brainweb-spiral"

    install_cmd = "conda"
    requirements = ["pip:brainweb-dl", "pip:mri-nufft", "pip:finufft"]

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # parameters = {
    #     'n_samples, n_features': [
    #         (1000, 500),
    #         (5000, 200)]
    # }
    parameters = {
    #    "sub_id": [4, 5, 6, 18, 20, 38] + list(range(41, 55)),
        "subid": [4],
        "contrast": ["T1"],
        "n_coils": [8],
        "accelz": [4],
        "nb_revolutions": [20],
        "n_samples": [25000],
        "shape":[(192,192,128)]
    }



    def get_data(self):

        shape = self.shape

        kspace_mask = stack_spiral_factory(shape, accelz=self.accelz, acsz=0.1, n_samples=self.n_samples, nb_revolutions=self.nb_revolutions)

        smaps =  get_smaps(shape, self.n_coils)
        image = get_mri(
            sub_id=self.subid,
            contrast=self.contrast,
            shape=shape,
        ).astype("complex64")

        nufft = get_operator("finufft")(
            samples = kspace_mask,
            shape=shape,
            smaps=smaps,
            n_coils=smaps.shape[0],
        )

        kspace_data = nufft.op(image)

        # `data` holds the keyword arguments for the `set_data` method of the
        # objective.
        # They are customizable.
        return dict(kspace_data=kspace_data, kspace_mask=kspace_mask, image=image, smaps=smaps)
