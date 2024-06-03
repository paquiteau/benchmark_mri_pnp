from benchopt import BaseDataset, safe_import_context
from pathlib import Path

with safe_import_context() as import_ctx:
    from mrinufft import get_operator
    from mrinufft.io import read_trajectory
    from benchmark_utils.smaps import get_smaps

    from brainweb_dl import get_mri


class Dataset(BaseDataset):
    name = "brainweb-sparkling"

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
        "trajectory":["sparkling3d-48-2688x5.bin"],
    }



    def get_data(self):
        kspace_mask, params = read_trajectory(Path(__file__).parent / self.trajectory)
        kspace_mask = kspace_mask.astype("float32")
        shape = params["img_size"]
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
