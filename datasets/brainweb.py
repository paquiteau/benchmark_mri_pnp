from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    from mri.operators import FFT
    from benchmark_utils.cartesian_sampling import get_cartesian_mask
    from brainweb_dl import get_mri


class Dataset(BaseDataset):
    name = "brainweb"

    install_cmd = "conda"
    requirements = ["pip:brainweb-dl"]

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
        "shape": [(-1,-1, 128)],
        "AF": [2, 4, 8],
        "seed": [42],
        "pdf": ["gaussian"],
    }

    #parameter_template = "{AF}-{pdf}"

   # def  __init__(self, **kwargs):
        # Store the parameters of the dataset


    def get_data(self):
        image = get_mri(
            sub_id=self.subid,
            contrast=self.contrast,
            shape=self.shape,
        )
        print(image.shape, image.dtype)
        mask = get_cartesian_mask(
            image.shape,
            rng=self.seed,
            center_prop=0.3,
            accel=self.AF,
            accel_axis=0,
            pdf=self.pdf,
        )

        # Loading input data
        # Generate the subsampled kspace
        fourier_op = FFT(mask=mask, shape=image.shape)
        kspace_data = fourier_op.op(image)

        # `data` holds the keyword arguments for the `set_data` method of the
        # objective.
        # They are customizable.
        return dict(kspace_data=kspace_data, fourier_op=fourier_op, image=image)
