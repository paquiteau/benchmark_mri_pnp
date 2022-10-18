from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    from pysap.data import get_sample_data
    from mri.operators.utils import convert_mask_to_locations
    from mri.operators import FFT


class Dataset(BaseDataset):
    name = "pysap-sample"

    install_cmd = 'conda'
    requirements = ['pip:python-pysap']

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # parameters = {
    #     'n_samples, n_features': [
    #         (1000, 500),
    #         (5000, 200)]
    # }
    parameters = {
        "datadir": ["/home/benoit/Desktop/retreat_2022/pysap_data"]
    }

    def __init__(self, datadir):
        # Store the parameters of the dataset
        self.datadir = datadir

    def get_data(self):

        # Loading input data
        image = get_sample_data('2d-mri', datadir=self.datadir)

        # Obtain K-Space Cartesian Mask
        mask = get_sample_data("cartesian-mri-mask", datadir=self.datadir)

        # Get the locations of the kspace samples
        kspace_loc = convert_mask_to_locations(mask.data)
        # Generate the subsampled kspace
        fourier_op = FFT(samples=kspace_loc, shape=image.shape)
        kspace_data = fourier_op.op(image)

        # `data` holds the keyword arguments for the `set_data` method of the
        # objective.
        # They are customizable.
        return dict(
            kspace_data=kspace_data,
            fourier_op=fourier_op,
            image=image
        )
