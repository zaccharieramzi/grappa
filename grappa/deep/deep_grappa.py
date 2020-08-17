import numpy as np
from tensorflow.keras.optimizers import Adam

from grappa.kernel_application import padding_for_kspace, list_targets_sources_for_application, inference_on_target_positions
from grappa.kernel_estimation import autocalibration_signal, list_target_source_values_for_estimation
from grappa.reconstruction import ifft2, crop, rss
from grappa.utils import number_geometries, eval_at_positions


class DeepGRAPPA:
    def __init__(self, model_init_function, ny=3, n_epochs=10, **model_kwargs):
        self.model_init_function = model_init_function
        self.model_kwargs = model_kwargs
        self.ny = ny
        self.n_epochs = n_epochs
        self.models = None

    def calibrate_models(self, kspace, mask=None, af=4):
        ac = autocalibration_signal(kspace, mask, af)
        n_geometries = number_geometries(mask)
        ncoils = kspace.shape[0]
        self.models = [
            self.model_init_function(ncoils=ncoils, **self.model_kwargs)
            for _ in range(n_geometries)
        ]
        for i_geom, model in enumerate(self.models):
            target_values, source_values = list_target_source_values_for_estimation(
                ac=ac,
                i_geom=i_geom,
                ny=self.ny,
                n_geometries=n_geometries,
                ncoils=ncoils,
            )
            model.compile(loss='mse', optimizer=Adam(lr=1e-3))
            model.fit(x=source_values.T, y=target_values.T, epochs=self.n_epochs)

    def apply_models(self, kspace, mask):
        ncoils = kspace.shape[0]
        spacing = number_geometries(mask)
        pad_right, pad_left = padding_for_kspace(mask, spacing)
        kspace_padded = np.pad(
            kspace,
            [
                (0, 0),
                (self.ny//2, self.ny//2),
                (pad_left, pad_right),
            ]
        )
        for i_geom, model in enumerate(self.models):
            targets, sources = list_targets_sources_for_application(
                kspace=kspace_padded,
                ny=self.ny,
                i_geom=i_geom,
                spacing=spacing,
                ncoils=ncoils,
            )
            source_values = eval_at_positions(kspace_padded, sources)
            target_values = model.predict(source_values.T).T
            inference_on_target_positions(kspace_padded, targets, target_values, ncoils=ncoils)
        crop_right_readout = kspace_padded.shape[-1] - pad_right
        crop_right_phase = kspace_padded.shape[-2] - (self.ny//2)
        kspace_cropped = kspace_padded[:, self.ny//2:crop_right_phase, pad_left:crop_right_readout]
        kspace_consistent = mask * kspace + (1-mask) * kspace_cropped
        return kspace_consistent

    def reconstruct(self, kspace, mask, af=4, output_shape=None):
        reco_slices = list()
        for kspace_slice, mask_slice in zip(kspace, mask):
            self.calibrate_models(kspace_slice, mask_slice, af=af)
            filled_kspace = self.apply_models(kspace_slice, mask_slice)
            reco_grappa = rss(filled_kspace)
            reco_slice = crop(reco_grappa, output_shape=output_shape)
            reco_slices.append(reco_slice)
        reco_slices = np.array(reco_slices)
        return reco_slices
