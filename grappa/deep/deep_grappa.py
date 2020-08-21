from pathlib import Path

import numpy as np
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.callbacks import TQDMProgressBar

from grappa.config import LOG_DIR
from grappa.kernel_application import padding_for_kspace, list_targets_sources_for_application, inference_on_target_positions
from grappa.kernel_estimation import autocalibration_signal, list_target_source_values_for_estimation
from grappa.reconstruction import crop, rss
from grappa.utils import number_geometries, eval_at_positions, cartesian_product


class DeepGRAPPA:
    def __init__(
            self,
            model_init_function,
            ny=3,
            n_epochs=10,
            lr=1e-3,
            distance_from_center_feat=False,
            verbose=0,
            logging_history=False,
            **model_kwargs,
        ):
        self.model_init_function = model_init_function
        self.model_kwargs = model_kwargs
        self.ny = ny
        self.n_epochs = n_epochs
        self.lr = lr
        self.distance_from_center_feat = distance_from_center_feat
        self.verbose = verbose
        self.logging_history = logging_history
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
            if self.distance_from_center_feat:
                distance = self._list_targets_distances_from_center(
                    ac,
                    n_geometries,
                    i_geom,
                    mode='calib',
                )
                X = np.concatenate([source_values, distance])
            else:
                X = source_values
            model.compile(loss='mse', optimizer=Adam(lr=self.lr))
            callbacks = []
            if self.verbose:
                tqdm_cback = TQDMProgressBar()
                callbacks.append(tqdm_cback)
            if self.logging_history:
                additional_info = ''
                if self.ny != 3:
                    additional_info += f'ny{self.ny}_'
                if self.distance_from_center_feat:
                    additional_info += 'distance_'
                run_id = f'deep_grappa_{model.name}_i{i_geom}_{additional_info}{int(time.time())}'
                log_dir = Path(LOG_DIR) / 'logs' / run_id
                tboard_cback = TensorBoard(
                    profile_batch=0,
                    log_dir=log_dir,
                    histogram_freq=0,
                    write_graph=False,
                    write_images=False,
                )
            model.fit(
                x=X.T,
                y=target_values.T,
                epochs=self.n_epochs,
                verbose=0,
            )

    def _list_targets_distances_from_center(self, kspace, n_geometries, i_geom, mode='calib'):
        if mode == 'calib':
            delta_phase = 1
        else:
            delta_phase = n_geometries + 1
        targets = cartesian_product(
            # readout dimension
            np.arange(self.ny // 2, kspace.shape[1] - self.ny + (self.ny // 2) + 1),
            # phase dimension
            np.arange(i_geom + 1, kspace.shape[2] - n_geometries + i_geom, delta_phase),
        )
        target_patch_shape = np.array(kspace.shape[1:3]) - np.array(self.ny, n_geometries)
        targets_offset = np.linalg.norm(targets - target_patch_shape / 2, axis=1)
        return targets_offset

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
            if self.distance_from_center_feat:
                distance = self._list_targets_distances_from_center(
                    kspace,
                    spacing,
                    i_geom,
                    mode='inference',
                )
                X = np.concatenate([source_values, distance])
            else:
                X = source_values
            target_values = model.predict(X.T).T
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
