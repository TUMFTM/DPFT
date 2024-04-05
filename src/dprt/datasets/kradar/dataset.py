from __future__ import annotations  # noqa: F407

import os
import os.path as osp

from itertools import chain
from typing import Any, Dict, List, Tuple, Union

import torch
import numpy as np

from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms.functional import resize

from dprt.datasets.kradar.utils import radar_info


class KRadarDataset(Dataset):
    def __init__(self,
                 src: str,
                 version: str = '',
                 split: str = 'train',
                 camera: str = 'M',
                 camera_dropout: float = 0.0,
                 image_size: Union[int, Tuple[int, int]] = None,
                 radar: str = 'BF',
                 radar_dropout: float = 0.0,
                 lidar: int = 0,
                 label: str = 'detection',
                 num_classes: int = 1,
                 sequential: bool = False,
                 scale: bool = True,
                 fov: Dict[str, Tuple[float, float]] = None,
                 dtype: str = 'float32',
                 **kwargs):
        """Dataset class for the K-Radar dataset.

        Arguments:
            src: Source path to the pre-processed
                dataset folder.
            version: Dataset version. One of either
                mini or None (full dataset).
            split: Dataset split to load. One of
                either train or test.
            camera: Camera modalities to use. One of
                either 'M' (mono camera), 'S' (stereo camera)
                or None.
            camera_dropout: Camera modality dropout probability
                between 0 and 1.
            image_size: Image size to resize image data to.
                Either None (no resizing), int or tuple of two
                int specifying the height and width.
            radar: Radar modalities to use. Any combination
                of 'B' (BEV) and 'F' (Front) or None
            radar_dropout: Radar modality dropout probability
                between 0 and 1.
            lidar: Lidar modality to use. One of either
                0 (no lidar), 1 (OS1) or 2 (OS2).
            label: Type of label data to use. One of either
                'detection' (3D bounding boxes), 'occupancy'
                (3D occupancy grid) or None.
            num_classes: Number of object classes used for
                one hot encoding.
            sequential: Whether to consume sequneces of
                samples or single samples.
            scale: Whether to scale the radar data to
                a value range of [0, 255] or not.
            fov: Field of view to limit the lables to. Can
                contain values for x, y, z and azimuth.
        """
        # Initialize parent dataset class
        super().__init__()

        # Check attribute values
        assert camera_dropout + radar_dropout <= 1.0

        # Initialize instance attributes
        self.src = src
        self.version = version
        self.split = split
        self.camera = camera
        self.camera_dropout = camera_dropout
        self.image_size = image_size
        self.radar = radar
        self.radar_dropout = radar_dropout
        self.lidar = lidar
        self.label = label
        self.num_classes = num_classes
        self.sequential = sequential
        self.scale = scale
        self.fov = fov if fov is not None else {}
        self.dtype = dtype

        # Adjust split according to dataset version
        if self.version:
            self.split = f"{self.version}_{self.split}"

        # Initialize moality dropout attributes
        # Define the lottery pot to draw from (None, camera, radar)
        self.lottery = [
            {},
            {'camera_mono', 'camera_stereo'},
            {'radar_bev', 'radar_front'}
        ]

        # Define dropout probabilities (sum of probabilities must be <= 1)
        self.dropout = [
            1 - (self.camera_dropout + self.radar_dropout),
            self.camera_dropout,
            self.radar_dropout
        ]

        # Get dataset path
        self.dataset_paths = self.get_dataset_paths(self.src)

    def __len__(self):
        return len(self.dataset_paths)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        """Returns an item of the dataset given its index.

        Whether or not the retured Tensors include a time
        dimension depends on whether or not sequential is
        true or false.

        Arguments:
            index: Index of the dataset item to return.

        Returns:
            item: Dataset item as dictionary of tensors.
        """
        # Map index to sequence number for sequential data
        if self.sequential:
            index = list(sorted(self.dataset_paths.keys()))[index]

        # Get item from dataset
        item = self._to_list(self.dataset_paths[index])

        # Load data from file paths
        for sample in item:
            sample = self.load_sample_data(sample)

        # Scale radar data
        if self.scale:
            sample = self.scale_radar_data(sample)

        # Apply modality dropout
        sample = self.modality_dropout(sample)

        # Get task specific label
        if self.label == 'detection':
            label = self.get_detection_label(sample.pop('label'))

        # Add description to label
        label.update({'description': sample.pop('description')})

        # Set sensor data transformations (transformations in cartesian space)
        sample = self._add_transformations(sample)

        # Set sensor data projections (projections in sensor space)
        sample = self._add_projections(sample)

        # Set sensor data input shape
        sample = self._add_shape(sample)

        # Resize image (if required)
        if self.image_size is not None:
            sample = self.resize_image(sample)

        # Convert list of dicts to dict of stacked tensors
        if self.sequential:
            # Stack tensors along the time dimension (use padding for variable
            # size inputs, e.g. label)
            # item = {key: default_collate([d[key] for d in item]) for key in sample}
            raise NotImplementedError()
        else:
            # There is just a single sample for non sequential data
            item = sample

        return item, label

    @classmethod
    def from_config(cls, config: Dict, *args, **kwargs) -> KRadarDataset:  # noqa: F821
        return cls(*args, **dict(config['computing'] | config['data']), **kwargs)

    @staticmethod
    def _to_list(item: Any) -> List[Any]:
        if not isinstance(item, (list, tuple, set)):
            return [item]
        return item

    def _add_transformations(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Adds the transformation matirces to the sample.

        Arguments:
            sample: Dictionary mapping the sample
                items to thier data tensors.

        Returns:
            sample: Dictionary mapping the sample
                items to thier scaled data tensors.
        """
        if 'M' in self.camera:
            sample['label_to_camera_mono_t'] = torch.zeros_like(sample['label_to_camera_mono'])
        if 'S' in self.camera:
            sample['label_to_camera_stereo_t'] = torch.zeros_like(sample['label_to_camera_stereo'])
        if 'B' in self.radar:
            sample['label_to_radar_bev_t'] = sample.pop('label_to_radar_bev')
        if 'F' in self.radar:
            sample['label_to_radar_front_t'] = sample.pop('label_to_radar_front')

        return sample

    def _add_projections(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Adds the projection matrices to the sample.

        Arguments:
            sample: Dictionary mapping the sample
                items to thier data tensors.

        Returns:
            sample: Dictionary mapping the sample
                items to thier scaled data tensors.
        """
        if 'M' in self.camera:
            sample['label_to_camera_mono_p'] = sample.pop('label_to_camera_mono')
        if 'S' in self.camera:
            sample['label_to_camera_stereo_p'] = sample.pop('label_to_camera_stereo')
        if 'B' in self.radar:
            sample['label_to_radar_bev_p'] = self._get_radar_ra_projection()
        if 'F' in self.radar:
            sample['label_to_radar_front_p'] = self._get_radar_ea_projection()

        return sample

    def _add_shape(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Adds the original input data shape to the sample.

        Arguments:
            sample: Dictionary mapping the sample
                items to thier data tensors.

        Returns:
            sample: Dictionary mapping the sample
                items to thier scaled data tensors.
        """
        if 'M' in self.camera:
            sample['camera_mono_shape'] = torch.as_tensor(sample['camera_mono'].shape)
        if 'S' in self.camera:
            sample['camera_stereo_shape'] = torch.as_tensor(sample['camera_stereo'].shape)
        if 'B' in self.radar:
            sample['radar_bev_shape'] = torch.as_tensor(sample['radar_bev'].shape)
        if 'F' in self.radar:
            sample['radar_front_shape'] = torch.as_tensor(sample['radar_front'].shape)

        return sample

    def _get_radar_ea_projection(self) -> torch.Tensor:
        """Returns a projection matrix for the elevation-azimuth projection.

        The projection matrix P is given that
        [u]
        [v] = P [r, phi, roh, 1]
        [1]

        with range (r), azimuth (phi) and elevation (roh) in spherical
        coordinates. So that, u and v represent the indices of the radar
        grid (raster).
        """
        return torch.Tensor([
            [0, -1, 0, (len(radar_info.azimuth_raster) - 1) / 2],
            [0, 0, 1, (len(radar_info.elevation_raster) - 1) / 2],
            [0, 0, 0, 1]
        ]).type(getattr(torch, self.dtype))

    def _get_radar_ra_projection(self) -> torch.Tensor:
        """Returns a projection matrix for the range-azimuth projection.

        The projection matrix P is given that
        [u]
        [v] = P [r, phi, roh, 1]
        [1]

        with range (r), azimuth (phi) and elevation (roh) in spherical
        coordinates. So that, u and v represent the indices of the radar
        grid (raster).
        """
        return torch.Tensor([
            [0, -1, 0, (len(radar_info.azimuth_raster) - 1) / 2],
            [len(radar_info.range_raster) / max(radar_info.range_raster), 0, 0, 0],
            [0, 0, 0, 1]
        ]).type(getattr(torch, self.dtype))

    def scale_radar_data(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Scales the radar data to a range of 0 to 255

        Arguments:
            sample: Dictionary mapping the sample
                items to thier data tensors.

        Returns:
            sample: Dictionary mapping the sample
                items to thier scaled data tensors.
        """
        for k, v in sample.items():
            if k in {'radar_bev', 'radar_front'}:
                # Scale data to target value range
                sample[k] = \
                    (v - radar_info.min_power) \
                    / (radar_info.max_power - radar_info.min_power) \
                    * (255 - 0) + 0

                # Ensure target value range
                sample[k] = torch.clip(sample[k], 0, 255)

        return sample

    def resize_image(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Resizes the images in the sample.

        Resizes the images to a Tensor of shape (C, self.image_size[0], self.image_size[1])
        if image size if given as a tuple of interger values, otherwise it resizes it to a
        tensor with shape (C, self.image_size, self.image_size * width / height).

        Arguments:
            sample: Dictionary mapping the sample
                items to thier data tensors.

        Returns:
            sample: Dictionary mapping the sample
                items to thier data tensors.
        """
        if 'M' in self.camera:
            sample['camera_mono'] = \
                resize(sample['camera_mono'].movedim(-1, 0), self.image_size).movedim(0, -1)
        if 'S' in self.camera:
            sample['camera_stereo'] = \
                resize(sample['camera_stereo'].movedim(-1, 0), self.image_size).movedim(0, -1)

        return sample

    def get_detection_label(self, raw_label: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Get detection task label data.

        Splits the K-Radar dataset label, given as bounding box of
        [x, y, z, theta, l, w, h, category index, object id], into
        its individual components.

        Arguments:
            raw_label: Collection of sample data.

        Returns:
            label: Modified collecton of sample data.
        """
        # Initialize label
        label = {}

        # Split label data into its components
        label['gt_center'] = raw_label[:, (0, 1, 2)]
        label['gt_size'] = raw_label[:, (4, 5, 6)]

        # Encode angle by its sin and cos part
        label['gt_angle'] = torch.cat(
            (torch.sin(raw_label[:, (3, )]), torch.cos(raw_label[:, (3, )])),
            dim=-1
        )

        # One hot encode class labels (+1 for ignore class)
        label['gt_class'] = torch.nn.functional.one_hot(
            raw_label[:, 7].long() + 1,
            self.num_classes
        ).type(getattr(torch, self.dtype))

        # Get configured field of view
        x_min, x_max = self.fov.get('x', torch.tensor([-float('inf'), float('inf')]))
        y_min, y_max = self.fov.get('y', torch.tensor([-float('inf'), float('inf')]))
        z_min, z_max = self.fov.get('z', torch.tensor([-float('inf'), float('inf')]))
        a_min, a_max = self.fov.get('azimuth', torch.tensor([-float('inf'), float('inf')]))

        # Get azimuth angle of the center points
        azimuth = torch.rad2deg(torch.arctan2(label['gt_center'][:, 1], label['gt_center'][:, 0]))

        # Limit lables to configured field of view (FoV)
        x_mask = (x_min < label['gt_center'][:, 0]) & (label['gt_center'][:, 0] < x_max)
        y_mask = (y_min < label['gt_center'][:, 1]) & (label['gt_center'][:, 1] < y_max)
        z_mask = (z_min < label['gt_center'][:, 2]) & (label['gt_center'][:, 2] < z_max)
        a_mask = (a_min < azimuth) & (azimuth < a_max)

        fov_mask = x_mask & y_mask & z_mask & a_mask

        # Mask lables according to the field of view
        label = {k: v[fov_mask] for k, v in label.items()}

        return label

    def get_sample_path(self, src: str) -> Dict[str, str]:
        """Returns all data paths of a given dataset sample.

        Arguments:
            src: Sourcce path to the data files
                of a single dataset sample.

        Returns:
            sample_batch: Dictionary mapping the sample
                items to filenames.
        """
        # Initialize sample paths
        sample_path = {}

        # Get sensor data and calibration information
        if 'M' in self.camera:
            sample_path['camera_mono'] = osp.join(src, 'mono.jpg')
            sample_path['label_to_camera_mono'] = osp.join(src, 'mono_info.npy')

        if 'S' in self.camera:
            sample_path['camera_stereo'] = osp.join(src, 'stereo.jpg')
            sample_path['label_to_camera_stereo'] = osp.join(src, 'stereo_info.npy')

        if 'B' in self.radar:
            sample_path['radar_bev'] = osp.join(src, 'ra.npy')
            sample_path['label_to_radar_bev'] = osp.join(src, 'ra_info.npy')

        if 'F' in self.radar:
            sample_path['radar_front'] = osp.join(src, 'ea.npy')
            sample_path['label_to_radar_front'] = osp.join(src, 'ea_info.npy')

        if self.lidar == 1:
            sample_path['lidar_top'] = osp.join(src, 'os1.npy')

        if self.lidar == 2:
            sample_path['lidar_top'] = osp.join(src, 'os2.npy')

        # Get annotation data
        if self.label == 'detection':
            sample_path['label'] = osp.join(src, 'labels.npy')

        # Get description data
        sample_path['description'] = osp.join(src, 'description.npy')

        return sample_path

    def get_dataset_paths(
        self,
        src: str
    ) -> Union[Dict[str, List[Dict[str, str]]], List[Dict[str, str]]]:
        """Returns the paths of all dataset items.

        The return type is either a list of dictionaries (each representing
        a single sample) or a dictionary of lists (each representing a
        single sequence), where each list holds the dictionaries of the
        single samples.

            sequential: Dict[sequence number, List[sample dicts]]
            non sequential: List[sample dicts]

        Arguments:
            src: Source path to the pre-processed
                dataset folder.

        Returns:
            dataset_paths: File paths of all dataset
                items (either sequences or samples).
        """
        # Initialize dataset paths
        dataset_paths = {}

        # List all sequences in the dataset
        for sequence in os.listdir(osp.join(src, self.split)):
            # Set sequence path
            sequence_path = osp.join(src, self.split, sequence)

            # List all samples in the sequence
            samples = sorted(os.listdir(sequence_path))

            # Disolve all sample data paths
            dataset_paths[sequence] = [
                self.get_sample_path(osp.join(sequence_path, sample)) for sample in samples
            ]

        # Concatenate all sequences for non sequential data
        if not self.sequential:
            dataset_paths = list(chain.from_iterable(dataset_paths.values()))

        return dataset_paths

    def load_sample_data(self, sample_path: Dict[str, str]) -> Dict[str, torch.Tensor]:
        """Returns the actual sample data given their paths.

        Arguments:
            sample: Dictionary mapping the sample
                items to filenames.

        Returns:
            sample: Dictionary mapping the sample
                items to thier data tensors.
        """
        # Initialize sample
        sample = {}

        # Load sample datak
        for key, path in sample_path.items():
            if osp.splitext(path)[-1] in {'.png', '.jpg'}:
                # Load image
                img: torch.Tensor = read_image(path).type(getattr(torch, self.dtype))

                # Change to channel last format (C, H, W) -> (H, W, C)
                sample[key] = img.movedim(0, -1)

            if osp.splitext(path)[-1] in {'.npy'}:
                sample[key] = torch.from_numpy(np.load(path)).type(getattr(torch, self.dtype))

        return sample

    def modality_dropout(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Applies modality dropout to the sample data.

        Randomly drops one input modality by setting all input
        values to zero. The drop ratio of each modality is given
        by their individual dropout propabilities.

        Note: It is ensured that not all modalities are dropped
        at the same time but at least one modality remains.

        Arguments:
            sample: Dictionary mapping the sample
                items to thier data tensors.

        Returns:
            sample: Dictionary mapping the sample items to
                thier data tensors with applied dropout.
        """
        # Draw of lots (select a modality based on their probabilities)
        drawing = self.lottery[np.random.choice(3, replace=True, p=self.dropout)]

        # Apply dropout (replace selected input modality with zeros)
        for modality in drawing:
            if modality in sample:
                sample[modality] = torch.zeros_like(sample[modality])

        return sample


def initialize_kradar(*args, **kwargs):
    return KRadarDataset.from_config(*args, **kwargs)
