from __future__ import annotations  # noqa: F407

import itertools
import os
import os.path as osp

from typing import Any, Dict, List

import torch

from dprt.utils.data import decollate_batch


class KRadarExporter():
    def __init__(self,
                 conf_thrs: List[float] = None,
                 categories: Dict[str, int] = None,
                 road_structures: Dict[str, int] = None,
                 weather_conditions: Dict[str, int] = None,
                 time_zone: Dict[str, int] = None,
                 **kwargs):
        """K-Radar exporter

        Arguments:
            conf_thrs: List of confidence thresholds to limit the
                export to.
            categories: Category mapping of the dataset classes.
                Maps a dataset class to a numerical category.
            road_structures: Road structures mapping of the dataset scene tags.
                Maps a dataset road structure tag to a numerical value.
            weather_conditions: Weather conditions mapping of the dataset scene tags.
                Maps a dataset weather conditions tag to a numerical value.
            time_zone: Time zone mapping of the dataset scene tags.
                Maps a dataset time zone tag to a numerical value.
        """
        # Initialize instance arguments
        self.conf_thrs = conf_thrs if conf_thrs is not None else [0.0, 0.3, 0.5, 0.7, 0.9]
        self.categories = categories
        self.road_structures = road_structures
        self.weather_conditions = weather_conditions
        self.time_zone = time_zone

        # Define category to cls (abbreviation) mapping
        self.category_to_cls = {
            "Sedan": 'sed',
            "Bus or Truck": 'bus',
            "Motorcycle": 'mot',
            "Bicycle": 'bic',
            "Bicycle Group": 'big',
            "Pedestrian": 'ped',
            "Pedestrian Group": 'peg',
            "Background": 'bg',
        }

    @property
    def categories(self):
        return {v: k for k, v in self._categories.items()}

    @categories.setter
    def categories(self, value: Dict[str, int]):
        if value is None:
            # Default categories of the K-Radar dataset
            self._categories = {
                0: "Sedan",
                1: "Bus or Truck",
                2: "Motorcycle",
                3: "Bicycle",
                4: "Bicycle Group",
                5: "Pedestrian",
                6: "Pedestrian Group",
                7: "Background"
            }

        elif len(value) != 8:
            raise ValueError(
                f"The categories property must provide a unique mapping "
                f"for each of the 8 classes but an input with "
                f"{len(value)} elements was given!"
            )

        elif isinstance(value, dict):
            self._categories = {v: k for k, v in value.items()}

        else:
            raise TypeError(
                f"The categories property must be of type 'dict' "
                f"but an input of type {type(value)} was given!"
            )

    @property
    def road_structures(self):
        return {v: k for k, v in self._road_structures.items()}

    @road_structures.setter
    def road_structures(self, value: Dict[str, int]):
        if value is None:
            # Default road structures of the K-Radar dataset
            self._road_structures = {
                0: "urban",
                1: "highway",
                2: "alleyway",
                3: "suburban",
                4: "university",
                5: "mountain",
                6: "parkinglots",
                7: "shoulder",
                8: "countryside"
            }

        elif len(value) != 8:
            raise ValueError(
                f"The road structures property must provide a unique mapping "
                f"for each of the 8 road structures but an input with "
                f"{len(value)} elements was given!"
            )

        elif isinstance(value, dict):
            self._road_structures = {v: k for k, v in value.items()}

        else:
            raise TypeError(
                f"The road structures property must be of type 'dict' "
                f"but an input of type {type(value)} was given!"
            )

    @property
    def weather_conditions(self):
        return {v: k for k, v in self._weather_conditions.items()}

    @weather_conditions.setter
    def weather_conditions(self, value: Dict[str, int]):
        if value is None:
            # Default weather conditions of the K-Radar dataset
            self._weather_conditions = {
                0: "normal",
                1: "overcast",
                2: "fog",
                3: "rain",
                4: "sleet",
                5: "lightsnow",
                6: "heavysnow",
            }

        elif len(value) != 7:
            raise ValueError(
                f"The weather conditions property must provide a unique "
                f"mapping for each of the 7 weather conditions but an input "
                f"with {len(value)} elements was given!"
            )

        elif isinstance(value, dict):
            self._weather_conditions = {v: k for k, v in value.items()}

        else:
            raise TypeError(
                f"The weather conditions property must be of type 'dict' "
                f"but an input of type {type(value)} was given!"
            )

    @property
    def time_zone(self):
        return {v: k for k, v in self._time_zone.items()}

    @time_zone.setter
    def time_zone(self, value: Dict[str, int]):
        if value is None:
            # Default time zones of the K-Radar dataset
            self._time_zone = {
                0: "day",
                1: "night",
            }

        elif len(value) != 2:
            raise ValueError(
                f"The time zone property must provide a unique mapping "
                f"for each of the 2 time zones but an input with"
                f"{len(value)} elements was given!"
            )

        elif isinstance(value, dict):
            self._time_zone = {v: k for k, v in value.items()}

        else:
            raise TypeError(
                f"The time zone property must be of type 'dict' "
                f"but an input of type {type(value)} was given!"
            )

    def __call__(self, *args, **kwargs) -> None:
        self.export(*args, **kwargs)

    @classmethod
    def from_config(cls, config:  Dict[str, Any]) -> KRadarExporter:  # noqa: F821
        return cls(
            conf_thrs=config['evaluate']['exporter'].get('conf_thrs'),
            categories=config['data'].get('categories'),
            road_structures=config['data'].get('road_structures'),
            weather_conditions=config['data'].get('weather_conditions'),
            time_zone=config['data'].get('time_zone')
        )

    @staticmethod
    def _get_dummy_object() -> List[str]:
        """Returns a dummy object as placeholder.

        If no model prediction or ground truth object is given
        for the current sample, a dummy object is exported as
        placeholder for the evaluation.

        Retruns:
            List of serialized dummy object values.
        """
        return ["dummy -1 -1 0 0 0 0 0 0 0 0 0 0 0 0 0"]

    @staticmethod
    def write(lines: List[str], dst: str) -> None:
        """Writes a list of srialized strings to a given file.

        Arguments:
            lines: Lines of serialized strings to add
                to the file given in dst.
            dst: Destination filename.
        """
        # Ensure that target folder exists
        os.makedirs(osp.dirname(dst), exist_ok=True)

        # Write lines to file
        with open(dst, 'a+') as f:
            f.writelines(s + "\n" for s in lines)

    def _construct_objects(self, objects: Dict[torch.Tensor],
                           conf_thr: float, pre: str = '') -> torch.Tensor:
        """Returns the required export properties of the given objects.

        The required export object properties are:
        name, truncated, occluded, alpha, bbox, bbox, bbox, bbox, h, w, l, y, z, x, theta

        Example:
        bus 0.00 0 0 50 50 150 150 3.7 2.8 12.11 -8.27 2.62 60.71 -0.02

        Argumens:
            objects: Dictionary of bounding box objects that contains these entries:
                "class": Bounding box class probabilities of shape (N, num_classes)
                "center": Bounding box center coordinates of shape (N, 3).
                "size": Bounding box size values of shape (N, 3).
                "angle": Bounding box orientation values of shape (N, 2).
            conf_thr: Confidence threshold to filter the objects for.
            pre: Prefix to add to the object keys.

        Returns:
            Tensor of object properties with shape (N, 15), where
            N is the number of objects.
        """
        # Adjust prefix
        pre = f"{pre}_" if pre else pre

        # Calculate object categories
        confidence, categories = torch.max(objects[f"{pre}class"], dim=-1)

        # Calculate yaw angle from sin and cos part
        angle = torch.atan2(objects[f'{pre}angle'][..., 0], objects[f'{pre}angle'][..., 1])

        # Discart background and ignore classes
        categories -= 1

        # Limit lables to kradar evaluation field of view (FoV)
        x_mask = (0 < objects[f"{pre}center"][:, 0]) & (objects[f"{pre}center"][:, 0] < 72)
        y_mask = (-6.4 < objects[f"{pre}center"][:, 1]) & (objects[f"{pre}center"][:, 1] < 6.4)
        z_mask = (-2.0 < objects[f"{pre}center"][:, 2]) & (objects[f"{pre}center"][:, 2] < 6.0)
        a_mask = (-50.0 < angle) & (angle < 50.0)
        fov_mask = x_mask & y_mask & z_mask & a_mask

        # Mask objects
        conf_mask = (confidence >= conf_thr)
        cls_mask = (categories >= 0)
        mask = cls_mask & conf_mask & fov_mask

        # Get number of valid objects
        N = torch.sum(mask)

        # Get device
        device = categories.device

        # Construct objects in specified format
        return torch.hstack([
            categories[mask].unsqueeze(-1),
            torch.zeros((N, 1), dtype=float, device=device),
            torch.zeros((N, 1), dtype=int, device=device),
            torch.zeros((N, 1), dtype=int, device=device),
            torch.tensor([[50, 50, 150, 150]], dtype=int, device=device).repeat(N, 1),
            torch.atleast_2d(objects[f"{pre}size"][mask, :][:, [2, 1, 0]]),
            torch.atleast_2d(objects[f"{pre}center"][mask, :][:, [1, 2, 0]]),
            angle[mask].unsqueeze(-1)
        ])

    def _serialize_description(self, description: torch.Tensor) -> List[str]:
        """Returns a serialized scene description.

        Arguments:
            description: Tensor of numerical scene description
                values with shape (3,).

        Returns:
            List of serialized scene descriptions according to the
            defined mappings with length 3.
        """
        description = description.detach().cpu().numpy()

        return [
            self._time_zone[int(description[1])],
            self._road_structures[int(description[0])],
            self._weather_conditions[int(description[2])]
        ]

    def _serialize_object(self, object: torch.Tensor) -> str:
        """Returns a serialized bounding box object.

        Arguments:
            object: Tensor of numerical object properties with shape (15,)
                in the following order: name, truncated, occluded, alpha,
                bbox, bbox, bbox, bbox, h, w, l, y, z, x, theta.

        Returns:
            String of serialized object properties in the same order
            as the input data.
        """
        return ' '.join([
            self.category_to_cls[self._categories[object[0]]],
            str(int(object[1])),
            str(int(object[2])),
            str(int(object[3])),
            str(int(object[4])),
            str(int(object[5])),
            str(int(object[6])),
            str(int(object[7])),
            str(round(object[8], 2)),
            str(round(object[9], 2)),
            str(round(object[10], 2)),
            str(round(object[11], 2)),
            str(round(object[12], 2)),
            str(round(object[13], 2)),
            str(round(object[14], 2)),
        ])

    def _serialize_objects(self, objects: torch.Tensor) -> List[str]:
        """Returns a list of serialized bounding box objects.

        Arguments:
            objects: Tensor of numerical object properties with shape (N, 15).

        Returns:
            List of serialized object properties with length N and properties
            name, truncated, occluded, alpha, bbox, bbox, bbox, bbox, h, w, l,
            y, z, x, theta.
        """
        # Convert tensor to numpy array
        objects = objects.detach().cpu().numpy()

        return [self._serialize_object(obj) for obj in objects]

    def _export_output_objects(self, objects: Dict[torch.Tensor],
                               conf_thr: float, step: int,
                               description: torch.Tensor, dst: str) -> None:
        """Exports the model prediction objects.

        Arguments:
            objects: Dictionary of bounding box objects that contains these entries:
                "class": Bounding box class probabilities of shape (N, num_classes)
                "center": Bounding box center coordinates of shape (N, 3).
                "size": Bounding box size values of shape (N, 3).
                "angle": Bounding box orientation values of shape (N, 2).
            conf_thr: Confidence threshold to filter the predictions for.
            step: Current model evaluation step.
            description: Tensor of scene descriptions.
            dst: Destination folder path to export the objects to.
        """
        # Construct, filter and serialize objects
        objects = self._construct_objects(objects, conf_thr)
        objects = self._serialize_objects(objects)

        if not objects:
            objects = self._get_dummy_object()

        # Serialize description
        description = self._serialize_description(description)

        # Add objects to all subsets in the description
        for desc in itertools.chain(['all'], description):
            # Define folder path
            folder = osp.join(dst, desc, 'preds')

            # Write objects to file
            self.write(objects, osp.join(folder, f"{str(step).zfill(6)}.txt"))

    def _export_target_objects(self, objects: Dict[torch.Tensor],
                               conf_thr: float, step: int,
                               description: torch.Tensor, dst: str) -> None:
        """Exports the ground truth objects.

        Arguments:
            objects: Dictionary of bounding box objects that contains these entries:
                "class": Bounding box class probabilities of shape (N, num_classes)
                "center": Bounding box center coordinates of shape (N, 3).
                "size": Bounding box size values of shape (N, 3).
                "angle": Bounding box orientation values of shape (N, 2).
            conf_thr: Confidence threshold to filter the predictions for.
            step: Current model evaluation step.
            description: Tensor of scene descriptions.
            dst: Destination folder path to export the objects to.
        """
        # Construct, filter and serialize objects
        objects = self._construct_objects(objects, conf_thr, pre='gt')
        objects = self._serialize_objects(objects)

        if not objects:
            objects = self._get_dummy_object()

        # Serialize description
        description = self._serialize_description(description)

        # Add objects to all subsets in the description
        for desc in itertools.chain(['all'], description):
            # Define folder path
            folder = osp.join(dst, desc)

            # Write description to file
            self.write(description, osp.join(folder, 'desc', f"{str(step).zfill(6)}.txt"))

            # Write objects to file
            self.write(objects, osp.join(folder, 'gts', f"{str(step).zfill(6)}.txt"))

            # Add step to value file
            self.write([str(step).zfill(6)], osp.join(folder, "val.txt"))

    def _export_output_batch(self, batch: Dict[str, torch.Tensor],
                             conf_thr: float, step: int,
                             description: List[str], dst: str) -> None:
        """Exports the batched model prediction objects.

        Arguments:
            batch: Dictionaries of bounding box objects that contains these entries:
                "class": Bounding box class probabilities of shape (B, N, num_classes)
                "center": Bounding box center coordinates of shape (B, N, 3).
                "size": Bounding box size values of shape (B, N, 3).
                "angle": Bounding box orientation values of shape (B, N, 2).
            conf_thr: Confidence threshold to filter the predictions for.
            step: Current model evaluation step.
            description: List of scene description tensors with length B.
            dst: Destination folder path to export the objects to.
        """
        # Decollate batch
        batch: List[Dict[str, torch.Tensor]] = decollate_batch(batch, detach=True, pad=False)

        for i, (data, desc) in enumerate(zip(batch, description)):
            self._export_output_objects(data, conf_thr, step + i, desc, dst)

    def _export_target_batch(self, batch: List[Dict[str, torch.Tensor]],
                             conf_thr: float, step: int, dst: str) -> List[torch.Tensor]:
        """Exports the batched ground truth objects.

        Arguments:
            objects: List of dictionaries with objects that contains these entries:
                "class": Bounding box class probabilities of shape (N, num_classes)
                "center": Bounding box center coordinates of shape (N, 3).
                "size": Bounding box size values of shape (N, 3).
                "angle": Bounding box orientation values of shape (N, 2).
            conf_thr: Confidence threshold to filter the predictions for.
            step: Current model evaluation step.
            description: List of scene description tensors with length B.
            dst: Destination folder path to export the objects to.
        """
        # Initialize description
        description = []

        for i, data in enumerate(batch):
            # Get description
            desc = data['description']
            description.append(desc)

            # Export objects
            self._export_target_objects(data, conf_thr, step + i, desc, dst)

        return description

    def export(self,
               outputs: Dict[str, torch.Tensor],
               targets: List[Dict[str, torch.Tensor]],
               step: int,
               dst: str) -> None:
        """Exports predictions and lables in the K-Radar evaluation format.

        Arguments:
            outputs: Dictionaries of model predictions that contains these entries:
                "class": Bounding box class probabilities of shape (B, N, num_classes)
                "center": Bounding box center coordinates of shape (B, N, 3).
                "size": Bounding box size values of shape (B, N, 3).
                "angle": Bounding box orientation values of shape (B, N, 2).
            targets: List of ground truth dicts that contains these entries:
                "class": Bounding box class probabilities of shape (N, num_classes)
                "center": Bounding box center coordinates of shape (N, 3).
                "size": Bounding box size values of shape (N, 3).
                "angle": Bounding box orientation values of shape (N, 2).
            step: Current evaluation step.
            dst: Destination folder path to export the data to.
        """
        for conf_thr in self.conf_thrs:
            # Define folder path
            folder = osp.join(dst, 'exports', 'kradar', str(conf_thr))

            # Export ground truth
            description = self._export_target_batch(targets, conf_thr, step, folder)

            # Export predictions
            self._export_output_batch(outputs, conf_thr, step, description, folder)


def build_kradar(*args, **kwargs):
    return KRadarExporter.from_config(*args, **kwargs)
