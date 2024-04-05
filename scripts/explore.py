import argparse

import numpy as np

from dprt.datasets import prepare
from dprt.datasets.kradar.utils import radar_info
from dprt.utils import visu
from dprt.utils.config import load_config
from dprt.utils.geometry import get_transformation, transform_boxes, transform_points


def main(src: str, cfg: str, dst: str):
    # Load dataset configuration
    config = load_config(cfg)

    # Create preperator instance
    preperator = prepare(config['dataset'], config)

    # Get single dataset sample
    dataset_paths = preperator.get_dataset_paths(src)
    sequence_paths = preperator.get_sequence_paths(next(iter(dataset_paths['train'].values())))
    for sample in sequence_paths.values():
        # Load bounding boxes
        boxes = preperator.get_boxes(sample['label'])

        if boxes.shape[0] > 1:
            break

    # Get calibration information
    trans, rot = preperator.get_calibration(sample['calibration'])

    # Load front camera data
    camera_front, _ = preperator.get_camera_data(sample['camera_front'])

    # Visualize camera data
    visu.visu_camera_data(camera_front)

    # Visualize lidar data
    point_cloud = preperator.get_lidar_data(sample['os2'])

    # Visualize lidar data
    # visu.visu_lidar_data(point_cloud, boxes, xlim=[-100, 100], ylim=[-100, 100])

    # Load radar tesseract with shape (doppler, range, elevation, azimuth)
    tesseract = preperator.get_radar_tesseract(sample['radar_tesseract'])

    # Get radar sensor information
    raster = {
        'r': radar_info.range_raster,
        'e': radar_info.elevation_raster,
        'a': radar_info.azimuth_raster,
        'd': radar_info.doppler_raster
    }

    # Transform boxes to radar frame
    tm =get_transformation(trans, rot, inverse=True)
    boxes = transform_boxes(boxes, tm)

    # Define region of interest
    point_cloud = transform_points(point_cloud, tm)

    # Visualize radar data
    visu.visu_radar_tesseract(tesseract, dims='ea', raster=raster, points=point_cloud, boxes=boxes, roi=True, cart=True, aggregation_func=np.max)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DPRT data preprocessing')
    parser.add_argument('--src', type=str, default='/data/kradar/raw',
                        help="Path to the raw dataset folder.")
    parser.add_argument('--cfg', type=str, default='/app/config/kradar.json',
                        help="Path to the configuration file.")
    parser.add_argument('--dst', type=str, default='/data/kradar/processed',
                        help="Path to save the processed dataset.")
    args = parser.parse_args()

    main(src=args.src, cfg=args.cfg, dst=args.dst)
