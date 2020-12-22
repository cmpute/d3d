'''
This script convert Waymo dataset tarballs to zipfiles or regular directory split by collection segments.
'''

import json
import os
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable logging
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # disable GPU usage

import tensorflow as tf
from d3d.dataset.base import NumberPool
from waymo_open_dataset import dataset_pb2, label_pb2
from waymo_open_dataset.utils import (frame_utils, range_image_utils,
                                      transform_utils)


camera_name_map = {
    dataset_pb2.CameraName.Name.FRONT: "front",
    dataset_pb2.CameraName.Name.FRONT_LEFT: "front_left",
    dataset_pb2.CameraName.Name.FRONT_RIGHT: "front_right",
    dataset_pb2.CameraName.Name.SIDE_LEFT: "side_left",
    dataset_pb2.CameraName.Name.SIDE_RIGHT: "side_right"
}

lidar_name_map = {
    dataset_pb2.LaserName.Name.TOP: "top",
    dataset_pb2.LaserName.Name.FRONT: "front",
    dataset_pb2.LaserName.Name.SIDE_LEFT: "side_left",
    dataset_pb2.LaserName.Name.SIDE_RIGHT: "side_right",
    dataset_pb2.LaserName.Name.REAR: "rear"
}

label_name_map = {
    label_pb2.Label.Type.TYPE_UNKNOWN: "Unknown",
    label_pb2.Label.Type.TYPE_VEHICLE: "Vehicle",
    label_pb2.Label.Type.TYPE_PEDESTRIAN: "Pedestrian",
    label_pb2.Label.Type.TYPE_SIGN: "Sign",
    label_pb2.Label.Type.TYPE_CYCLIST: "Cyclist"
}


# ======= Modified from waymo_open_dataset.utils.frame_utils to report intensity =======

def convert_range_image_to_point_cloud(frame,
                                       range_images,
                                       camera_projections,
                                       range_image_top_pose,
                                       ri_index=0):
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    points = []
    cp_points = []
    channels = []

    frame_pose = tf.convert_to_tensor(
        value=np.reshape(np.array(frame.pose.transform), [4, 4]))
    # [H, W, 6]
    range_image_top_pose_tensor = tf.reshape(
        tf.convert_to_tensor(value=range_image_top_pose.data),
        range_image_top_pose.shape.dims)
    # [H, W, 3, 3]
    range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
        range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1],
        range_image_top_pose_tensor[..., 2])
    range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
    range_image_top_pose_tensor = transform_utils.get_transform(
        range_image_top_pose_tensor_rotation,
        range_image_top_pose_tensor_translation)
    for c in calibrations:
        range_image = range_images[c.name][ri_index]
        if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
            beam_inclinations = range_image_utils.compute_inclination(
                tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                height=range_image.shape.dims[0])
        else:
            beam_inclinations = tf.constant(c.beam_inclinations)

        beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
        extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(value=range_image.data), range_image.shape.dims)
        pixel_pose_local = None
        frame_pose_local = None
        if c.name == dataset_pb2.LaserName.TOP:
            pixel_pose_local = range_image_top_pose_tensor
            pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
            frame_pose_local = tf.expand_dims(frame_pose, axis=0)
        range_image_mask = range_image_tensor[..., 0] > 0
        range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
                tf.expand_dims(range_image_tensor[..., 0], axis=0),
                tf.expand_dims(extrinsic, axis=0),
                tf.expand_dims(tf.convert_to_tensor(value=beam_inclinations), axis=0),
                pixel_pose=pixel_pose_local,
                frame_pose=frame_pose_local)

        range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
        points_tensor = tf.gather_nd(range_image_cartesian, tf.compat.v1.where(range_image_mask))
        channels_tensor = tf.gather_nd(range_image_tensor, tf.compat.v1.where(range_image_mask)) # <- modified

        cp = camera_projections[c.name][0]
        cp_tensor = tf.reshape(tf.convert_to_tensor(value=cp.data), cp.shape.dims)
        cp_points_tensor = tf.gather_nd(cp_tensor, tf.compat.v1.where(range_image_mask))
        points.append(points_tensor.numpy())
        cp_points.append(cp_points_tensor.numpy())
        channels.append(channels_tensor.numpy()[:, [1,2]]) # <- modified

    return points, cp_points, channels

# =========================================================================

def add_property(proto, dict_, name):
    if proto.HasField(name):
        dict_[name] = getattr(proto, name)

def write_file(out_archive, fname, data: bytes):
    if isinstance(out_archive, Path):
        ofile = out_archive / fname
        ofile.parent.mkdir(exist_ok=True)
        ofile.write_bytes(data)
    else: # zipfile
        out_archive.writestr(fname, data)

def write_file_np(out_archive, fname, data: np.ndarray):
    if isinstance(out_archive, Path):
        ofile = out_archive / fname
        ofile.parent.mkdir(exist_ok=True)
        data.tofile(ofile)
    else: # zipfile
        out_archive.writestr(fname, data.tobytes())

def save_context(frame, frame_count, out_archive):
    # save stats
    stats = {}
    add_property(frame.context.stats, stats, "time_of_day")
    add_property(frame.context.stats, stats, "location")
    add_property(frame.context.stats, stats, "weather")

    for objcount in frame.context.stats.laser_object_counts:
        if "laser_object_counts" not in stats:
            stats["laser_object_counts"] = {}
        stats['laser_object_counts'][label_name_map[objcount.type]] = objcount.count

    for objcount in frame.context.stats.camera_object_counts:
        if "camera_object_counts" not in stats:
            stats["camera_object_counts"] = {}
        stats['camera_object_counts'][label_name_map[objcount.type]] = objcount.count

    stats['frame_count'] = frame_count
    write_file(out_archive, "context/stats.json", json.dumps(stats).encode())

    # save calibrations
    calibs = {}
    for calib_object in frame.context.camera_calibrations:
        calib_dict = dict(
            intrinsic=list(calib_object.intrinsic),
            extrinsic=list(calib_object.extrinsic.transform),
            width=calib_object.width,
            height=calib_object.height,
            # rolling_shutter_direction is currently ignored
        )
        calibs[camera_name_map[calib_object.name]] = calib_dict
    write_file(out_archive, "context/calib_cams.json", json.dumps(calibs).encode())

    calibs = {}
    for calib_object in frame.context.laser_calibrations:
        calib_dict = dict(
            extrinsic=list(calib_object.extrinsic.transform),
            # beam_inclinations are ignored
        )
        calibs[lidar_name_map[calib_object.name]] = calib_dict
    write_file(out_archive, "context/calib_lidars.json", json.dumps(calibs).encode())

def save_timestamp(frame, frame_idx, out_archive):
    ts = str(frame.timestamp_micros).encode()
    write_file(out_archive, "timestamp/%04d.txt" % frame_idx, ts)

def save_pose(frame, frame_idx, out_archive):
    values = np.array(frame.pose.transform).astype("f8").reshape(4, 4)
    write_file_np(out_archive, "pose/%04d.bin" % frame_idx, values)

def save_image(frame, frame_idx, out_archive):
    for image in frame.images:
        fname = "camera_%s/%04d.jpg" % (camera_name_map[image.name], frame_idx)
        write_file(out_archive, fname, image.image)

def save_point_cloud(frame, frame_idx, out_archive):
    range_images, camera_projections, range_image_top_pose =\
        frame_utils.parse_range_image_and_camera_projection(frame)
    points, cp_points, channels = convert_range_image_to_point_cloud(
        frame, range_images, camera_projections, range_image_top_pose)
    points_ri2, cp_points_ri2, channels_ri2 = convert_range_image_to_point_cloud(
        frame, range_images, camera_projections, range_image_top_pose, ri_index=1)

    for i in range(5):
        name = lidar_name_map[i+1]
        cloud = np.hstack((points[i], channels[i]))
        write_file_np(out_archive, "lidar_%s/%04d.bin" % (name, frame_idx), cloud)
        cloud_ri2 = np.hstack((points_ri2[i], channels_ri2[i]))
        write_file_np(out_archive, "lidar_%s_ri2/%04d.bin" % (name, frame_idx), cloud_ri2)

def save_labels(frame, frame_idx, out_archive):
    # labels in lidar frame
    label_list = []
    for label in frame.laser_labels:
        label_obj = dict(
            center=[label.box.center_x, label.box.center_y, label.box.center_z],
            size=[label.box.length, label.box.width, label.box.height],
            heading=label.box.heading,
            label=label_name_map[label.type],
            id=label.id,
            detection_difficulty_level=label.detection_difficulty_level,
            tracking_difficulty_level=label.tracking_difficulty_level
        )
        label_list.append(label_obj)
    label_json = json.dumps(label_list).encode()
    write_file(out_archive, "label_lidars/%04d.json" % frame_idx, label_json)

    # labels in camera frames
    for label_tuple in frame.camera_labels:
        label_list = []
        name = camera_name_map[label_tuple.name]
        for label in label_tuple.labels:
            label_obj = dict(
                center=[label.box.center_x, label.box.center_y],
                size=[label.box.length, label.box.width],
                label=label_name_map[label.type],
                id=label.id,
                detection_difficulty_level=label.detection_difficulty_level,
                tracking_difficulty_level=label.tracking_difficulty_level
            )
            label_list.append(label_obj)
        label_json = json.dumps(label_list).encode()
        write_file(out_archive, "label_camera_%s/%04d.json" % (name, frame_idx), label_json)

    # no_label_zones are ignored

def convert_tfrecord(ntqdm, input_file, output_path: Path, zip_output=False, delele_input=True):
    output_path.mkdir(exist_ok=True, parents=True)
    dataset = tf.data.TFRecordDataset(str(input_file), compression_type='')
    archive = None

    # parse compression
    compression = None
    if zip_output:
        if zip_output == "deflated":
            compression = zipfile.ZIP_DEFLATED
        elif zip_output == "bzip2":
            compression = zipfile.ZIP_BZIP2
        elif zip_output == "lzma":
            compression = zipfile.ZIP_LZMA
        else:
            compression = zipfile.ZIP_STORED

    disp = os.path.split(input_file)[1]
    disp = "Converting %s..." % disp[8:disp.find("_")]
    for idx, data in tqdm(enumerate(dataset), desc=disp, position=ntqdm, unit="frames", dynamic_ncols=True, leave=False):
        if idx > 9999:
            raise RuntimeError("Frame index is larger than file name capacity!")

        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        if archive is None:
            if zip_output:
                archive = zipfile.ZipFile(output_path / (frame.context.name + ".zip"), "w",
                    compression=compression)
            else:
                archive = output_path / frame.context.name
                archive.mkdir(exist_ok=True)

        save_timestamp(frame, idx, archive)
        save_image(frame, idx, archive)
        save_point_cloud(frame, idx, archive)
        save_labels(frame, idx, archive)
        save_pose(frame, idx, archive)
    save_context(frame, idx, archive) # save metadata at last

    if archive is not None and zip_output:
        archive.close()
    if delele_input: # delete intermediate file
        os.remove(input_file)

    return idx

def convert_dataset_inpath(input_path, output_path, nworkers=8, debug=False, zip_output=False):
    pool = NumberPool(processes=nworkers, offset=1)
    temp_dir = tempfile.mkdtemp()
    total_records = 0
    print("Extracting tfrecords from tarballs to %s..." % temp_dir)

    try:
        for tar_name in tqdm(os.listdir(input_path), desc="Extract tfrecords",
                             position=0, unit="tars", leave=False, dynamic_ncols=True):
            if os.path.splitext(tar_name)[1] != ".tar":
                continue

            phase = tar_name.split('_')[0]
            tarf = tarfile.open(name=os.path.join(input_path, tar_name), mode='r|*')
            for member in tarf:
                if os.path.splitext(member.name)[1] != ".tfrecord":
                    continue

                pool.wait_for_once(margin=3) # prevent extracted too much records into temp folder
                tarf.extract(member, temp_dir)

                pool.apply_async(convert_tfrecord,
                    (Path(temp_dir, member.name),
                     Path(output_path, phase),
                     zip_output)
                )
                total_records += 1

                if debug and total_records > 1: # only convert two tfrecord when debugging
                    break
            tarf.close()

            if debug: # only convert one tarball when debugging
                break

        pool.close()
        pool.join()

    finally:
        shutil.rmtree(temp_dir)
        print("Terminated, cleaned temporary files")

def main():
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Convert waymo dataset tarballs to normal zip files with numpy arrays.')

    parser.add_argument('input', type=str,
        help='Input directory')
    parser.add_argument('-o', '--output', type=str,
        help='Output file (in .zip format) or directory. If not provided, it will be the same as input')
    parser.add_argument('-d', '--debug', action="store_true",
        help='Run the script in debug mode, only convert part of the tarballs')
    parser.add_argument('-p', '--parallel-workers', type=int, dest="workers", default=8,
        help="Number of parallet workers to convert tfrecord")
    parser.add_argument('-z', '--zip', action="store_true",
        help="Convert the result into zip files rather than flat directory")
    parser.add_argument('-c', '--compression', type=str, default='stored', choices=['stored', 'deflated', 'bzip2', 'lzma'],
        help="Choose zip compression type")
    args = parser.parse_args()

    convert_dataset_inpath(args.input, args.output or args.input, nworkers=args.workers,
                           debug=args.debug, zip_output=args.compression if args.zip else False)

if __name__ == "__main__":
    main()
