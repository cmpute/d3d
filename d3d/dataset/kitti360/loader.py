import logging
import shutil
import tempfile
import time
from collections import OrderedDict
from io import BytesIO
from itertools import chain
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import tqdm
from d3d.abstraction import (EgoPose, ObjectTag, ObjectTarget3D, Target3DArray,
                             TransformSet)
from d3d.dataset.base import (NumberPool, TrackingDatasetBase, check_frames,
                              expand_idx, expand_idx_name, split_trainval)
from d3d.dataset.kitti360.utils import (Kitti360Class, id2label, kittiId2label,
                                        load_bboxes, load_sick_scan)
from d3d.dataset.kitti.utils import (load_calib_file, load_image,
                                     load_timestamps, load_velo_scan)
from d3d.dataset.zip import PatchedZipFile
from filelock import FileLock
from PIL import Image
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

_logger = logging.getLogger("d3d")

class KITTI360Loader(TrackingDatasetBase):
    """
    Load KITTI-360 dataset into a usable format.
    The dataset structure should follow the official documents.

    # Zip Files
    - calibration.zip
    - data_3d_bboxes.zip
    - data_3d_semantics.zip
    - data_poses.zip
    - data_timestamps_sick.zip
    - data_timestamps_velodyne.zip
    - 2013_05_28_drive_0000_sync_sick.zip
    - 2013_05_28_drive_0000_sync_velodyne.zip
    - ...

    # Unzipped Structure
    - <base_path directory>
        - calibration
        - data_2d_raw
            - 2013_05_28_drive_0000_sync
            - ...
        - data_2d_semantics
            - 2013_05_28_drive_0000_sync
            - ...
        - data_3d_raw
            - 2013_05_28_drive_0000_sync
            - ...
        - data_3d_semantics
            - 2013_05_28_drive_0000_sync
            - ...
    """
    VALID_CAM_NAMES = ['cam1', 'cam2', 'cam3', 'cam4'] # cam 1,2 are persective
    VALID_LIDAR_NAMES = ['velo'] # velo stands for velodyne
    VALID_OBJ_CLASSES = Kitti360Class
    # TODO: add 'sick' frame to the list after the loader for sick points is completed

    FRAME_PATH_MAP = dict(
        sick=("data_3d_raw", "sick_points"    , "data"     , "data_timestamps_sick.zip"),
        velo=("data_3d_raw", "velodyne_points", "data"     , "data_timestamps_velodyne.zip"),
        cam1=("data_2d_raw", "image_00"       , "data_rect", "data_timestamps_perspective.zip"),
        cam2=("data_2d_raw", "image_01"       , "data_rect", "data_timestamps_perspective.zip"),
        cam3=("data_2d_raw", "image_02"       , "data_rgb" , "data_timestamps_fisheye.zip"),
        cam4=("data_2d_raw", "image_03"       , "data_rgb" , "data_timestamps_fisheye.zip"),
    )

    def __init__(self, base_path, phase="training", inzip=False,
                 trainval_split=1, trainval_random=False, nframes=0):
        """
        :param phase: training, validation or testing
        :param trainval_split: placeholder for interface compatibility with other loaders
        :param visible_range: range for visible objects. Objects beyond that distance will be removed when reporting
        """

        if phase not in ['training', 'validation', 'testing']:
            raise ValueError("Invalid phase tag")

        self.base_path = Path(base_path)
        self.inzip = inzip
        self.phase = phase
        self.nframes = nframes

        # count total number of frames
        frame_count = dict()
        _dates = ["2013_05_28"]
        if self.inzip:
            _archives = [
                # ("sick", ".bin"), # SICK points are out of synchronization
                ("velodyne", ".bin"),
                ("image_00", ".png"), ("image_01", ".png"),
                ("image_02", ".png"), ("image_03", ".png")
            ]
            for aname, ext in _archives:
                globs = [self.base_path.glob(f"{date}_drive_*_sync_{aname}.zip")
                            for date in _dates]
                for archive in chain(*globs):
                    with ZipFile(archive) as data:
                        data_files = (name for name in data.namelist() if name.endswith(ext))

                        seq = archive.stem[:archive.stem.rfind("_")]
                        frame_count[seq] = sum(1 for _ in data_files)

                # successfully found
                if len(frame_count) > 0:
                    break
        else:
            _folders = [
                # ("data_3d_raw", "sick_points", "data"), # SICK points are out of synchronization
                ("data_3d_raw", "velodyne_points", "data"),
                ("data_2d_raw", "image_00", "data_rect"),
                ("data_2d_raw", "image_01", "data_rect"),
                ("data_2d_raw", "image_02", "data_rgb"),
                ("data_2d_raw", "image_03", "data_rgb"),
            ]
            for ftype, fname, dname in _folders:
                globs = [self.base_path.glob(f"{ftype}/{date}_drive_*_sync")
                            for date in _dates]
                for archive in chain(*globs):
                    if not archive.is_dir(): # skip calibration files
                        continue
                    if not (archive / fname / "data").exists():
                        continue

                    seq = archive.name
                    frame_count[seq] = sum(1 for _ in (archive / fname / dname).iterdir())

                # successfully found
                if len(frame_count) > 0:
                    break

        if not frame_count:
            raise ValueError("Cannot parse dataset, please check path, inzip option and file structure")
        self.frame_dict = OrderedDict(frame_count)

        total_count = sum(frame_count.values()) - nframes * len(frame_count)
        self.frames = split_trainval(phase, total_count, trainval_split, trainval_random)
        self._poses_idx = {} # store loaded poses indices
        self._poses_t = {} # pose translation
        self._poses_r = {} # pose rotation
        self._3dobjects_cache = {} # store loaded object labels
        self._3dobjects_mapping =  {} # store frame to objects mapping
        self._timestamp_cache = {} # store timestamps

        # load calibration
        self._calibration = None
        self._preload_calib()

    def __len__(self):
        return len(self.frames)

    @property
    def sequence_ids(self):
        return list(self.frame_dict.keys())

    @property
    def sequence_sizes(self):
        return dict(self.frame_dict)

    def _locate_frame(self, idx):
        # use underlying frame index
        idx = self.frames[idx]

        for k, v in self.frame_dict.items():
            if idx < (v - self.nframes):
                return k, idx
            idx -= (v - self.nframes)
        raise ValueError("Index larger than dataset size")

    @expand_idx_name(VALID_CAM_NAMES)
    def camera_data(self, idx, names='cam1'):
        seq_id, frame_idx = idx

        _, folder_name, dname, _ = self.FRAME_PATH_MAP[names]
        fname = Path(seq_id, folder_name, dname, '%010d.png' % frame_idx)
        if self._return_file_path:
            return self.base_path / "data_2d_raw" / fname

        if self.inzip:
            with PatchedZipFile(self.base_path / f"{seq_id}_{folder_name}.zip", to_extract=fname) as source:
                return load_image(source, fname, gray=False)
        else:
            return load_image(self.base_path / "data_2d_raw", fname, gray=False)

    @expand_idx_name(['velo'])
    def lidar_data(self, idx, names='velo'):
        seq_id, frame_idx = idx
        
        if names == "velo":
            # load velodyne points
            fname = Path(seq_id, "velodyne_points", "data", '%010d.bin' % frame_idx)
            if self._return_file_path:
                return self.base_path / "data_3d_raw" / fname

            if self.inzip:
                with PatchedZipFile(self.base_path / f"{seq_id}_velodyne.zip", to_extract=fname) as source:
                    return load_velo_scan(source, fname)
            else:
                return load_velo_scan(self.base_path / "data_3d_raw", fname)

        elif names == "sick":
            # TODO: sick data is not supported due to synchronization issue currently
            #       official way to accumulate SICK data is by linear interpolate between adjacent point clouds
            fname = Path(seq_id, "sick_points", "data", '%010d.bin' % frame_idx)
            if self._return_file_path:
                return self.base_path / "data_3d_raw" / fname

            if self.inzip:
                with PatchedZipFile(self.base_path / f"{seq_id}_sick.zip", to_extract=fname) as source:
                    return load_sick_scan(source, fname)
            else:
                return load_sick_scan(self.base_path / "data_3d_raw", fname)
            
    def _preload_3dobjects(self, seq_id):
        assert self.phase in ["training", "validation"], "Testing set doesn't contains label"
        if seq_id in self._3dobjects_mapping:
            return

        fname = Path("data_3d_bboxes", "train", f"{seq_id}.xml")
        if self.inzip:
            with PatchedZipFile(self.base_path / "data_3d_bboxes.zip", to_extract=fname) as source:
                objlist, fmap = load_bboxes(source, fname)
        else:
            objlist, fmap = load_bboxes(self.base_path, fname)
            
        self._3dobjects_cache[seq_id] = objlist
        self._3dobjects_mapping[seq_id] = fmap

    @expand_idx
    def annotation_3dobject(self, idx, raw=False, visible_range=80): # TODO: it seems that dynamic objects need interpolation
        assert not self._return_file_path, "The annotation is not in a single file!"
        seq_id, frame_idx = idx
        self._preload_3dobjects(seq_id)
        objects = [self._3dobjects_cache[seq_id][i.data]
                    for i in self._3dobjects_mapping[seq_id][frame_idx]]
        if raw:
            return objects

        self._preload_poses(seq_id)
        pr, pt = self._poses_r[seq_id][frame_idx], self._poses_t[seq_id][frame_idx]

        boxes = Target3DArray(frame="pose")
        for box in objects:
            RS, T = box.transform[:3, :3], box.transform[:3, 3]
            S = np.linalg.norm(RS, axis=0) # scale
            R = Rotation.from_matrix(RS / S) # rotation
            R = pr.inv() * R # relative rotation
            T = pr.inv().as_matrix().dot(T - pt) # relative translation

            # skip static objects beyond vision
            if np.linalg.norm(T) > visible_range:
                continue

            global_id = box.semanticID * 1000 + box.instanceID
            tag = ObjectTag(kittiId2label[box.semanticId].name, Kitti360Class)
            boxes.append(ObjectTarget3D(T, R, S, tag, tid=global_id))
        return boxes

    def _preload_calib(self):
        # load data
        import yaml
        if self.inzip:
            source = ZipFile(self.base_path / "calibration.zip")
        else:
            source = self.base_path

        cam2pose = load_calib_file(source, "calibration/calib_cam_to_pose.txt")
        perspective = load_calib_file(source, "calibration/perspective.txt")
        if self.inzip:
            cam2velo = np.fromstring(source.read("calibration/calib_cam_to_velo.txt"), sep=" ")
            sick2velo = np.fromstring(source.read("calibration/calib_sick_to_velo.txt"), sep=" ")
            intri2 = yaml.safe_load(source.read("calibration/image_02.yaml")[10:]) # skip header
            intri3 = yaml.safe_load(source.read("calibration/image_03.yaml")[10:])
        else:
            cam2velo = np.loadtxt(source / "calibration/calib_cam_to_velo.txt")
            sick2velo = np.loadtxt(source / "calibration/calib_sick_to_velo.txt")
            intri2 = yaml.safe_load((source / "calibration/image_02.yaml").read_text()[10:])
            intri3 = yaml.safe_load((source / "calibration/image_03.yaml").read_text()[10:])

        if self.inzip:
            source.close()

        # parse calibration
        calib = TransformSet("pose")
        calib.set_intrinsic_lidar("velo")
        calib.set_intrinsic_lidar("sick")
        calib.set_intrinsic_camera("cam1", perspective["P_rect_00"].reshape(3, 4), perspective["S_rect_00"], rotate=False)
        calib.set_intrinsic_camera("cam2", perspective["P_rect_01"].reshape(3, 4), perspective["S_rect_01"], rotate=False)

        def parse_mei_camera(intri):
            size = [intri['image_width'], intri['image_height']]
            distorts = intri['distortion_parameters']
            distorts = np.array([distorts['k1'], distorts['k2'], distorts['p1'], distorts['p2']])
            projection = intri['projection_parameters']
            pmatrix = np.diag([projection["gamma1"], projection["gamma2"], 1])
            pmatrix[0, 2] = projection["u0"]
            pmatrix[1, 2] = projection["v0"]
            return size, pmatrix, distorts, intri['mirror_parameters']['xi']            

        S, P, D, xi = parse_mei_camera(intri2)
        calib.set_intrinsic_camera("cam3", P, S, distort_coeffs=D, intri_matrix=P, mirror_coeff=xi)
        S, P, D, xi = parse_mei_camera(intri3)
        calib.set_intrinsic_camera("cam4", P, S, distort_coeffs=D, intri_matrix=P, mirror_coeff=xi)
        
        calib.set_extrinsic(cam2pose["image_00"].reshape(3, 4), frame_from="cam1")
        calib.set_extrinsic(cam2pose["image_01"].reshape(3, 4), frame_from="cam2")
        calib.set_extrinsic(cam2pose["image_02"].reshape(3, 4), frame_from="cam3")
        calib.set_extrinsic(cam2pose["image_03"].reshape(3, 4), frame_from="cam4")
        calib.set_extrinsic(cam2velo.reshape(3, 4), frame_from="cam1", frame_to="velo")
        calib.set_extrinsic(sick2velo.reshape(3, 4), frame_from="sick", frame_to="velo")
        self._calibration = calib

    def calibration_data(self, idx):
        return self._calibration

    # TODO: fix the points (with large distance?) with bounding box
    # TODO: support sick points
    def _parse_semantic_ply(self, ntqdm, seq: str, fname: Path, dynamic: bool, result_path: Path, expand_frames: int):
        # match point cloud in aggregated semantic point clouds
        import pcl
        from sklearn.neighbors import KDTree

        fstart, fend = fname.stem.split('_')
        fstart, fend = int(fstart), int(fend)
        fstart, fend = max(fstart - expand_frames, 0), min(fend + expand_frames, self.sequence_sizes[seq])
        frame_desc = "%s frames %d-%d" % ("dynamic" if dynamic else "static", fstart, fend)

        _logger.debug("loading semantics for " + frame_desc)
        semantics = pcl.io.load_ply(str(fname))
        if len(semantics) == 0:
            return

        # create fast semantic id to Kitti360Class id mapping
        idmap = np.zeros(max(id2label.keys()) + 1, dtype='u1')
        for i in range(len(idmap)):
            idmap[i] = id2label[i].name.value
        
        if dynamic:
            timestamps = semantics.to_ndarray()['timestamp'].flatten()
        else:
            tree = KDTree(semantics.xyz)

        for i in tqdm.trange(fstart, fend, desc=frame_desc, position=ntqdm, leave=False):
            cloud = self.lidar_data((seq, i), names="velo", bypass=True)
            cloud = self._calibration.transform_points(cloud[:, :3], frame_to="pose", frame_from="velo")
            cloud = cloud.dot(self._poses_r[seq][i].as_matrix().T) + self._poses_t[seq][i]

            if dynamic:
                cur_semantics = semantics[timestamps == i]
                if len(cur_semantics) == 0:
                    continue
                tree = KDTree(cur_semantics.xyz)
                distance, sidx = tree.query(cloud[:, :3], return_distance=True)
                selected = cur_semantics.to_ndarray()[sidx]
            else:
                distance, sidx = tree.query(cloud[:, :3], return_distance=True)
                selected = semantics.to_ndarray()[sidx]
            distance = distance.flatten()

            rgb = selected["rgb"].flatten()
            slabels = selected["semantic"].flatten()
            slabels = idmap[slabels]
            ilabels = selected["instance"].flatten().astype('u2')
            visible = selected["visible"].flatten().astype(bool)

            label_path = result_path / ("%010d.npz" % i)
            dist_path = result_path / ("%010d.dist.npy" % i)
            lock_path = FileLock(result_path / ("%010d.lock" % i))
            
            with lock_path:
                if dist_path.exists():
                    old_distance = np.load(dist_path)
                    update_mask = (distance < old_distance)
                    distance = np.where(update_mask, distance, old_distance)

                    old_labels = np.load(label_path)
                    rgb = np.where(update_mask, rgb, old_labels['rgb'])
                    slabels = np.where(update_mask, slabels, old_labels["semantic"])
                    ilabels = np.where(update_mask, ilabels, old_labels["instance"])
                    visible = np.where(update_mask, visible, old_labels["visible"])

                np.savez(label_path, rgb=rgb, semantic=slabels, instance=ilabels, visible=visible)
                np.save(dist_path, distance)

    def _preload_3dsemantics(self, seq, nworkers=7, expand_frames=150):
        """
        This method will convert the combined semantic point cloud back in to frames

        expand_frames: number of expanded frames to be considered based on original sequence split
            Better painting results will be generated with larger expansion, but it will be slower
        """
        if self.inzip:
            if (self.base_path / "data_3d_semantics_indexed.zip").exists():
                return
            result_path = Path(tempfile.mkdtemp())
            data_path = Path(tempfile.mkdtemp())
        else:
            result_path = self.base_path / "data_3d_semantics" / seq / "indexed"
            data_path = self.base_path
            if result_path.exists():
                return
            result_path.mkdir()

        _logger.info("Converting 3d semantic labels for sequence %s...", seq)
        tstart = time.time()

        # load poses for aligning point cloud
        self._preload_poses(seq)

        pool = NumberPool(nworkers)
        for fspan in (data_path / "data_3d_semantics" / seq / "static").iterdir():
            if fspan.suffix != ".ply":
                continue
            pool.apply_async(self._parse_semantic_ply, (seq, fspan, False, result_path, expand_frames))
        for fspan in (data_path / "data_3d_semantics" / seq / "dynamic").iterdir():
            if fspan.suffix != ".ply":
                continue
            pool.apply_async(self._parse_semantic_ply, (seq, fspan, True, result_path, expand_frames))
        pool.close()
        pool.join()
        tend = time.time()
        _logger.info("Conversion finished, consumed time: %.4fs", tend - tstart)

        # statistics on distance error
        total_points = 0
        unmatched_points = 0
        for f in result_path.iterdir():
            if f.suffix == ".npy":
                d = np.load(f)
                total_points += len(d)
                unmatched_points += np.sum(d > 5)
        _logger.debug("Unmatched ratio (distance > 5): %.2f%", unmatched_points / total_points * 100)

        # clean up
        if self.inzip:
            _logger.info("Saving indexed semantic labels...")
            with ZipFile(self.base_path / "data_3d_semantics_indexed.zip", "w") as archive:
                for f in result_path.iterdir():
                    if f.suffix != ".npz":
                        continue
                    archive.write(f, f"data_3d_semantics/{seq}/indexed/{f.name}")

            # remove temporary files
            shutil.rmtree(result_path)
            shutil.rmtree(data_path)
            
        else:
            # remove distance files
            for f in list(result_path.iterdir()):
                if f.suffix == ".npy" or f.suffix == ".lock":
                    f.unlink()
        _logger.debug("Conversion clean up finished!")


    @expand_idx
    def annotation_3dpoints(self, idx):
        seq_id, frame_idx = idx
        self._preload_3dsemantics(seq_id)
        
        fname = Path("data_3d_semantics", seq_id, "indexed", "%010d.npz" % frame_idx)
        if self._return_file_path:
            return self.base_path / fname

        if self.inzip:
            fname = str(fname)
            with PatchedZipFile("data_3d_semantics_indexed.zip", to_extract=fname) as ar:
                return dict(np.load(BytesIO(ar.read(fname))))
        else:
            return dict(np.load(self.base_path / fname))

    def annotation_2dpoints(self, idx):
        raise NotImplementedError()

    def _preload_timestamps(self, seq, name):
        if (seq, name) in self._timestamp_cache:
            return

        folder, subfolder, _, archive = self.FRAME_PATH_MAP[name]
        fname = Path(seq, subfolder, "timestamps.txt")
        if self.inzip:
            with PatchedZipFile(self.base_path / archive, to_extract=fname) as data:
                ts = load_timestamps(data, fname, formatted=True)
        else:
            ts = load_timestamps(self.base_path / folder, fname, formatted=True)

        self._timestamp_cache[(seq, name)] = ts.astype(int) // 1000

    @expand_idx
    def timestamp(self, idx, names="velo"):
        if names == "sick":
            raise NotImplementedError("Indexing for sick points are unavailable yet!")

        seq_id, frame_idx = idx
        self._preload_timestamps(seq_id, names)

        return self._timestamp_cache[(seq_id, names)][frame_idx]

    def _preload_poses(self, seq):
        if seq in self._poses_idx:
            return

        fname = Path("data_poses", seq, "poses.txt")
        if self.inzip:
            with PatchedZipFile(self.base_path / "data_poses.zip", to_extract=fname) as data:
                plist = np.loadtxt(data.open(str(fname)))
        else:
            plist = np.loadtxt(self.base_path / fname)

        # do interpolation
        pose_indices = plist[:, 0].astype(int)
        pose_matrices = plist[:, 1:].reshape(-1, 3, 4)
        positions = pose_matrices[:, :, 3]
        rotations = Rotation.from_matrix(pose_matrices[:, :, :3])
        
        ts_frame = "velo" # the frame used for timestamp extraction
        self._preload_timestamps(seq, ts_frame)
        timestamps = self._timestamp_cache[(seq, ts_frame)]

        fpos = interp1d(timestamps[pose_indices], positions, axis=0, fill_value="extrapolate")
        positions = fpos(timestamps)
        frot = interp1d(timestamps[pose_indices], rotations.as_rotvec(), axis=0, fill_value="extrapolate")
        rotations = frot(timestamps)
        
        self._poses_idx[seq] = set(pose_indices)
        self._poses_t[seq] = positions
        self._poses_r[seq] = Rotation.from_rotvec(rotations)

    @expand_idx
    def pose(self, idx, interpolate=True):
        """
        :param interpolate: Not all frames contain pose data in KITTI-360. The loader
            returns interpolated pose if this param is set as True, otherwise returns None
        """
        seq_id, frame_idx = idx

        self._preload_poses(seq_id)
        if frame_idx not in self._poses_idx[seq_id] and not interpolate:
            return None

        return EgoPose(self._poses_t[seq_id][frame_idx], self._poses_r[seq_id][frame_idx])

if __name__ == "__main__":
    # loader = KITTI360Loader("/media/jacob/Storage/Datasets/kitti360-raw/", inzip=True)
    loader = KITTI360Loader("/media/jacob/Storage/Datasets/kitti360", inzip=False, nframes=0)
    # print(loader.camera_data(0))
    # print(loader.lidar_data(0))
    # for i in range(50):
    #     print(loader.pose(i))
    #     print(loader.annotation_3dobject(i))
    # print(loader.timestamp(1))
    # for obj in loader.annotation_3dobject(174):
    #     print(obj.dimension, obj.tag)
    # print(loader.annotation_3dobject(174))
    import sys
    import warnings

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(name)s][%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    _logger.addHandler(handler)
    _logger.setLevel(logging.INFO)
    warnings.simplefilter("ignore")

    sem = loader.annotation_3dpoints(0)
    print(np.sum(sem['semantic'] - sem['instance'] // 1000 > 0)/len(sem['semantic']))
