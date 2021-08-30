import json
import logging
import shutil
import tempfile
import warnings
import zipfile
from io import RawIOBase
from pathlib import Path
from typing import Dict, Union

import msgpack
import numpy as np
import tqdm
from addict import Dict as edict
from d3d.abstraction import (EgoPose, ObjectTag, ObjectTarget3D, Target3DArray,
                             TrackingTarget3D, TransformSet)
from d3d.dataset.base import (TrackingDatasetBase, expand_idx, expand_idx_name,
                              split_trainval_seq)
from d3d.dataset.zip import PatchedZipFile
from PIL import Image
from scipy.spatial.transform import Rotation
from sortedcontainers import SortedDict

from d3d.dataset.nuscenes.constants import (NuscenesDetectionClass,
    NuscenesObjectClass, NuscenesSegmentationClass, train_split, val_split)

_logger = logging.getLogger("d3d")

_default_ranges = { # settings from detection_cvpr_2019
    NuscenesDetectionClass.car: 50,
    NuscenesDetectionClass.truck: 50,
    NuscenesDetectionClass.bus: 50,
    NuscenesDetectionClass.trailer: 50,
    NuscenesDetectionClass.construction_vehicle: 50,
    NuscenesDetectionClass.pedestrian: 40,
    NuscenesDetectionClass.motorcycle: 40,
    NuscenesDetectionClass.bicycle: 40,
    NuscenesDetectionClass.traffic_cone: 30,
    NuscenesDetectionClass.barrier: 30
}

class NuscenesLoader(TrackingDatasetBase):
    '''
    Load Nuscenes dataset into a usable format.
    Please use the d3d_nuscenes_convert command to convert the dataset first into following formats

    * Directory Structure::

        - <base_path directory>
            - trainval
                - scene_xxx(.zip)
                - ...
            - test
                - scene_xxx(.zip)
                - ...

    For description of constructor parameters, please refer to :class:`d3d.dataset.base.TrackingDatasetBase`
    '''
    VALID_CAM_NAMES = ["cam_front", "cam_front_left", "cam_front_right", "cam_back", "cam_back_left", "cam_back_right"]
    VALID_LIDAR_NAMES = ["lidar_top"]
    VALID_OBJ_CLASSES = NuscenesDetectionClass
    VALID_PTS_CLASSES = NuscenesSegmentationClass

    def __init__(self, base_path, inzip=False, phase="training",
                 trainval_split="official", trainval_random=False, trainval_byseq=False, nframes=0):
        super().__init__(base_path, inzip=inzip, phase=phase, nframes=nframes,
                         trainval_split=trainval_split, trainval_random=trainval_random,
                         trainval_byseq=trainval_byseq)

        self.base_path = Path(base_path) / ("trainval" if phase in ["training", "validation"] else "test")
        self.inzip = inzip

        self._metadata = None
        self._rawmapping = None
        self._segmapping = None
        self._load_metadata()

        # split trainval
        if trainval_split == "official":
            if phase == "training":
                trainval_split = train_split
                trainval_byseq = True
            elif phase == "validation":
                trainval_split = val_split
                trainval_byseq = True
            else:
                trainval_split = 1 # all samples

        frames_counts = SortedDict((k, v.nbr_samples) for k, v in self._metadata.items())
        self.frames = split_trainval_seq(phase, frames_counts, trainval_split, trainval_random, trainval_byseq)

    def _load_metadata(self):
        meta_path = self.base_path / "metadata.msg"
        if not meta_path.exists():
            _logger.info("Creating metadata of Nuscenes dataset (%s)...", self.phase)
            metadata = {}

            if self.inzip:
                for archive in self.base_path.iterdir():
                    if archive.is_dir() or archive.suffix != ".zip":
                        continue

                    with PatchedZipFile(archive, to_extract="scene/stats.json") as ar:
                        metadata[archive.stem] = json.loads(ar.read("scene/stats.json"))
            else:
                for folder in self.base_path.iterdir():
                    if not folder.is_dir() or folder.name == "maps":
                        continue

                    metadata[folder.name] = json.loads((folder / "scene/stats.json").read_text())

            assert len(metadata) > 0, "The dataset folder contains no valid frame, "\
                "please check path or parameters!"
            with open(meta_path, "wb") as fout:
                msgpack.pack(metadata, fout)

        with open(meta_path, "rb") as fin:
            self._metadata = SortedDict()
            meta_json = msgpack.unpack(fin)
            for k, v in meta_json.items():
                self._metadata[k] = edict(v)

        # load category mapping for segmentation
        with open(self.base_path / "category.json") as fin:
            cat_json = json.load(fin)
        cat_dict = {}
        for item in cat_json:
            if 'index' in item:
                cat_dict[item['index']] = NuscenesObjectClass.parse(item['name'])

        builtin_table = NuscenesObjectClass._get_nuscenes_id_table()
        self._rawmapping = np.empty(len(builtin_table) + 1, dtype='u4')
        self._segmapping = np.empty(len(builtin_table) + 1, dtype='u1')
        for idx, clsobj in enumerate(builtin_table):
            if idx in cat_dict: # test against offcial definition
                assert cat_dict[idx] == clsobj, "Builtin Nuscenes-lidarseg table is incorrect! Please report this bug."
            self._rawmapping[idx] = clsobj.value
            self._segmapping[idx] = clsobj.to_segmentation().value

    def __len__(self):
        return len(self.frames)

    @property
    def sequence_ids(self):
        return list(self._metadata.keys())

    @property
    def sequence_sizes(self):
        return {k: v.nbr_samples for k, v in self._metadata.items()}

    def _locate_frame(self, idx):
        # use underlying frame index
        idx = self.frames[idx]

        # find corresponding sample
        for k, v in self._metadata.items():
            if idx < (v.nbr_samples - self.nframes):
                return k, idx
            idx -= (v.nbr_samples - self.nframes)
        raise ValueError("Index larger than dataset size")

    def map_data(self, idx):
        # TODO(v0.5): see https://jdhao.github.io/2019/02/23/crop_rotated_rectangle_opencv/ for image cropping
        raise NotImplementedError()

    @expand_idx_name(VALID_LIDAR_NAMES)
    def lidar_data(self, idx, names='lidar_top', formatted=False):
        seq_id, frame_idx = idx
        assert names == 'lidar_top', "Only lidar_top is valid in Nuscenes dataset"
        fname = "lidar_top/%03d.pcd" % frame_idx

        if self._return_file_path:
            return self.base_path / seq_id / fname

        if self.inzip:
            with PatchedZipFile(self.base_path / f"{seq_id}.zip", to_extract=fname) as ar:
                buffer = ar.read(fname)
        else:
            buffer = (self.base_path / seq_id / fname).read_bytes()

        scan = np.frombuffer(buffer, dtype=np.float32)
        scan = np.copy(scan.reshape(-1, 5))  # (x, y, z, intensity, ring index)

        if not formatted:
            return scan
        columns = ['x', 'y', 'z', 'intensity', 'ring_index']
        return scan.view([(c, 'f4') for c in columns])

    @expand_idx_name(VALID_CAM_NAMES)
    def camera_data(self, idx, names=None):
        seq_id, frame_idx = idx
        fname = "%s/%03d.jpg" % (names, frame_idx)

        if self._return_file_path:
            return self.base_path / seq_id / fname

        if self.inzip:
            with PatchedZipFile(self.base_path / f"{seq_id}.zip", to_extract=fname) as ar:
                return Image.open(ar.open(fname)).convert('RGB')
        else:
            return Image.open(self.base_path / seq_id / fname)

    @expand_idx_name(VALID_CAM_NAMES + VALID_LIDAR_NAMES)
    def intermediate_data(self, idx, names=None, ninter_frames=None):
        seq_id, frame_idx = idx
        fname = "intermediate/%03d/meta.json" % frame_idx

        # Load meta json
        if self.inzip:
            with PatchedZipFile(self.base_path / f"{seq_id}.zip", to_extract=fname) as ar:
                meta = json.loads(ar.read(fname))
        else:
            with Path(self.base_path, seq_id, fname).open() as fin:
                meta = json.load(fin)
        if not meta:
            return []

        # Select frames
        if ninter_frames is None:
            meta = [edict(item) for item in meta[names]]
        else:
            meta = [edict(item) for item in meta[names][:ninter_frames]]

        # parse pose part
        for item in meta:
            rotation = item.pop("rotation")
            rotation = Rotation.from_quat(rotation[1:] + [rotation[0]])
            translation = item.pop("translation")
            translation = rotation.inv().as_matrix().dot(translation)
            item.pose = EgoPose(translation, rotation)

        if self._return_file_path:
            for item in meta:
                item.file = self.base_path / seq_id / "intermediate" / f"{frame_idx:03}" / item.file
            return meta

        # Load actual data
        for item in meta:
            fname = "intermediate/%03d/%s" % (frame_idx, item.pop("file"))
            if names in self.VALID_CAM_NAMES:
                item.data = self._load_camera_data(seq_id, fname)
            else: # names in VALID_LIDAR_NAMES:
                item.data = self._load_lidar_data(seq_id, fname)
        return meta

    @expand_idx
    def annotation_3dobject(self, idx, raw=False, convert_tag=True, with_velocity=True):
        seq_id, frame_idx = idx
        fname = "annotation/%03d.json" % frame_idx
        if self._return_file_path:
            return self.base_path / seq_id / fname

        if self.inzip:
            with PatchedZipFile(self.base_path / f"{seq_id}.zip", to_extract=fname) as ar:
                labels = json.loads(ar.read(fname))
        else:
            with Path(self.base_path, seq_id, fname).open() as fin:
                labels = json.load(fin)
        labels = list(map(edict, labels))
        if raw:
            return labels

        # parse annotations
        ego_pose = self.pose(idx, bypass=True)
        ego_r, ego_t = ego_pose.orientation, ego_pose.position
        ego_ri = ego_r.inv()
        ego_rim = ego_ri.as_matrix()
        outputs = Target3DArray(frame="ego")
        for label in labels:
            # convert tags
            tag = NuscenesObjectClass.parse(label.category)
            for attr in label.attribute:
                tag = tag | NuscenesObjectClass.parse(attr)
            if convert_tag:
                tag = ObjectTag(tag.to_detection(), NuscenesDetectionClass)
            else:
                tag = ObjectTag(tag, NuscenesObjectClass)
            aux = dict(num_lidar_pts=label['num_lidar_pts'], num_radar_pts=label['num_radar_pts'])

            # caculate relative pose
            r = Rotation.from_quat(label.rotation[1:] + [label.rotation[0]])
            t = label.translation
            rel_r = ego_ri * r
            rel_t = np.dot(ego_rim, t - ego_t)
            size = [label.size[1], label.size[0], label.size[2]] # wlh -> lwh
            tid = int(label.instance[:8], 16) # truncate into uint64

            # create object
            if with_velocity:
                v = np.dot(ego_rim, label.velocity)
                w = label.angular_velocity
                target = TrackingTarget3D(rel_t, rel_r, size, v, w, tag, tid=tid, aux=aux)
                outputs.append(target)
            else:
                target = ObjectTarget3D(rel_t, rel_r, size, tag, tid=tid, aux=aux)
                outputs.append(target)

        return outputs

    @expand_idx_name(VALID_LIDAR_NAMES)
    def annotation_3dpoints(self, idx, names='lidar_top', parse_tag=True, convert_tag=True):
        '''
        :param parse_tag: Parse tag from original nuscenes id defined in category.json into :class:`NuscenesObjectClass`
        :param convert_tag: Convert tag from :class:`NuscenesObjectClass` to :class:`NuscenesSegmentationClass`
        '''
        assert names == 'lidar_top'
        seq_id, frame_idx = idx

        fname = "lidar_top_seg/%03d.bin" % frame_idx
        if self._return_file_path:
            return edict(semantic=self.base_path / seq_id / fname)

        if self.inzip:
            with PatchedZipFile(self.base_path / f"{seq_id}.zip", to_extract=fname) as ar:
                buffer = ar.read(fname)
        else:
            buffer = (self.base_path / seq_id / fname).read_bytes()
        label = np.frombuffer(buffer, dtype='u1')

        if parse_tag:
            if convert_tag:
                return edict(semantic=self._segmapping[label])
            else:
                return edict(semantic=self._rawmapping[label])
        else:
            return edict(semantic=label)

    @expand_idx
    def metadata(self, idx):
        seq_id, frame_idx = idx
        assert not self._return_file_path, "The metadata is not in a single file!"

        meta = self._metadata[seq_id]
        return edict(
            scene_description=meta.description,
            scene_token=meta.token,
            sample_token=meta.sample_tokens[frame_idx],
            logfile=meta.logfile,
            date_captured=meta.date_captured,
            vehicle=meta.vehicle,
            location=meta.location
        )

    @expand_idx_name(VALID_CAM_NAMES + VALID_LIDAR_NAMES)
    def token(self, idx, names='lidar_top'):
        '''
        Return the sample data token in original Nuscenes data given data index and sensor name
        '''
        seq_id, frame_idx = idx
        fname = "scene/tokens.json"
        assert not self._return_file_path, "The tokens are not stored in a single file!"

        if self.inzip:
            with PatchedZipFile(self.base_path / (seq_id + ".zip"), to_extract=fname) as ar:
                token_data = json.loads(ar.read(fname))
        else:
            with Path(self.base_path, seq_id, fname).open() as fin:
                token_data = json.load(fin)
        return token_data[names][frame_idx]

    @expand_idx
    def calibration_data(self, idx):
        seq_id, _ = idx
        assert not self._return_file_path, "The calibration is not in a single file!"

        calib_params = TransformSet("ego")
        calib_fname = "scene/calib.json"
        if self.inzip:
            with PatchedZipFile(self.base_path / (seq_id + ".zip"), to_extract=calib_fname) as ar:
                calib_data = json.loads(ar.read(calib_fname))
        else:
            with Path(self.base_path, seq_id, calib_fname).open() as fin:
                calib_data = json.load(fin)

        for frame, calib in calib_data.items():
            # set intrinsics
            if frame.startswith('cam'):
                image_size = (1600, 900) # currently all the images have the same size
                projection = np.array(calib['camera_intrinsic'])
                calib_params.set_intrinsic_camera(frame, projection, image_size, rotate=False)
            elif frame.startswith('lidar'):
                calib_params.set_intrinsic_lidar(frame)
            elif frame.startswith('radar'):
                calib_params.set_intrinsic_radar(frame)
            else:
                raise ValueError("Unrecognized frame name.")

            # set extrinsics
            r = Rotation.from_quat(calib['rotation'][1:] + [calib['rotation'][0]])
            t = np.array(calib['translation'])
            extri = np.eye(4)
            extri[:3, :3] = r.as_matrix()
            extri[:3, 3] = t
            calib_params.set_extrinsic(extri, frame_from=frame)

        return calib_params

    @expand_idx
    def identity(self, idx):
        return idx

    @expand_idx
    def timestamp(self, idx, names="lidar_top"):
        seq_id, frame_idx = idx
        fname = "timestamp/%03d.json" % frame_idx
        if self.inzip:
            with PatchedZipFile(self.base_path / f"{seq_id}.zip", to_extract=fname) as ar:
                tsdict = json.loads(ar.read(fname))
        else:
            with Path(self.base_path, seq_id, fname).open() as fin:
                tsdict = json.load(fin)
        if names in tsdict:
            return tsdict[names]
        else:
            return tsdict["lidar_top"]

    @expand_idx_name(VALID_LIDAR_NAMES + VALID_CAM_NAMES)
    def pose(self, idx, names="lidar_top", raw=False):
        # Note that here pose always return the pose of the vehicle, names are for different timestamps
        seq_id, frame_idx = idx
        fname = "pose/%03d.json" % frame_idx
        if self.inzip:
            with PatchedZipFile(self.base_path / f"{seq_id}.zip", to_extract=fname) as ar:
                data = json.loads(ar.read(fname))
        else:
            with Path(self.base_path, seq_id, fname).open() as fin:
                data = json.load(fin)

        data = data[names]
        if raw:
            return data

        r = Rotation.from_quat(data['rotation'][1:] + [data['rotation'][0]])
        t = np.array(data['translation'])
        return EgoPose(t, r)

    @property
    def pose_name(self):
        return 'ego'

    @expand_idx
    def dump_detection_output(self, idx: Union[int, tuple], detections: Target3DArray,
                              fout: Union[str, Path, RawIOBase],
                              ranges: Dict[NuscenesObjectClass, float] = _default_ranges) -> None:
        calib = self.calibration_data(idx)
        ego_pose = self.pose(idx)
        sample_token = self.metadata(idx).sample_token

        default_attr = {
            NuscenesDetectionClass.car: NuscenesObjectClass.vehicle_parked.attribute_name,
            NuscenesDetectionClass.pedestrian: NuscenesObjectClass.pedestrian_standing.attribute_name,
            NuscenesDetectionClass.trailer: NuscenesObjectClass.vehicle_parked.attribute_name,
            NuscenesDetectionClass.truck: NuscenesObjectClass.vehicle_parked.attribute_name,
            NuscenesDetectionClass.bus: NuscenesObjectClass.vehicle_stopped.attribute_name,
            NuscenesDetectionClass.motorcycle: NuscenesObjectClass.cycle_without_rider.attribute_name,
            NuscenesDetectionClass.construction_vehicle: NuscenesObjectClass.vehicle_parked.attribute_name,
            NuscenesDetectionClass.bicycle: NuscenesObjectClass.cycle_without_rider.attribute_name,
            NuscenesDetectionClass.barrier: "",
            NuscenesDetectionClass.traffic_cone: "",
        }
        output = []

        for box in calib.transform_objects(detections, "ego"):
            if isinstance(box.tag_top, NuscenesObjectClass):
                box_cat = box.tag_top.to_detection()
                box_attr = box.tag_top.attribute
            elif isinstance(box.tag_top, NuscenesDetectionClass):
                box_cat = box.tag_top
                box_attr = NuscenesObjectClass.unknown
            else:
                raise ValueError("Incorrect object tag type")

            if box_cat in ranges:
                if np.hypot(box.position[0], box.position[1]) > ranges[box_cat]:
                    continue

            # parse category and attribute
            if box_attr == NuscenesObjectClass.unknown:
                if isinstance(box, TrackingTarget3D) and np.hypot(box.velocity[0], box.velocity[1]) > 0.2:
                    if box_cat in [
                        NuscenesDetectionClass.car,
                        NuscenesDetectionClass.construction_vehicle,
                        NuscenesDetectionClass.bus,
                        NuscenesDetectionClass.truck,
                        NuscenesDetectionClass.trailer,
                    ]:
                        attr = NuscenesObjectClass.vehicle_moving.attribute_name
                    elif box_cat in [
                        NuscenesDetectionClass.bicycle,
                        NuscenesDetectionClass.motorcycle
                    ]:
                        attr = NuscenesObjectClass.cycle_with_rider.attribute_name
                    elif box_cat in [
                        NuscenesDetectionClass.pedestrian
                    ]:
                        attr = NuscenesObjectClass.pedestrian_moving.attribute_name
                    else:
                        attr = default_attr[box_cat]
                else:
                    attr = default_attr[box_cat]
            else:
                attr = box.tag_top.attribute_name

            # calculate properties, this part is the exact inverse of annotation_3dobject
            rel_r, rel_t = box.orientation, box.position
            ego_r, ego_t = ego_pose.orientation, ego_pose.position
            ego_rm = ego_r.as_matrix() 
            ego_t = ego_rm.dot(ego_t)
            t = ego_rm.dot(rel_t) + ego_t
            r = (ego_r * rel_r).as_quat().tolist()
            l, w, h = box.dimension.tolist()

            odict = dict(
                sample_token=sample_token,
                translation=t.tolist(),
                size=[w, l, h], # lwh -> wlh
                rotation=[r[3]] + r[:3], # xyzw -> wxyz
                detection_name=box_cat.name,
                detection_score=box.tag_top_score,
                attribute_name=attr
            )
            if isinstance(box, TrackingTarget3D):
                vel = ego_rm.dot(box.velocity)
                odict['velocity'] = vel[:2].tolist()
            else:
                odict['velocity'] = [0, 0]
            output.append(odict)

        if not output: # add token if no objects available
            output.append(sample_token)

        if isinstance(fout, (str, Path)):
            Path(fout).write_text(json.dumps(output))
        else:
            fout.write(json.dumps(output).encode())

    @expand_idx
    def dump_segmentation_output(self, idx: Union[int, tuple], segmentation: np.ndarray,
                                 folder_out: str, raw2seg: bool = True,
                                 default_class: Union[int, NuscenesSegmentationClass] = 15):
        '''
        :param raw2seg: If set as true, input array will be considered as raw id (consistent with values stored in label)
        :param default_class: Class to be selected when the label is 0 (ignore)
        '''
        folder_out = Path(folder_out)
        folder_out.mkdir(exist_ok=True, parents=True)

        if isinstance(default_class, NuscenesSegmentationClass):
            default_class = default_class.value
        if default_class == 0:
            warnings.warn("Class 0 (ignored) is not removed!")

        fname = "%s_lidarseg.bin" % self.token(idx, 'lidar_top')
        arr = self._segmapping[segmentation] if raw2seg else segmentation.astype('u1')
        arr = np.where(arr == 0, default_class, arr) # 0 is not acceptable in submission
        arr.tofile(folder_out / fname)

def create_submission(result_path, output_file, task="detection", modality=None, eval_set="test"):
    '''
    For lidarseg submission, you need to have file name as f"{lidar_sample_data_token}_lidarseg.bin"

    :param result_path: path to the dump results
    :param output_file: path to the output file
    :param modality: modality used
    :param eval_set: Evaluation split, required when create submission for lidarseg
    '''
    if not modality:
        modality = {
            "use_camera":   False,  # Whether this submission uses camera data as an input.
            "use_lidar":    True,   # Whether this submission uses lidar data as an input.
            "use_radar":    False,  # Whether this submission uses radar data as an input.
            "use_map":      False,  # Whether this submission uses map data as an input.
            "use_external": False,  # Whether this submission uses external data as an input.
        }

    if task == "detection":
        nusc_submissions = {
            'meta': modality,
            'results': {},
        }

        fjsons = list(Path(result_path).iterdir())
        for fdump in tqdm.tqdm(fjsons, "Reading dumped objects"):
            with open(fdump) as fin:
                dump_data = json.load(fin)
            if isinstance(dump_data[0], str):
                nusc_submissions['results'][dump_data[0]] = {}
            else:
                token = dump_data[0]['sample_token']
                nusc_submissions['results'][token] = dump_data
        
        fsubmission = Path(output_file)
        if fsubmission.suffix != ".json":
            fsubmission = fsubmission.parent / (fsubmission.name + ".json")
        fsubmission.parent.mkdir(exist_ok=True, parents=True)
        fsubmission.write_bytes(json.dumps(nusc_submissions).encode())
    elif task == "lidarseg":
        fsubmission = Path(output_file)
        fsubmission.parent.mkdir(exist_ok=True, parents=True)
        with zipfile.ZipFile(fsubmission, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(eval_set + '/submission.json', json.dumps(dict(meta=modality)))

            fjsons = list(Path(result_path).iterdir())
            for fdump in tqdm.tqdm(fjsons, "Reading dumped arrays"):
                archive.write(fdump, 'lidarseg/' + eval_set + '/' + fdump.name)
    else:
        raise ValueError("Unrecognized task")

def execute_official_evaluator(nusc_path, result_path, output_path,
                               task="detection",
                               nusc_version="v1.0-trainval",
                               eval_version="detection_cvpr_2019",
                               verbose=True):
    '''
    Call official evaluator implemented in Nuscenes devkit on evaluation set

    :param result_path: For detection, it's the path to the submission json file; For lidarseg,
        it's the path to the submission zip
    '''
    from nuscenes import NuScenes
    nusc = NuScenes(version=nusc_version, dataroot=nusc_path, verbose=verbose)

    if task == "detection":
        from nuscenes.eval.detection.config import config_factory
        from nuscenes.eval.detection.evaluate import NuScenesEval

        nusc_eval = NuScenesEval(
            nusc,
            config=config_factory(eval_version),
            result_path=result_path,
            eval_set='val',
            output_dir=output_path,
            verbose=verbose
        )
        nusc_eval.main(render_curves=False)
    elif task == "lidarseg":
        from nuscenes.eval.lidarseg.evaluate import LidarSegEval

        tempfolder = tempfile.mkdtemp()
        if verbose:
            print("Extracting submission to", tempfolder)
        with zipfile.ZipFile(result_path, "r") as archive:
            archive.extractall(tempfolder)

        try:
            nusc_eval = LidarSegEval(
                nusc,
                results_folder=tempfolder,
                eval_set='val',
                verbose=verbose
            )
            results = nusc_eval.evaluate()
            if verbose:
                print("Results:", results)

            output_path = Path(output_path)
            output_path.mkdir(exist_ok=True, parents=True)
            with open(output_path / "lidarseg_results.json", "w") as fout:
                json.dump(results, fout, indent="  ")
        finally:
            shutil.rmtree(tempfolder)
    else:
        raise ValueError("Unsupported evaluation task!")
