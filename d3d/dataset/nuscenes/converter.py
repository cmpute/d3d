'''
This script convert Nuscenes dataset tarballs to zipfiles or regular directory split by collection segments.
'''

import json
import os
import shutil
import tarfile
import tempfile
import time
import zipfile
from collections import defaultdict
from multiprocessing import Pool, Value
from pathlib import Path, PurePath

import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm


def _load_dict(item):
    token = int(item.pop('token'), 16)
    value = {
        k: (((v and int(v, 16) or None) if isinstance(v, str) # convert token to integer
            else [int(lv, 16) for lv in v]) # convert list of tokens
            if ('token' in k or k in ['prev', 'next']) # if the field is for token
            else v) # otherwise copy
        for k, v in item.items()
    }
    return token, value

def _load_table(path):
    with open(path) as fin:
        data = json.load(fin)
        kvdata = (_load_dict(item) for item in data)
        kvdata = {k: v for (k, v) in kvdata}
        return kvdata

def hex_pad32(value): # convert hex string and pad to 32 length
    return hex(value)[2:].zfill(32)

class KeyFrameConverter:
    def __init__(self, phase, input_meta_path, input_blob_paths, output_path, input_seg_path=None,
        zip_output=False, compression=zipfile.ZIP_STORED, **conversion_args):
        '''
        :param phase: mini / trainval / test
        :param local_map_range: Range of generated local map in meters, range <= 0 means not to output map
        '''
        assert isinstance(input_blob_paths, list), "blobs path should be a list"
        self.meta_path = Path(input_meta_path)
        self.blob_paths = [Path(p) for p in input_blob_paths]
        self.seg_path = Path(input_seg_path) if input_seg_path is not None else None
        self.phase = phase
        self.output_path = Path(output_path)
        self.zip_output = zip_output
        self.zip_compression = compression
        self.store_inter = conversion_args.get('store_inter', 0)
        self.estimate_velocity = conversion_args.get('estimate_velocity', False)

        # nuscenes tables
        self.sample_table = None # key frames
        self.sample_data_table = None # data frames
        self.scene_table = None # sequence
        self.log_table = None
        self.map_table = None
        self.sensor_table = None
        self.calibrated_sensor_table = None
        self.ego_pose_table = None
        self.instance_table = None
        self.annotation_table = None
        self.attribute_table = None
        self.category_table = None
        self.lidarseg_table = None
        
        # temporary mappings and objects
        self.temp_dir = None
        self.table_path = None
        self.sample_order = None # sample -> (scene token, order)
        self.filename_table = None # filename -> (sample_data token, scene token, sensor name, order, file extension/full name)
        self.scene_sensor_table = None # scene -> calibrated_sensor
        self.scene_map_table = None # scene -> {map category: map file}
        self.ohandles = {} # scene -> scene directory or zipfile handle
        self.oscenes = set() # mark output scenes
        self.oframes = defaultdict(dict) # scene -> order -> (timestamp_dict, pose_dict)
        self.oframes_inter = defaultdict(list) # (scene, order) -> [(sensor, rotation, translation, timestamp)]

    def _save_file(self, stoken, fname, data):
        if self.zip_output:
            self.ohandles[stoken].writestr(fname, data)
        else:
            ofile = self.ohandles[stoken] / fname
            ofile.parent.mkdir(exist_ok=True, parents=True)
            ofile.write_bytes(data)

    def _parse_scenes(self):
        self.log_table = _load_table(self.table_path / "log.json")
        self.map_table = _load_table(self.table_path / "map.json")

        # create log to map table
        log_map_table = defaultdict(dict)
        for mtoken, mdata in self.map_table.items():
            for ltoken in mdata["log_tokens"]:
                log_map_table[ltoken][mdata['category']] = mdata['filename']

        # create output handles, sort samples and save timestamps and metadata
        self.scene_map_table = {}
        self.sample_order = {}
        for stoken, data in self.scene_table.items():
            # add scene table mapping
            log = self.log_table[data['log_token']]
            self.scene_map_table[stoken] = log_map_table[data['log_token']]

            # sort samples
            count = 0
            cur = data['first_sample_token']
            token_list = []

            while True:
                # add mapping and save timestamp
                self.sample_order[cur] = (stoken, count)

                # move to the next sample
                token_list.append(hex_pad32(cur))
                if self.sample_table[cur]['next'] is None:
                    break
                cur = self.sample_table[cur]['next']
                count += 1
                assert count < 1000, "Frame index is larger than file name capacity!"

            # save metadata
            meta = dict(
                nbr_samples=data['nbr_samples'],
                description=data['description'],
                token=hex_pad32(stoken),
                map=self.scene_map_table[stoken],
                sample_tokens=token_list
            )
            meta.update(log)
            meta_json = json.dumps(meta).encode()

            if self.zip_output:
                self.ohandles[stoken] = zipfile.ZipFile(self.output_path / ("%s.zip" % data['name']),
                    "w", compression=self.zip_compression)
                self.ohandles[stoken].writestr("scene/stats.json", meta_json)
            else:
                self.ohandles[stoken] = self.output_path / data['name']
                (self.ohandles[stoken] / "scene").mkdir(exist_ok=True, parents=True)
                (self.ohandles[stoken] / "scene/stats.json").write_bytes(meta_json)

        # clean used tables
        self.log_table = None
        self.map_table = None

    def _parse_sample_data(self):
        # parse and reverse sample_data table
        self.filename_table = dict()
        self.scene_sensor_table = defaultdict(set)
        for token, data in self.sample_data_table.items():

            fname = data['filename']
            fname = fname[fname.rfind('/')+1:]
            scene, order = self.sample_order[data['sample_token']]
            ctoken = data['calibrated_sensor_token']
            sensor = self.calibrated_sensor_table[ctoken]['sensor_token']
            sensor = self.sensor_table[sensor]['channel'].lower()
            
            if data['is_key_frame']: # process key frames
                # store output format
                self.filename_table[fname] = (token, scene, sensor, order, data['fileformat'])

                # store calibration object
                if ctoken not in self.scene_sensor_table[scene]:
                    self.scene_sensor_table[scene].add(ctoken)

                # store lidar segmentation
                if self.lidarseg_table and token in self.lidarseg_table:
                    fname = self.lidarseg_table[token]['filename']
                    fname = fname[fname.rfind('/')+1:]
                    self.filename_table[fname] = (token, scene, sensor + '_seg', order, 'bin')

            elif self.store_inter: # process intermediate frames
                cur_data = data
                counter = 0

                # check if frame is with self.store_inter count range
                while cur_data['next'] and counter < self.store_inter:
                    cur_data = self.sample_data_table[cur_data['next']]
                    counter += 1

                    # found corresponding key frame
                    if cur_data['is_key_frame']:
                        # store output format
                        packed_name = "%03d/%s-%d.%s" % (order, sensor, counter, data['fileformat'])
                        self.filename_table[fname] = (token, scene, "intermediate", order, packed_name)

                        # store metadata
                        pose = self.ego_pose_table[data['ego_pose_token']]
                        self.oframes_inter[(scene, order)].append(dict(
                            file=packed_name[4:],
                            sensor=sensor,
                            rotation=pose['rotation'],
                            translation=pose['translation'],
                            timestamp=data['timestamp'],
                            token=token
                        ))
                        break

    def _save_calibrations(self):
        for stoken, calib_tokens in self.scene_sensor_table.items():
            if stoken not in self.oscenes:
                continue
            
            calib = dict()
            for ctoken in calib_tokens:
                cdata = self.calibrated_sensor_table[ctoken]
                sensor = cdata.pop('sensor_token')
                sensor = self.sensor_table[sensor]['channel'].lower()
                calib[sensor] = cdata
            calib_json = json.dumps(calib).encode()
            self._save_file(stoken, "scene/calib.json", calib_json)

    def _save_annotations(self):
        self.instance_table = _load_table(self.table_path / "instance.json")
        self.attribute_table = _load_table(self.table_path / "attribute.json")
        self.category_table = _load_table(self.table_path / "category.json")
        self.annotation_table = _load_table(self.table_path / "sample_annotation.json")
        anno_list = defaultdict(list)

        def extract_rt(adata): # utility function for velocity estimation
            t = np.array(adata['translation'])
            r = adata['rotation'][1:] + [adata['rotation'][0]]
            r = Rotation.from_quat(r).as_rotvec()
            ts = self.sample_table[adata['sample_token']]['timestamp']
            return t, r, ts

        # parse annotations from table
        time_delta_threshold = 1.5
        static_speed_threshold = 0.01
        for itoken, data in self.instance_table.items():
            cur = data['first_annotation_token']
            instance_id = hex_pad32(itoken)
            instance_category = self.category_table[data['category_token']]['name']

            while True:
                adata = self.annotation_table[cur]
                scene, order = self.sample_order[adata['sample_token']]

                # save this annotation if the sample has corresponding sensor data
                if order in self.oframes[scene]:
                    anno = dict(
                        category=instance_category,
                        instance=instance_id,
                        attribute=[self.attribute_table[t]['name'] for t in adata['attribute_tokens']],
                        size=adata['size'],
                        rotation=adata['rotation'],
                        translation=adata['translation'],
                        num_lidar_pts=adata['num_lidar_pts'],
                        num_radar_pts=adata['num_radar_pts'],
                        visibility=adata['visibility_token'],
                    )

                    # estimate velocity if required
                    if self.estimate_velocity:
                        aprev, anext = adata['prev'], adata['next']
                        if not aprev and not anext:
                            v = w = np.array([np.nan] * 3)
                        else:
                            t1, r1, ts1 = extract_rt(self.annotation_table[aprev] if aprev else adata)
                            t2, r2, ts2 = extract_rt(self.annotation_table[anext] if anext else adata)
                            dt = (ts2 - ts1) * 1e-6
                            if dt > time_delta_threshold: # gating differential time delta
                                v = w = np.array([np.nan] * 3)
                            else:
                                v, w = (t2 - t1) / dt, (r2 - r1) / dt
                                v[np.abs(v) < static_speed_threshold] = 0 # gating values for static objects
                                w[np.abs(w) < static_speed_threshold] = 0
                        anno['velocity'] = v.tolist()
                        anno['angular_velocity'] = w.tolist()

                    anno_list[adata['sample_token']].append(anno)

                # move to the next sample
                if adata['next'] is None:
                    break
                cur = adata['next']

        # save annotations
        for stoken, annos in anno_list.items():
            scene, order = self.sample_order[stoken]
            self._save_file(scene, "annotation/%03d.json" % order, json.dumps(annos).encode())

        # clean used tables
        self.instance_table = None
        self.attribute_table = None
        self.category_table = None
        self.annotation_table = None

    def _save_definitions(self):
        # save definition table
        shutil.copy(self.table_path / "visibility.json", self.output_path)
        shutil.copy(self.table_path / "category.json", self.output_path)
        shutil.copy(self.table_path / "attribute.json", self.output_path)
        if self.seg_path is not None:
            shutil.copy(self.table_path / "lidarseg_category.json", self.output_path / "category.json")

    def _save_tokens(self):
        token_dicts = defaultdict(dict)

        # organize tokens
        for token, scene, sensor, order, ext in self.filename_table.values():
            if sensor == "intermediate" or sensor.endswith("seg"):
                continue

            if sensor not in token_dicts[scene]:
                nsamples = self.scene_table[scene]['nbr_samples']
                token_dicts[scene][sensor] = [None] * nsamples
            token_dicts[scene][sensor][order] = hex_pad32(token)

        # save tokens
        for scene, tokens in token_dicts.items():
            self._save_file(scene, "scene/tokens.json", json.dumps(tokens).encode())

    def load_metadata(self):
        # extract metadata
        self.temp_dir = Path(tempfile.mkdtemp())
        print("Extracting tables to %s..." % self.temp_dir)

        if self.seg_path is not None:
            seg_file = tarfile.open(self.seg_path, "r|*")
            for tinfo in seg_file:
                if tinfo.name.endswith("lidarseg.json") and self.phase in tinfo.name:
                    seg_file.extract(tinfo, self.temp_dir)
                elif tinfo.name.endswith("category.json") and self.phase in tinfo.name:
                    seg_file.extract(tinfo, self.temp_dir)
                    json_path = self.temp_dir / tinfo.path
                    json_path.rename(json_path.with_name("lidarseg_category.json"))
            seg_file.close()

        meta_file = tarfile.open(self.meta_path, "r|*")
        for tinfo in meta_file:
            if tinfo.name.startswith('v'):
                version = PurePath(tinfo.name).parts[0]
                meta_file.extract(tinfo, self.temp_dir)
            elif tinfo.name.startswith('map'):
                # directly extract map to output directory
                meta_file.extract(tinfo, self.output_path)
        meta_file.close()

        # load tables
        print("Constructing tables...")
        assert version.endswith(self.phase), "Phase mismatch in loading nuscenes!"
        self.table_path = self.temp_dir / version
        self.sample_table = _load_table(self.table_path / "sample.json")
        self.sample_data_table = _load_table(self.table_path / "sample_data.json")
        self.scene_table = _load_table(self.table_path / "scene.json")
        self.sensor_table = _load_table(self.table_path / "sensor.json")
        self.calibrated_sensor_table = _load_table(self.table_path / "calibrated_sensor.json")
        self.ego_pose_table = _load_table(self.table_path / "ego_pose.json")
        if self.seg_path is not None:
            self.lidarseg_table = _load_table(self.table_path / "lidarseg.json")

        # parse tables
        self._parse_scenes()
        self._parse_sample_data()

    def load_blobs(self, debug):
        # iterate through all the sensor data
        blobs = self.blob_paths
        if self.seg_path is not None:
            blobs = [self.seg_path] + blobs

        # iterate through all the sensor data
        for iblob, blob_path in tqdm(enumerate(blobs), desc="Loading blobs", unit="tars"):
            if debug and iblob > 0:
                break

            blob_file = tarfile.open(blob_path)
            for counter, tinfo in enumerate(tqdm(
                blob_file, desc="Reading files",unit="files", leave=False)):
                # skip files that are not samples
                if tinfo.isdir():
                    continue
                fname = PurePath(tinfo.name).name
                if fname not in self.filename_table:
                    continue
                token, scene, sensor, order, ext = self.filename_table[fname]

                # dealing with duplicates in mini and trainval blobs
                if sensor.endswith("seg") and self.phase not in tinfo.name:
                    continue

                # save sample data
                data = blob_file.extractfile(tinfo).read()
                if sensor == 'intermediate':
                    self._save_file(scene, "intermediate/%s" % ext, data)
                else:
                    self._save_file(scene, "%s/%03d.%s" % (sensor, order, ext), data)

                # tag the scene to be filled
                if scene not in self.oscenes:
                    self.oscenes.add(scene)
                if order not in self.oframes[scene]:
                    self.oframes[scene][order] = ({}, {})

                # save timestamp and pose
                sample_data = self.sample_data_table[token]
                self.oframes[scene][order][0][sensor] = sample_data["timestamp"]
                self.oframes[scene][order][1][sensor] = sample_data["ego_pose_token"]

                # only convert 2 files in debug
                if debug and counter > 1:
                    break

    def save_metadata(self):
        # save things stored in metadata
        print("Saving metadata...")
        self._save_calibrations()
        self._save_annotations()
        self._save_definitions()
        self._save_tokens()

    def clean_up(self, debug):
        print("Cleaning up...")
        # clean zip files and close handles
        if self.zip_output:
            for scene, handle in self.ohandles.items():
                nsamples = self.scene_table[scene]['nbr_samples']

                # remove archives where the data is incomplete
                if scene not in self.oscenes or len(self.oframes[scene]) < nsamples:
                    handle.close()
                    if not debug:
                        os.remove(handle.filename)
                
                else:
                    nlist = set(handle.namelist())
                    for i in range(nsamples):
                        # fill scene samples without annotation
                        aname = "annotation/%03d.json" % i
                        if aname not in nlist:
                            handle.writestr(aname, b"[]")

                        # save timestamp and pose
                        timestamps, poses = self.oframes[scene][i]
                        poses = {k: dict(rotation=self.ego_pose_table[v]["rotation"],
                                         translation=self.ego_pose_table[v]["translation"])
                                    for k, v in poses.items()}
                        handle.writestr("timestamp/%03d.json" % i, json.dumps(timestamps).encode())
                        handle.writestr("pose/%03d.json" % i, json.dumps(poses).encode())

                        # sort and save intermediate table
                        inter_table = defaultdict(list)
                        for item in self.oframes_inter[(scene, i)]:
                            sensor = item.pop('sensor')
                            inter_table[sensor].append(item)
                        for items in inter_table.values():
                            items.sort(key=lambda item: item['timestamp'], reverse=True) # lastest to old frames
                        if self.store_inter:
                            handle.writestr("intermediate/%03d/meta.json" % i, json.dumps(inter_table).encode())
                    handle.close()

        else:
            for scene, path in self.ohandles.items():
                nsamples = self.scene_table[scene]['nbr_samples']

                # remove archives where the data is incomplete
                if scene not in self.oscenes or len(self.oframes[scene]) < nsamples:
                    if not debug:
                        shutil.rmtree(path)

                else:
                    for i in range(nsamples):
                        # fill scene samples without annotation
                        afile = path / ("annotation/%03d.json" % i)
                        afile.parent.mkdir(exist_ok=True, parents=True)
                        if not afile.exists():
                            afile.write_bytes(b"[]")

                        # save timestamp and pose
                        timestamps, poses = self.oframes[scene][i]
                        poses = {k: dict(rotation=self.ego_pose_table[v]["rotation"],
                                         translation=self.ego_pose_table[v]["translation"])
                                    for k, v in poses.items()}
                        self._save_file(scene, "timestamp/%03d.json" % i, json.dumps(timestamps).encode())
                        self._save_file(scene, "pose/%03d.json" % i, json.dumps(poses).encode())

                        # sort and save intermediate table
                        inter_table = defaultdict(list)
                        for item in self.oframes_inter[(scene, i)]:
                            sensor = item.pop('sensor')
                            inter_table[sensor].append(item)
                        for items in inter_table.values():
                            items.sort(key=lambda item: item['timestamp'], reverse=True) # lastest to old frames
                        if self.store_inter:
                            self._save_file(scene, "intermediate/%03d/meta.json" % i, json.dumps(inter_table).encode())

    def convert(self, debug=False):
        try:
            self.load_metadata()
            self.load_blobs(debug=debug)
            self.save_metadata()
            self.clean_up(debug=debug)
        finally:
            if self.temp_dir is not None:
                shutil.rmtree(self.temp_dir)

def convert_dataset_inpath(input_path, output_path, debug=False,
    mini=False, zip_output=False, **conversion_args):
    input_path, output_path = Path(input_path), Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    # parse compression options
    if zip_output:
        if zip_output == "deflated":
            compression = zipfile.ZIP_DEFLATED
        elif zip_output == "bzip2":
            compression = zipfile.ZIP_BZIP2
        elif zip_output == "lzma":
            compression = zipfile.ZIP_LZMA
        else:
            compression = zipfile.ZIP_STORED
    else:
        compression = zipfile.ZIP_STORED

    if mini: # convert mini dataset
        phase_path = output_path / "trainval"
        if not phase_path.exists():
            phase_path.mkdir()

        mini_archive = next(input_path.glob("*-mini.*"))
        mini_seg = list(input_path.glob("*lidarseg-mini*"))
        mini_seg = mini_seg[0] if len(mini_seg) > 0 else None
        KeyFrameConverter("mini", input_meta_path=mini_archive, input_blob_paths=[mini_archive],
            input_seg_path=mini_seg, output_path=phase_path, zip_output=zip_output,
            compression=compression, **conversion_args).convert(debug)
    else:
        # convert trainval dataset
        print("Processing trainval datasets...")
        phase_path = output_path / "trainval"
        if not phase_path.exists():
            phase_path.mkdir()

        trainval_meta = next(input_path.glob("*-trainval_meta.*"))
        trainval_seg = list(input_path.glob("*lidarseg-all*"))
        trainval_seg = trainval_seg[0] if len(trainval_seg) > 0 else None
        trainval_blobs = list(p for p in input_path.glob("*blobs*") if 'trainval' in p.name)
        KeyFrameConverter("trainval", input_meta_path=trainval_meta, input_blob_paths=trainval_blobs,
            input_seg_path=trainval_seg, output_path=phase_path, zip_output=zip_output,
            compression=compression, **conversion_args).convert(debug)

        # convert test dataset
        print("Processing test datasets")
        phase_path = output_path / "test"
        if not phase_path.exists():
            phase_path.mkdir()

        test_meta = next(input_path.glob("*-test_meta.*"))
        test_blobs = list(p for p in input_path.glob("*blobs*") if 'test' in p.name)
        KeyFrameConverter("test", input_meta_path=test_meta, input_blob_paths=test_blobs,
            output_path=phase_path, zip_output=zip_output, compression=compression,
            **conversion_args).convert(debug)

def main():
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Convert nuscenes dataset tarballs to normal zip files with numpy arrays."
        "Multi-processing is not used since the metadata could be too large to be copied among processes."
        "The whole conversion process will takes hours.")

    parser.add_argument('input', type=str,
        help='Input directory')
    parser.add_argument('-o', '--output', type=str,
        help='Output file (in .zip format) or directory. If not provided, it will be the same as input')
    parser.add_argument('-d', '--debug', action="store_true",
        help='Run the script in debug mode, only convert part of the tarballs')
    parser.add_argument('-m', '--mini', action="store_true",
        help="Only convert the mini dataset")
    parser.add_argument('-i', '--store-intermediate-frames', dest="store_inter", type=int, default=0,
        help="Store certain number of intermediate frames before key frames. Disabled by default.")
    parser.add_argument('-v', '--estimate-box-velocity', dest="estimate_velocity", action="store_true",
        help="Estimate velocity of bounding boxes. Disabled by default")
    parser.add_argument('-z', '--zip', action="store_true",
        help="Convert the result into zip files rather than flat directory")
    parser.add_argument('-c', '--compression', type=str, default='stored', choices=['stored', 'deflated', 'bzip2', 'lzma'],
        help="Choose zip compression type")
    args = parser.parse_args()

    conversion_args = dict(store_inter = args.store_inter, estimate_velocity=args.estimate_velocity)
    convert_dataset_inpath(args.input, args.output or args.input,
        debug=args.debug, mini=args.mini,
        zip_output=args.compression if args.zip else False, **conversion_args)

if __name__ == "__main__":
    main()
