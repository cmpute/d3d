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

class KeyFrameConverter:
    def __init__(self, phase, input_meta_path, input_blob_paths, output_path, input_seg_path=None):
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

        # nuscenes tables
        self.sample_table = None
        self.sample_data_table = None
        self.scene_table = None
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
        self.filename_table = None # filename -> (sample_data token, scene token, sensor name, order, file extension)
        self.scene_sensor_table = None # scene -> calibrated_sensor
        self.scene_map_table = None # scene -> {map category: map file}
        self.ohandles = {} # scene -> zipfile handle
        self.oscenes = set() # mark output scenes
        self.oframes = defaultdict(set) # mark output frames

    def _parse_scenes(self):
        self.log_table = _load_table(self.table_path / "log.json")
        self.map_table = _load_table(self.table_path / "map.json")

        # create log to map table
        log_map_table = defaultdict(dict)
        for mtoken, mdata in self.map_table.items():
            for ltoken in mdata["log_tokens"]:
                log_map_table[ltoken][mdata['category']] = mdata['filename']

        # create zipfile handles and save metadata
        self.scene_map_table = {}
        for stoken, data in self.scene_table.items():
            log = self.log_table[data['log_token']]
            self.scene_map_table[stoken] = log_map_table[data['log_token']]

            self.ohandles[stoken] = zipfile.ZipFile(self.output_path / ("%s.zip" % data['name']), "w")
            with self.ohandles[stoken].open("scene/stats.json", "w") as fout:
                meta = dict(
                    nbr_samples=data['nbr_samples'],
                    description=data['description'],
                    token=hex(stoken)[2:],
                    map=self.scene_map_table[stoken]
                )
                meta.update(log)
                fout.write(json.dumps(meta).encode())

        # clean used tables
        self.log_table = None
        self.map_table = None

    def _sort_samples(self):
        # sort samples and save timestamps
        self.sample_order = dict() 
        for stoken, data in self.scene_table.items():
            count = 0
            cur = data['first_sample_token']
            
            while True:
                # add mapping and save timestamp
                self.sample_order[cur] = (stoken, count)
                with self.ohandles[stoken].open("timestamp/%03d.txt" % count, "w") as fout:
                    fout.write(b'%d' % self.sample_table[cur]['timestamp'])

                # move to the next sample
                if self.sample_table[cur]['next'] is None:
                    break
                cur = self.sample_table[cur]['next']
                count += 1
                if count > 999:
                    raise RuntimeError("Frame index is larger than file name capacity!")

    def _parse_sample_data(self):
        # parse and reverse sample_data table
        self.filename_table = dict()
        self.scene_sensor_table = defaultdict(set)
        for token, data in self.sample_data_table.items():
            if not data['is_key_frame']:
                continue # only key frames are considered

            fname = data['filename']
            fname = fname[fname.rfind('/')+1:]
            scene, order = self.sample_order[data['sample_token']]
            ctoken = data['calibrated_sensor_token']
            sensor = self.calibrated_sensor_table[ctoken]['sensor_token']
            sensor = self.sensor_table[sensor]['channel'].lower()
            
            self.filename_table[fname] = (token, scene, sensor, order, data['fileformat'])
            if ctoken not in self.scene_sensor_table[scene]:
                self.scene_sensor_table[scene].add(ctoken)
            if self.lidarseg_table and token in self.lidarseg_table:
                fname = self.lidarseg_table[token]['filename']
                fname = fname[fname.rfind('/')+1:]
                self.filename_table[fname] = (token, scene, sensor + '_seg', order, 'bin')

    def _save_calibrations(self):
        for stoken, calib_tokens in self.scene_sensor_table.items():
            if stoken not in self.oscenes:
                continue

            with self.ohandles[stoken].open("scene/calib.json", "w") as fout:
                calib = dict()
                for ctoken in calib_tokens:
                    cdata = self.calibrated_sensor_table[ctoken]
                    sensor = cdata.pop('sensor_token')
                    sensor = self.sensor_table[sensor]['channel'].lower()
                    calib[sensor] = cdata
                fout.write(json.dumps(calib).encode())

    def _save_annotations(self):
        self.instance_table = _load_table(self.table_path / "instance.json")
        self.attribute_table = _load_table(self.table_path / "attribute.json")
        self.category_table = _load_table(self.table_path / "category.json")
        self.annotation_table = _load_table(self.table_path / "sample_annotation.json")
        anno_list = defaultdict(list)

        # parse annotations from table
        for itoken, data in self.instance_table.items():
            cur = data['first_annotation_token']
            instance_id = hex(itoken)[2:]
            instance_category = self.category_table[data['category_token']]['name']
            
            while True:
                adata = self.annotation_table[cur]
                scene, order = self.sample_order[adata['sample_token']]
                if order in self.oframes[scene]:
                    # save this annotation if the sample has corresponding sensor data
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
                    anno_list[adata['sample_token']].append(anno)

                # move to the next sample
                if adata['next'] is None:
                    break
                cur = adata['next']

        # save annotations
        for stoken, annos in anno_list.items():
            scene, order = self.sample_order[stoken]
            with self.ohandles[scene].open("annotation/%03d.json" % order, "w") as fout:
                fout.write(json.dumps(annos).encode())

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
            shutil.copy(self.table_path / "lidarseg_category.json", self.output_path)

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
        self._sort_samples()
        self._parse_sample_data()

    def load_blobs(self, debug):
        # iterate through all the sensor data
        blobs = self.blob_paths
        if self.seg_path is not None:
            blobs = [self.seg_path] + blobs

        # iterate through all the sensor data
        for iblob, blob_path in tqdm(enumerate(blobs), desc="Loading blobs", unit="tars"):
            blob_file = tarfile.open(blob_path)
            if debug and iblob > 0:
                break

            for counter, tinfo in enumerate(tqdm(blob_file, desc="Reading files", unit="files")):
                # skip files that are not samples
                if tinfo.isdir():
                    continue
                fname = PurePath(tinfo.name).name
                if fname not in self.filename_table:
                    continue

                # save sample data and pose
                token, scene, sensor, order, ext = self.filename_table[fname]
                if sensor.endswith("seg") and self.phase not in tinfo.name:
                    continue # dealing with duplicates in mini and trainval blobs

                with self.ohandles[scene].open("%s/%03d.%s" % (sensor, order, ext), "w") as fout:
                    shutil.copyfileobj(blob_file.extractfile(tinfo), fout)
                if sensor == "lidar_top": # here we choose the pose of lidar as the pose of the key frame
                # TODO: save pose and timestamp for each sensor, can be in one single json file
                    ptoken = self.sample_data_table[token]["ego_pose_token"]
                    pose = self.ego_pose_table[ptoken]
                    with self.ohandles[scene].open("pose/%03d.json" % order, "w") as fout:
                        fout.write(json.dumps(dict(
                            rotation=pose["rotation"], translation=pose["translation"]
                        )).encode())

                # tag the scene to be filled
                if scene not in self.oscenes:
                    self.oscenes.add(scene)
                if order not in self.oframes[scene]:
                    self.oframes[scene].add(order)

                # only convert 2 files in debug
                if debug and counter > 1:
                    break

    def save_metadata(self):
        # save things stored in metadata
        print("Saving metadata...")
        self._save_calibrations()
        self._save_annotations()
        self._save_definitions()

    def clean_up(self):
        # clean zip files and close handles
        for scene, handle in self.ohandles.items():
            nsamples = self.scene_table[scene]['nbr_samples']

            # remove archives where the data is incomplete
            if scene not in self.oscenes or len(self.oframes[scene]) < nsamples:
                handle.close()
                os.remove(handle.filename)
            
            # fill scene samples without annotation
            else:
                nlist = set(handle.namelist())
                for i in range(nsamples):
                    aname = "annotation/%03d.json" % i
                    if aname not in nlist:
                        with self.ohandles[scene].open(aname, "w") as fout:
                            fout.write(b"[]")
                handle.close()

    def convert(self, debug=False):
        try:
            self.load_metadata()
            self.load_blobs(debug=debug)
            self.save_metadata()
            self.clean_up()
        finally:
            if self.temp_dir is not None:
                shutil.rmtree(self.temp_dir)

def convert_dataset_inpath(input_path, output_path, debug=False, mini=False):
    input_path, output_path = Path(input_path), Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    if mini: # convert mini dataset
        phase_path = output_path / "trainval"
        if not phase_path.exists():
            phase_path.mkdir()

        mini_archive = next(input_path.glob("*-mini.*"))
        mini_seg = list(input_path.glob("*lidarseg-mini*"))
        mini_seg = mini_seg[0] if len(mini_seg) > 0 else None
        KeyFrameConverter("mini", input_meta_path=mini_archive, input_blob_paths=[mini_archive],
            input_seg_path=mini_seg, output_path=phase_path).convert(debug)
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
            input_seg_path=trainval_seg, output_path=phase_path).convert(debug)

        # convert test dataset
        print("Processing test datasets")
        phase_path = output_path / "test"
        if not phase_path.exists():
            phase_path.mkdir()

        test_meta = next(input_path.glob("*-test_meta.*"))
        test_blobs = list(p for p in input_path.glob("*blobs*") if 'test' in p.name)
        KeyFrameConverter("test", input_meta_path=test_meta, input_blob_paths=test_blobs,
            output_path=phase_path).convert(debug)

def main():
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Convert nuscenes dataset tarballs to normal zip files with numpy arrays."
        "Multi-processing is not used since the metadata could be too large to be copied among processes.")

    parser.add_argument('input', type=str,
        help='Input directory')
    parser.add_argument('-o', '--output', type=str,
        help='Output file (in .zip format) or directory. If not provided, it will be the same as input')
    parser.add_argument('-d', '--debug', action="store_true",
        help='Run the script in debug mode, only convert part of the tarballs')
    parser.add_argument('-m', '--mini', action="store_true",
        help="Only convert the mini dataset")
    parser.add_argument('-k', '--all-frames', dest="allframes", action="store_true",
        help="Convert all data frames. By default only key frames are preserved.")
    parser.add_argument('-u', '--unzip', action="store_true",
        help="Convert the result into directory rather than zip files")
    args = parser.parse_args()

    if args.unzip: # XXX: implement this
        raise NotImplementedError("Converting into directories is not implemented")
    if args.allframes:
        # XXX: implement this.
        #      To support all frames we may store all the data just using their token or using a data structure like rosbag.
        #      Canbus extension and Vector map should be included when converting all frames
        raise NotImplementedError("Converting all frames is not implemented")

    convert_dataset_inpath(args.input, args.output or args.input, debug=args.debug, mini=args.mini)

if __name__ == "__main__":
    main()
