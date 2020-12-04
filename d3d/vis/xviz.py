try:
    import xviz_avs as xa
    from xviz_avs.builder import (XVIZBuilder, XVIZMetadataBuilder,
                                  XVIZUIBuilder)
    from xviz_avs.io import DirectorySource, XVIZGLBWriter
    from xviz_avs.v2.session_pb2 import StateUpdate
except ImportError:
    raise SystemError("Please install xviz library.")

from enum import Enum

import numpy as np
from d3d.abstraction import Target3DArray, TransformSet
from d3d.dataset.base import TrackingDatasetBase
from matplotlib import pyplot as plt
from tqdm import trange


def _parse_color(color, tag_enum):
    '''
    Broadcast color in different tags and convert color values to 0~255 range.
    '''
    if isinstance(color, (tuple, list)):
        color = {k: color for k in tag_enum}
    for k in tag_enum:
        if all(c <= 1 for c in color[k]):
            color[k] = [int(c * 255) for c in color[k]]
    return color

def visualize_detections_metadata(builder: XVIZMetadataBuilder, tag_enum: Enum, stream_prefix='/tracklets',
    box_color=(1, 1, 1), text_color=(1, 1, 1)):
    '''
    :param tag_enum: Enumeration of all possible tags.
    :param box_color: tuple or dict of tuple. Define bounding box color for each category
    :param text_color: tuple or dict of tuple. Define text color for each category
    '''
    stream_prefix = stream_prefix.rstrip("/")
    box_color = _parse_color(box_color, tag_enum)
    text_color = _parse_color(text_color, tag_enum)
    
    obj_builder = builder.stream(stream_prefix + '/objects')\
        .category(xa.CATEGORY.PRIMITIVE)\
        .type(xa.PRIMITIVE_TYPES.POLYGON)\
        .coordinate(xa.COORDINATE_TYPES.VEHICLE_RELATIVE)\
        .stream_style({
            "extruded": True,
            "fill_color": "#00000080"
        })
    for tag in tag_enum:
        color = box_color[tag]
        if len(color) == 3:
            cfill = color + [128]
            cstroke = color
        else:
            cfill = color
            cstroke = color[:3]
        obj_builder.style_class(tag.name, {
            "fill_color": cfill,
            "stroke_color": cstroke
        })

    builder.stream(stream_prefix + '/tracking_point')\
        .category(xa.CATEGORY.PRIMITIVE)\
        .type(xa.PRIMITIVE_TYPES.CIRCLE)\
        .coordinate(xa.COORDINATE_TYPES.VEHICLE_RELATIVE)\
        .stream_style({
            "radius": 0.2,
            "stroke_width": 0,
            "fill_color": "#FFC043"
        })

    builder.stream(stream_prefix + '/label')\
        .category(xa.CATEGORY.PRIMITIVE)\
        .type(xa.PRIMITIVE_TYPES.TEXT)\
        .coordinate(xa.COORDINATE_TYPES.VEHICLE_RELATIVE)\
        .stream_style({
            "text_size": 18,
            "fill_color": "#DCDCCD"
        })

def visualize_detections(builder: XVIZBuilder, visualizer_frame: str, targets: Target3DArray, calib: TransformSet,
    stream_prefix: str, id_prefix="", tags=None, text_offset=None):
    '''
    Add detection results to xviz builder
    '''
    # change frame to the same
    if targets.frame != visualizer_frame:
        targets = calib.transform_objects(targets, frame_to=visualizer_frame)

    stream_prefix = stream_prefix.rstrip("/")

    for box in targets:
        vertices = box.corners[[0,1,3,2,0]]
        builder.primitive(stream_prefix + "/objects")\
            .polygon(vertices.tolist())\
            .id(box.tid64)\
            .style({"height": box.dimension[2]})\
            .classes([box.tag.mapping(t).name for t in box.tag.labels])

        builder.primitive(stream_prefix + "/label")\
            .text("#" + box.tid64)\
            .position(box.position if text_offset is None else box.position + text_offset)

        builder.primitive(stream_prefix + "/tracking_point")\
            .circle(box.position, 0.2)\
            .id(box.tid64)

class TrackingDatasetConverter:
    '''
    This class converts tracking dataset to data blobs like https://github.com/uber/xviz-data
    You can derive this class and custom the visualization results
    '''
    def __init__(self, loader: TrackingDatasetBase, lidar_names = None, camera_names = None, lidar_colormap = "hot"):
        '''
        :param lidar_names: Frame names of lidar to be visualized
        :param camera_names: Frame names of camera to be visualized
        :param lidar_colormap: Matplotlib colormap used to color lidar points
        '''
        self._loader = loader
        assert loader.nframes == 0

        self._lidar_names = lidar_names or self._loader.VALID_LIDAR_NAMES
        self._camera_names = camera_names or self._loader.VALID_CAM_NAMES
        if isinstance(lidar_colormap, str):
            self._lidar_colormap = plt.get_cmap(lidar_colormap)
        else:
            self._lidar_colormap = lidar_colormap

        self._metadata = None

    def get_metadata(self, seq_id):
        builder = XVIZMetadataBuilder()
        builder.start_time(self._loader.timestamp((seq_id, 0)) / 1e6)\
            .end_time(self._loader.timestamp((seq_id, self._loader.sequence_sizes[seq_id] - 1)) / 1e6)
        builder.stream('/vehicle_pose').category(xa.CATEGORY.POSE)
        builder.stream("/vehicle/autonomy_state")\
            .category(xa.CATEGORY.TIME_SERIES)\
            .type("string")

        # add lidars
        for name in self._lidar_names:
            builder.stream('/lidar/' + name)\
                .coordinate(xa.COORDINATE_TYPES.VEHICLE_RELATIVE)\
                .category(xa.CATEGORY.PRIMITIVE)\
                .type(xa.PRIMITIVE_TYPES.POINT)\
                .stream_style({
                    'radius_pixels': 1
                })

        # add images
        for name in self._camera_names:
            builder.stream('/camera/' + name)\
                .category(xa.CATEGORY.PRIMITIVE)\
                .type(xa.PRIMITIVE_TYPES.IMAGE)

        # add objects
        box_colors = {}
        for clsname in self._loader.VALID_OBJ_CLASSES:
            colors = np.random.rand(3) * 256
            box_colors[clsname] = colors.astype('u1').tolist()
        visualize_detections_metadata(builder, self._loader.VALID_OBJ_CLASSES, box_color=box_colors)

        # add UI configuration
        ui_builder = XVIZUIBuilder()
        cam_panel = ui_builder.panel('Camera')
        cam_panel.child(ui_builder.video(['/camera/' + n for n in self._camera_names]))
        ui_builder.child(cam_panel)

        builder.ui(ui_builder)

        self._metadata = builder.get_message()
        return self._metadata

    def add_lidars(self, builder, idx_tuple):
        calib = self._loader.calibration_data(idx_tuple)
        clouds = self._loader.lidar_data(idx_tuple, names=self._lidar_names)
        for name, cloud in zip(self._lidar_names, clouds):
            cloud = calib.transform_points(cloud, frame_to="bottom_center") # RT reversed?
            intensities = cloud[:, 3]
            intensities = (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities))
            intensities = (self._lidar_colormap(intensities) * 255).astype('u1')
            builder.primitive("/lidar/" + name)\
                .points(cloud[:, :3])\
                .colors(intensities)

    def add_cameras(self, builder, idx_tuple, birate=250000):
        images = self._loader.camera_data(idx_tuple, names=self._camera_names)
        for name, image in zip(self._camera_names, images):
            scale = birate / (image.width * image.height)
            scale_w, scale_h = int(image.width * scale), int(image.height * scale)
            image.thumbnail((scale_w, scale_h))
            builder.primitive("/camera/" + name).image(image)

    def add_pose(self, builder, idx_tuple, timestamp):
        init_pose = self._loader.pose((idx_tuple[0], 0))
        x0, y0, z0 = init_pose.position

        pose = self._loader.pose(idx_tuple)
        x, y, z = pose.position
        yaw, pitch, roll = pose.orientation.as_euler("ZYX")

        builder.pose().timestamp(timestamp)\
            .position(x - x0, y - y0, z - z0)\
            .orientation(roll, pitch, yaw)
        return timestamp

    def add_objects(self, builder, idx_tuple):
        visualize_detections(builder, "bottom_center",
            self._loader.annotation_3dobject(idx_tuple),
            self._loader.calibration_data(idx_tuple),
            "/tracklets"
        )

    def dump_sequence(self, output_path, seq_id):
        sink = DirectorySource(output_path)
        writer = XVIZGLBWriter(sink, image_encoding='JPEG', use_xviz_extension=False)
        writer.write_message(self.get_metadata(seq_id))

        for frame_idx in trange(self._loader.sequence_sizes[seq_id]):
            builder = XVIZBuilder(self._metadata, update_type=StateUpdate.UpdateType.SNAPSHOT)
            idx_tuple = (seq_id, frame_idx)
            timestamp = self._loader.timestamp(idx_tuple) / 1e6

            self.add_pose(builder, idx_tuple, timestamp)
            self.add_lidars(builder, idx_tuple)
            self.add_cameras(builder, idx_tuple)
            self.add_objects(builder, idx_tuple)
            builder.time_series("/vehicle/autonomy_state")\
                .timestamp(timestamp)\
                .value("autonomous")

            data = builder.get_message()
            writer.write_message(data)

        writer.close()
