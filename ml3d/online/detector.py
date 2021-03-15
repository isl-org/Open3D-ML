#!/usr/bin/env python

###############################################################################
# The MIT License (MIT)
#
# Open3D: www.open3d.org
# Copyright (c) 2020 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
###############################################################################
"""
Online 3D object detection pipeline.

    - Connects to a RGBD camera or RGBD video file (currently
      RealSense camera and bag file format are supported).
    - Captures / reads color and depth frames.
    - Convert frames to point cloud.
    - Run object detector on point cloud.
    - Visualize point cloud video and results.
"""

import json
import time
import logging as log
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from open3d.ml.utils import Config
from open3d.ml.vis import BoundingBox3D, LabelLUT
from open3d.ml.datasets import utils
BEVBox3D = utils.bev_box.BEVBox3D


class PipelineViewer(object):
    """ GUI for the frame pipeline. """

    def __init__(self,
                 labels=None,
                 vfov=60,
                 pcd_name='points',
                 max_pcd_vertices=1 << 20,
                 score_threshold=0.1):
        """ Initialize GUI.

        Args:
            labels: List of class labels for the detector.
            vfov: Camera vertical field of view in degrees.
            pcd_name: Name for displayed point cloud.
            max_pcd_vertices: max vertices allowed in the point cloud display
                (default 2^20).
            score_threshold: Initial detector score threshold for valid
                detections
        """
        self.vfov = vfov
        self.lut = LabelLUT()

        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window(
            "Open3D || 3D Object Detection", 1024, 768)
        # Called on window layout (eg: resize)
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_window_close)  # Call on window close

        self.pcd_material = o3d.visualization.rendering.Material()
        self.pcd_material.shader = "defaultLit"
        # Set n_pixels displayed for each 3D point, accounting for HiDPI scaling
        self.pcd_material.point_size = 8 * self.window.scaling

        self.box_material = o3d.visualization.rendering.Material()
        self.box_material.shader = "defaultLine"
        # Bounding box line width
        self.box_material.line_width = 8 * self.window.scaling

        # 3D scene
        self.pcdview = gui.SceneWidget()
        self.window.add_child(self.pcdview)
        self.pcdview.enable_scene_caching(
            True)  # makes UI _much_ more responsive
        self.pcdview.scene = rendering.Open3DScene(self.window.renderer)
        self.pcdview.scene.set_background([1, 1, 1, 1])  # White background
        self.pcdview.scene.set_lighting(
            rendering.Open3DScene.LightingProfile.SOFT_SHADOWS, [0, -6, 0])
        self.pcdview.scene.show_axes(True)
        # Point cloud bounds, depends on the sensor range
        self.pcd_bounds = o3d.geometry.AxisAlignedBoundingBox([-3, -3, 0],
                                                              [3, 3, 6])
        self._camera_view()  # Initially look from the camera
        self.pcd_name = pcd_name
        self.max_pcd_vertices = max_pcd_vertices
        em = self.window.theme.font_size

        # Options panel
        self.panel = gui.Vert(em / 2, gui.Margins(em, 0, em, 0))
        self.window.add_child(self.panel)
        self.panel.add_fixed(em)  # top spacing

        self.flag_capture = False
        self.cv_capture = threading.Condition()
        toggle_capture = gui.Checkbox("Capture / Play")
        toggle_capture.checked = self.flag_capture
        toggle_capture.set_on_checked(self._on_toggle_capture)  # callback
        self.panel.add_child(toggle_capture)

        self.flag_detector = False
        self.score_threshold = score_threshold
        if labels:
            for val in sorted(labels):
                self.lut.add_label(val, val)
            toggle_detect = gui.Checkbox("Run Detector")
            toggle_detect.checked = self.flag_detector
            toggle_detect.set_on_checked(self._on_toggle_detector)  # callback
            self.panel.add_child(toggle_detect)
            self.panel.add_child(gui.Label("Detector score threshold"))
            slider_score_thr = gui.Slider(gui.Slider.DOUBLE)
            slider_score_thr.set_limits(0., 1.)
            slider_score_thr.double_value = self.score_threshold
            slider_score_thr.set_on_value_changed(self._on_score_thr_changed)
            self.panel.add_child(slider_score_thr)

        camera_view = gui.Button("Camera view")
        camera_view.set_on_clicked(self._camera_view)  # callback
        self.panel.add_child(camera_view)
        birds_eye_view = gui.Button("Bird's eye view")
        birds_eye_view.set_on_clicked(self._birds_eye_view)  # callback
        self.panel.add_child(birds_eye_view)
        self.panel.add_fixed(em)  # bottom spacing

        if labels:
            self.status_message = gui.Label("No detections")
            self.panel.add_child(self.status_message)
            self.panel.add_fixed(em)  # bottom spacing

        self.flag_exit = False
        self.flag_gui_empty = True
        self.label_list = []  # List of 3D text labels currently shown

    def update(self, frame_elements):
        """Update visualization with point cloud and bounding boxes
        Must run in main thread since this makes GUI calls.

        Args:
            frame_elements: dict {element_type: geometry element}.
                Dictionary of element types to geometry elements to be updated
                in the GUI: (self.pcd_name: point cloud, 'boxes':lineset,
                'labels': bbox description, 'status_message': message)
        """
        if self.flag_gui_empty:
            # Set dummy point cloud to allocate graphics memory
            dummy_pcd = o3d.t.geometry.PointCloud({
                'points':
                    o3d.core.Tensor.zeros((self.max_pcd_vertices, 3),
                                          o3d.core.Dtype.Float32),
                'colors':
                    o3d.core.Tensor.zeros((self.max_pcd_vertices, 3),
                                          o3d.core.Dtype.Float32)
            })
            self.pcdview.scene.add_geometry(self.pcd_name, dummy_pcd,
                                            self.pcd_material)
            self.flag_gui_empty = False
        else:
            if self.pcdview.scene.has_geometry('boxes'):
                self.pcdview.scene.remove_geometry('boxes')
                for label in self.label_list:
                    self.pcdview.remove_3d_label(label)

        # Update point cloud and add bounding boxes
        if 'boxes' in frame_elements and not frame_elements['boxes'].is_empty():
            self.pcdview.scene.add_geometry('boxes', frame_elements['boxes'],
                                            self.box_material)
            self.label_list = [
                self.pcdview.add_3d_label(pos, text)
                for pos, text in zip(*frame_elements['labels'])
            ]

        if 'status_message' in frame_elements:
            self.status_message.text = frame_elements["status_message"]

        self.pcdview.scene.scene.update_geometry(
            self.pcd_name, frame_elements[self.pcd_name].cpu(),
            rendering.Scene.UPDATE_POINTS_FLAG |
            rendering.Scene.UPDATE_COLORS_FLAG)
        self.pcdview.force_redraw()

    def _on_layout(self, theme):
        """ Callback on window initialize / resize. """
        frame = self.window.content_rect
        em = theme.font_size
        panel_size = self.panel.calc_preferred_size(theme)
        panel_rect = gui.Rect(frame.get_right() - panel_size.width - 2 * em,
                              frame.y + 2 * em, panel_size.width,
                              panel_size.height)
        self.panel.frame = panel_rect
        self.pcdview.frame = frame

    def _on_window_close(self):
        """ Callback when the user closes the application window. """
        self.flag_exit = True
        with self.cv_capture:
            self.cv_capture.notify_all()
        return True  # OK to close window

    def _on_toggle_detector(self, is_enabled):
        """ Callback to toggle the detector. """
        self.flag_detector = is_enabled

    def _on_toggle_capture(self, is_enabled):
        """ Callback to toggle capture. """
        self.flag_capture = is_enabled
        if is_enabled:
            with self.cv_capture:
                self.cv_capture.notify()

    def _camera_view(self):
        """ Callback to reset point cloud view to the camera. """
        self.pcdview.setup_camera(self.vfov, self.pcd_bounds, [0, 0, 0])
        # Look at [0, 0, 1] from camera placed at [0, 0, 0] with Y axis
        # pointing at [0, -1, 0]
        self.pcdview.scene.camera.look_at([0, 0, 1], [0, 0, 0], [0, -1, 0])

    def _birds_eye_view(self):
        """ Callback to reset point cloud view to birds eye (overhead) view. """
        self.pcdview.setup_camera(self.vfov, self.pcd_bounds, [0, 0, 0])
        self.pcdview.scene.camera.look_at([0, 0, 1.5], [0, 3, 1.5], [0, -1, 0])

    def _on_score_thr_changed(self, new_threshold_value):
        """ Callback to update detector model score threshold """
        self.score_threshold = new_threshold_value


class DetectorPipeline(object):
    """Capture RGBD frames, convert to point cloud, run detector and show
    bounding boxes overlayed on the Point Cloud."""

    def __init__(self,
                 detector_config_file=None,
                 camera_config_file=None,
                 rgbd_video=None,
                 device=None):
        """
        Args:
            detector_config_file: Optional YAML config file for detector
                pipeline.
            camera_config_file: Optional JSON camera configuration file.
            rgbd_video: Optional RGBD video file (RS bag).
                        This will be used over a camera if provided.
            device: Optional compute device in the format '{cpu|cuda}:DEVICE_ID', e.g.
                'cpu:0'.
        """

        # Detector
        if device:
            self.device = device.lower()
        else:
            self.device = 'cuda:0' if o3d.core.cuda.is_available() else 'cpu:0'
        self.o3d_device = o3d.core.Device(self.device)
        self.run_detector = None
        if detector_config_file:
            self.detector_config = Config.load_from_file(detector_config_file)
            ckpt_path = self.detector_config['model']['ckpt_path']
            if ckpt_path is None:  # No checkpoint in config file
                pass
            elif ckpt_path.endswith('.pth'):
                import open3d.ml.torch as ml3d
                import torch
                self.run_detector = self._run_detector_torch
                log.info("Using PyTorch for inference")
                self.net = ml3d.models.PointPillars(**
                                                    self.detector_config.model,
                                                    device=self.device)
                ckpt = torch.load(ckpt_path,
                                  map_location=self.device.split(':')[0])
                self.net.load_state_dict(ckpt['model_state_dict'])
                self.net.eval()

            else:
                import open3d.ml.tf as ml3d
                import tensorflow as tf
                self.run_detector = self._run_detector_tf
                log.info("Using Tensorflow for inference")
                self.net = ml3d.models.PointPillars(**
                                                    self.detector_config.model,
                                                    device=self.device)
                ckpt = tf.train.Checkpoint(step=tf.Variable(1), model=self.net)
                ckpt.restore(ckpt_path).expect_partial()

        if self.run_detector is None:
            log.info("No model checkpoint provided. Detector disabled.")
        else:
            log.info(
                f"Loaded model weights from {self.detector_config['model']['ckpt_path']}"
            )
        self.det_inputs = None

        self.video = None
        self.camera = None
        if rgbd_video:  # Video file
            self.video = o3d.t.io.RGBDVideoReader.create(rgbd_video)
            self.rgbd_metadata = self.video.metadata

        else:  # Depth camera
            self.camera = o3d.t.io.RealSenseSensor()
            if camera_config_file:
                with open(camera_config_file) as ccf:
                    self.camera.init_sensor(
                        o3d.t.io.RealSenseSensorConfig(json.load(ccf)))

            self.camera.start_capture()
            self.rgbd_metadata = self.camera.get_metadata()

        self.max_points = self.rgbd_metadata.width * self.rgbd_metadata.height
        log.info(self.rgbd_metadata)

        # RGBD -> PCD
        self.extrinsics = o3d.core.Tensor.eye(4, dtype=o3d.core.Dtype.Float32)
        self.intrinsic_matrix = o3d.core.Tensor(
            self.rgbd_metadata.intrinsics.intrinsic_matrix,
            dtype=o3d.core.Dtype.Float32)
        self.calib = {
            'world_cam':
                self.extrinsics.numpy(),
            'cam_img':
                np.block([[self.intrinsic_matrix.numpy(),
                           np.zeros((3, 1))], [np.zeros((1, 3)), 1]]).T
        }
        self.depth_max = 3.0  # m
        self.pcd_stride = 1  # downsample point cloud

        # GUI
        labels = self.detector_config['model'][
            'classes'] if self.run_detector else None
        vfov = np.rad2deg(2 * np.arctan(self.intrinsic_matrix[1, 2].item() /
                                        self.intrinsic_matrix[1, 1].item()))
        self.gui = PipelineViewer(labels=labels,
                                  vfov=vfov,
                                  pcd_name=self.rgbd_metadata.serial_number,
                                  max_pcd_vertices=self.max_points,
                                  score_threshold=self.net.bbox_head.score_thr
                                  if self.run_detector else 0.1)

    class _calib_wrapper(object):
        """FIXME: torch version of PointPillars.inference_end() needs calib as
        an attribute instead of dict key"""

        def __init__(self, calib):
            self.calib = calib

    def _run_detector_torch(self, pcd_frame):
        """ Run PyTorch 3D detector. """
        import torch
        from torch.utils import dlpack as torch_dlpack

        with torch.no_grad():
            if self.det_inputs is None:
                # Detector model requires 4 channel (XYZ+reflectance) input
                self.det_inputs = torch.ones((1, self.max_points, 4),
                                             dtype=torch.float32,
                                             device=self.device)
            pcd_points = pcd_frame.point['points']
            self.det_inputs[
                0, :pcd_points.shape[0], :3] = torch_dlpack.from_dlpack(
                    pcd_points.to_dlpack())

            self.net.bbox_head.score_thr = self.gui.score_threshold
            results = self.net(self.det_inputs[:, :pcd_points.shape[0], :])
            boxes = self.net.inference_end(results,
                                           self._calib_wrapper(self.calib))
            return boxes[0]

    def _run_detector_tf(self, pcd_frame):
        """ Run Tensorflow 3D detector. """
        import tensorflow as tf
        from tensorflow.experimental import dlpack as tf_dlpack

        with tf.device(self.device.replace('cuda', 'gpu').upper()):
            if self.det_inputs is None:
                # Detector model requires 4 channel (XYZ+reflectance) input
                self.det_inputs = tf.Variable(tf.ones((1, self.max_points, 4),
                                                      dtype=tf.float32),
                                              trainable=False)
            pcd_points = pcd_frame.point['points']
            self.det_inputs[0, :pcd_points.shape[0], :3].assign(
                tf_dlpack.from_dlpack(pcd_points.to_dlpack()))

            self.net.bbox_head.score_thr = self.gui.score_threshold
            results = self.net(self.det_inputs[:, :pcd_points.shape[0], :])
            boxes = self.net.inference_end(results, {
                'point': pcd_points,
                'calib': self.calib
            })
            return boxes[0]

    def launch(self):
        """ Launch frame pipeline thread and start GUI. """
        threading.Thread(name='FramePipeline', target=self._run).start()
        gui.Application.instance.run()

    def _run(self):
        """ Run pipeline. """
        with ThreadPoolExecutor(max_workers=1,
                                thread_name_prefix='Capture') as executor:
            n_pts = 0
            frame_id = 0
            t1 = time.perf_counter()
            if self.video:
                rgbd_frame = self.video.next_frame()
            else:
                rgbd_frame = self.camera.capture_frame(
                    wait=True, align_depth_to_color=True)

            pcd_errors = 0
            while (not self.gui.flag_exit and
                   (self.video and not self.video.is_eof())):
                if self.video:
                    future_rgbd_frame = executor.submit(self.video.next_frame)
                else:
                    future_rgbd_frame = executor.submit(
                        self.camera.capture_frame,
                        wait=True,
                        align_depth_to_color=True)

                try:
                    pcd_frame = o3d.t.geometry.PointCloud.create_from_rgbd_image(
                        rgbd_frame.to(self.o3d_device), self.intrinsic_matrix,
                        self.extrinsics, self.rgbd_metadata.depth_scale,
                        self.depth_max, self.pcd_stride)
                except RuntimeError:
                    pcd_errors += 1

                n_pts += pcd_frame.point['points'].shape[0]
                frame_elements = {self.rgbd_metadata.serial_number: pcd_frame}
                if pcd_frame.is_empty():
                    log.warning(f"No valid depth data in frame {frame_id})")
                    continue
                if self.gui.flag_detector:
                    bboxes = self.run_detector(pcd_frame)
                    for box in bboxes:
                        box.arrow_length = 0  # disable showing arrows
                    frame_elements['boxes'] = BoundingBox3D.create_lines(
                        bboxes, self.gui.lut)
                    frame_elements['labels'] = BoundingBox3D.create_labels(
                        bboxes, show_class=True, show_confidence=True)
                    log.debug(repr(bboxes))
                    frame_elements[
                        'status_message'] = f"{len(bboxes)} detections."

                gui.Application.instance.post_to_main_thread(
                    self.gui.window, lambda: self.gui.update(frame_elements))

                rgbd_frame = future_rgbd_frame.result()
                if frame_id % 30 == 0:
                    t0, t1 = t1, time.perf_counter()
                    log.debug(
                        f"\nframe_id = {frame_id}, \t {(t1-t0)*1000./30:0.2f}"
                        f"ms/frame \t {(t1-t0)*1e9/n_pts} ms/Mp\t",
                        end='')
                    n_pts = 0

                with self.gui.cv_capture:  # Wait for capture to be enabled
                    self.gui.cv_capture.wait_for(
                        predicate=lambda: self.gui.flag_capture or self.gui.
                        flag_exit)
                frame_id += 1

        if self.camera:
            self.camera.stop_capture()
        else:
            self.video.close()
        log.debug(f"create_from_depth_image() errors = {pcd_errors}")


if __name__ == "__main__":

    log.basicConfig(level=log.INFO)
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--camera-config',
                        help='RGBD camera configuration JSON file')
    parser.add_argument('--rgbd-video', help='RGBD video file (RealSense bag)')
    parser.add_argument('--detector-config',
                        required=False,
                        help='Detector configuration YAML file')
    parser.add_argument(
        '--device',
        help='Device to run model inference. e.g. cpu:0 or cuda:0 '
        'Default is CUDA GPU if available, else CPU.')

    args = parser.parse_args()
    if args.camera_config and args.rgbd_video:
        log.critical(
            "Please provide only one of --camera-config and --rgbd-video arguments"
        )
    else:
        DetectorPipeline(args.detector_config, args.camera_config,
                         args.rgbd_video, args.device).launch()
