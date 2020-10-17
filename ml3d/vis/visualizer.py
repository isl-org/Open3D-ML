import numpy as np
import threading
import open3d as o3d
from open3d.visualization import gui
from open3d.visualization import rendering
from collections import deque
from .colormap import *
from .labellut import *

import time


class Model:
    """Attributes, data, and methods of visulization models."""

    def __init__(self):
        # Note: the tpointcloud cannot store the actual data arrays, because
        # the tpointcloud requires specific names for some arrays (e.g. "points",
        # "colors"). So the tpointcloud exists for rendering and initially only
        # contains the "points" array.
        self.tclouds = {}  # name -> tpointcloud
        self.data_names = []  # the order data will be displayed / animated

        self._data = {}  # name -> {attr_name -> numpyarray}
        self._known_attrs = {}  # name -> set(attrs)
        self._attr2minmax = {}  # only access in _get_attr_minmax()

        self._attr_rename = {"label": "labels", "feat": "feature"}

    def _init_data(self, name):
        tcloud = o3d.t.geometry.PointCloud(o3d.core.Dtype.Float32,
                                           o3d.core.Device("CPU:0"))
        self.tclouds[name] = tcloud
        self._data[name] = {}
        self.data_names.append(name)

    def is_loaded(self, name):
        return len(self._data[name]) > 0

    def load(self, name, fail_if_no_space=False):
        assert (False)  # pure virtual

    def unload(self, name):
        assert (False)  # pure virtual

    def create_point_cloud(self, data):
        assert ("name" in data)  # name is a required field
        assert ("points" in data)  # 'points' is a required field

        name = data["name"]
        pts = self._convert_to_numpy(data["points"])
        tcloud = o3d.t.geometry.PointCloud(o3d.core.Dtype.Float32,
                                           o3d.core.Device("CPU:0"))
        known_attrs = set()
        if pts.shape[1] >= 4:
            # We can't use inplace Tensor creation (e.g. from_numpy())
            # because the resulting arrays won't be contiguous. However,
            # TensorList can be inplace.
            xyz = pts[:, [0, 1, 2]]
            tcloud.point["points"] = Visualizer._make_tcloud_array(xyz,
                                                                   copy=True)
        else:
            tcloud.point["points"] = Visualizer._make_tcloud_array(pts)
        self.tclouds[name] = tcloud

        # Add scalar attributes and vector3 attributes
        attrs = {}
        for k, v in data.items():
            attr = self._convert_to_numpy(v)
            if attr is None:
                continue
            attr_name = k
            if attr_name == "point":
                continue

            new_name = self._attr_rename.get(attr_name)
            if new_name is not None:
                attr_name = new_name

            if len(attr.shape) == 1 or len(attr.shape) == 2:
                attrs[attr_name] = attr
                known_attrs.add(attr_name)

        self._data[name] = attrs
        self._known_attrs[name] = known_attrs

    def _convert_to_numpy(self, ary):
        if isinstance(ary, list):
            return np.array(ary, dtype='float32')
        elif isinstance(ary, np.ndarray):
            if len(ary.shape) == 2 and ary.shape[0] == 1:
                ary = ary[0]  # "1D" array as 2D: [[1, 2, 3,...]]
            if ary.dtype.name.startswith('int'):
                return np.array(ary, dtype='float32')
            else:
                return ary

        try:
            import tensorflow as tf
            if isinstance(ary, tf.Tensor):
                return self._convert_to_numpy(ary.numpy())
        except:
            pass

        try:
            import torch
            if isinstance(ary, torch.Tensor):
                return self._convert_to_numpy(ary.detach().cpu().numpy())
        except:
            pass

        return None

    def get_attr(self, name, attr_name):
        if name in self._data:
            attrs = self._data[name]
            if attr_name in attrs:
                return attrs[attr_name]
        return None

    def get_attr_shape(self, name, attr_name):
        attr = self.get_attr(name, attr_name)
        if attr is not None:
            return attr.shape
        return []

    def get_attr_minmax(self, attr_name, channel):
        attr_key_base = attr_name + ":" + str(channel)

        attr_min = 1e30
        attr_max = -1e30
        for name in self._data.keys():
            key = name + ":" + attr_key_base
            if key not in self._attr2minmax:
                attr = self.get_attr(name, attr_name)
                if attr is None:  # clouds may not have all the same attributes
                    continue
                if len(attr.shape) > 1:
                    attr = attr[:, channel]
                self._attr2minmax[key] = (attr.min(), attr.max())
            amin, amax = self._attr2minmax[key]
            attr_min = min(attr_min, amin)
            attr_max = max(attr_max, amax)

        if attr_min > attr_max:
            return (0.0, 0.0)
        return (attr_min, attr_max)

    def get_available_attrs(self, names):
        attr_names = None
        for n in names:
            known = self._known_attrs.get(n)
            if known is not None:
                if attr_names is None:
                    attr_names = known
                else:
                    attr_names = attr_names.intersection(known)
        if attr_names is None:
            return []
        return sorted(attr_names)

    def calc_bounds_for(self, name):
        if name in self.tclouds and not self.tclouds[name].is_empty():
            tcloud = self.tclouds[name]
            # Ideally would simply return tcloud.compute_aabb() here, but it can
            # be very slow on macOS with clang 11.0
            pts = tcloud.point["points"].as_tensor().numpy()
            min_val = (pts[:, 0].min(), pts[:, 1].min(), pts[:, 2].min())
            max_val = (pts[:, 0].max(), pts[:, 1].max(), pts[:, 2].max())
            return [min_val, max_val]
        else:
            return [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]


class DataModel(Model):
    """Class for data i/o and storage."""

    def __init__(self, userdata):
        super().__init__()
        # We could just create the TPointCloud here, but that would cause the UI
        # to block. If we do it on load then the loading dialog will display.
        self._name2srcdata = {}
        for d in userdata:
            name = d["name"]
            while name in self._data:  # ensure each name is unique
                name = name + "_"
            self._init_data(name)
            self._name2srcdata[name] = d

    def load(self, name, fail_if_no_space=False):
        if self.is_loaded(name):
            return

        self.create_point_cloud(self._name2srcdata[name])

    def unload(self, name):
        pass


class DatasetModel(Model):

    def __init__(self, dataset, split, indices):
        super().__init__()
        self._dataset = None
        self._name2datasetidx = {}
        self._memory_limit = 8192 * 1024 * 1024  # memory limit in bytes
        self._current_memory_usage = 0
        self._cached_data = deque()

        self._dataset = dataset.get_split(split)
        if len(self._dataset) > 0:
            if indices is None:
                indices = range(0, len(self._dataset))
            # Some results from get_split() (like "training") are randomized.
            # Sort, so that the same index always returns the same piece of data.
            path2idx = {}
            for i in range(0, len(self._dataset.path_list)):
                path2idx[self._dataset.path_list[i]] = i
            real_indices = [path2idx[p] for p in sorted(path2idx.keys())]
            indices = [real_indices[idx] for idx in indices]

            for i in indices:
                info = self._dataset.get_attr(i)
                name = info["name"]
                while name in self._data:  # ensure each name is unique
                    name = name + "_"

                self._init_data(name)
                self._name2datasetidx[name] = i

            if dataset.__class__.__name__ in [
                    "Toronto3D", "Semantic3D", "S3DIS"
            ]:
                self._attr_rename["feat"] = "colors"
                self._attr_rename["feature"] = "colors"
        else:
            print("[ERROR] Dataset split has no data")

    def is_loaded(self, name):
        loaded = super().is_loaded(name)
        if loaded:
            # make this point cloud the most recently used
            self._cached_data.remove(name)
            self._cached_data.append(name)
        return loaded

    def load(self, name, fail_if_no_space=False):
        assert (name in self._name2datasetidx)

        if self.is_loaded(name):
            return True

        idx = self._name2datasetidx[name]
        data = self._dataset.get_data(idx)
        data["name"] = name
        data["points"] = data["point"]

        self.create_point_cloud(data)
        size = self._calc_pointcloud_size(self._data[name], self.tclouds[name])
        if size + self._current_memory_usage > self._memory_limit:
            if fail_if_no_space:
                self.unload(name)
                return False
            else:
                # Remove oldest from cache
                remove_name = self._cached_data.popleft()
                remove_size = self._calc_pointcloud_size(
                    self._data[remove_name], self.tclouds[remove_name])
                self._current_memory_usage -= remove_size
                self.unload(remove_name)
                # Add new point cloud to cache
                self._cached_data.append(name)
                self._current_memory_usage += size
                return True
        else:
            self._current_memory_usage += size
            self._cached_data.append(name)
            return True

    def _calc_pointcloud_size(self, raw_data, pcloud):
        pcloud_size = 0
        for (attr, arr) in raw_data.items():
            pcloud_size += arr.size * 4
        # Point cloud consumes 64 bytes of per point of GPU memory
        pcloud_size += pcloud.point["points"].size * 64
        return pcloud_size

    def unload(self, name):
        # Only unload if this was loadable; we might have an in-memory,
        # user-specified data created directly through create_point_cloud().
        if name in self._name2datasetidx:
            tcloud = o3d.t.geometry.PointCloud(o3d.core.Dtype.Float32,
                                               o3d.core.Device("CPU:0"))
            self.tclouds[name] = tcloud
            self._data[name] = {}


class Visualizer:
    """Visualizer for Dataset objects and custom point clouds"""

    class LabelLUTEdit:

        def __init__(self):
            self.widget = gui.TreeView()
            self._on_changed = None  # takes no args, returns no value
            self.clear()

        def clear(self):
            self.widget.clear()
            self._label2color = {}

        def is_empty(self):
            return len(self._label2color) == 0

        def get_colors(self):
            return [
                self._label2color[label]
                for label in sorted(self._label2color.keys())
            ]

        def set_on_changed(self, callback):  # takes no args, no return value
            self._on_changed = callback

        def set_labels(self, labellut):
            self.widget.clear()
            root = self.widget.get_root_item()
            for key in sorted(labellut.labels.keys()):
                lbl = labellut.labels[key]
                color = lbl.color
                if len(color) == 3:
                    color += [1.0]
                self._label2color[key] = color
                color = gui.Color(lbl.color[0], lbl.color[1], lbl.color[2])
                cell = gui.LUTTreeCell(
                    str(key) + ": " + lbl.name, True, color, None, None)
                cell.checkbox.set_on_checked(
                    self._make_on_checked(key, self._on_label_checked))
                cell.color_edit.set_on_value_changed(
                    self._make_on_color_changed(key,
                                                self._on_label_color_changed))
                self.widget.add_item(root, cell)

        def _make_on_color_changed(self, label, member_func):

            def on_changed(color):
                member_func(label, color)

            return on_changed

        def _on_label_color_changed(self, label, gui_color):
            self._label2color[label] = [
                gui_color.red, gui_color.green, gui_color.blue,
                self._label2color[label][3]
            ]
            if self._on_changed is not None:
                self._on_changed()

        def _make_on_checked(self, label, member_func):

            def on_checked(checked):
                member_func(label, checked)

            return on_checked

        def _on_label_checked(self, label, checked):
            if checked:
                alpha = 1.0
            else:
                alpha = 0.0
            color = self._label2color[label]
            self._label2color[label] = [color[0], color[1], color[2], alpha]
            if self._on_changed is not None:
                self._on_changed()

    class ColormapEdit:

        def __init__(self, window, em):
            self.colormap = None
            self.widget = gui.Vert()
            self._window = window
            self._min_value = 0.0
            self._max_value = 1.0
            self._on_changed = None  # takes no args, no return value
            self._itemid2idx = {}

            self._min_label = gui.Label("")
            self._max_label = gui.Label("")
            grid = gui.VGrid(2)
            grid.add_child(gui.Label("Range (min):"))
            grid.add_child(self._min_label)
            grid.add_child(gui.Label("Range (max):"))
            grid.add_child(self._max_label)
            self.widget.add_child(grid)
            self.widget.add_fixed(0.5 * em)
            self.widget.add_child(gui.Label("Colormap"))
            self._edit = gui.TreeView()
            self._edit.set_on_selection_changed(self._on_selection_changed)
            self.widget.add_child(self._edit)

            self._delete = gui.Button("Delete")
            self._delete.horizontal_padding_em = 0.5
            self._delete.vertical_padding_em = 0
            self._delete.set_on_clicked(self._on_delete)
            self._add = gui.Button("Add")
            self._add.horizontal_padding_em = 0.5
            self._add.vertical_padding_em = 0
            self._add.set_on_clicked(self._on_add)
            h = gui.Horiz()
            h.add_stretch()
            h.add_child(self._delete)
            h.add_fixed(0.25 * em)
            h.add_child(self._add)
            h.add_stretch()
            self.widget.add_fixed(0.5 * em)
            self.widget.add_child(h)
            self.widget.add_fixed(0.5 * em)

        def set_on_changed(self, callback):  # takes no args, no return value
            self._on_changed = callback

        def update(self, colormap, min_val, max_val):
            self.colormap = colormap

            self._min_value = min_val
            self._max_value = max_val
            self._min_label.text = str(min_val)
            self._max_label.text = str(max_val)

            if self._min_value >= self._max_value:
                self._max_value = self._min_value + 1.0

            self._edit.clear()
            self._itemid2idx = {}
            root_id = self._edit.get_root_item()
            for i in range(0, len(self.colormap.points)):
                p = self.colormap.points[i]
                color = gui.Color(p.color[0], p.color[1], p.color[2])
                val = min_val + p.value * (max_val - min_val)
                cell = gui.ColormapTreeCell(val, color, None, None)
                cell.color_edit.set_on_value_changed(
                    self._make_on_color_changed(i, self._on_color_changed))
                cell.number_edit.set_on_value_changed(
                    self._make_on_value_changed(i, self._on_value_changed))
                item_id = self._edit.add_item(root_id, cell)
                self._itemid2idx[item_id] = i
            self._update_buttons_enabled()

        def _make_on_color_changed(self, idx, member_func):

            def on_changed(color):
                member_func(idx, color)

            return on_changed

        def _on_color_changed(self, idx, gui_color):
            self.colormap.points[idx].color = [
                gui_color.red, gui_color.green, gui_color.blue
            ]
            if self._on_changed is not None:
                self._on_changed()

        def _make_on_value_changed(self, idx, member_func):

            def on_changed(value):
                member_func(idx, value)

            return on_changed

        def _on_value_changed(self, idx, value):
            value = (value - self._min_value) / (self._max_value -
                                                 self._min_value)
            needs_update = False
            value = min(1.0, max(0.0, value))

            if ((idx > 0 and value < self.colormap.points[idx - 1].value) or
                (idx < len(self.colormap.points) - 1 and
                 value > self.colormap.points[idx + 1].value)):
                self.colormap.points[idx].value = value
                o = self.colormap.points[idx]
                self.colormap.points.sort(key=lambda cmap_pt: cmap_pt.value)
                for i in range(0, len(self.colormap.points)):
                    if self.colormap.points[i] is o:
                        idx = i
                        break
                needs_update = True
            if idx > 0 and value == self.colormap.points[idx - 1].value:
                if idx < len(self.colormap.points):
                    upper = self.colormap.points[idx + 1].value
                else:
                    upper = 1.0
                value = value + 0.5 * (upper - value)
                needs_update = True
            if idx < len(self.colormap.points
                        ) - 1 and value == self.colormap.points[idx + 1].value:
                if idx > 0:
                    lower = self.colormap.points[idx - 1].value
                else:
                    lower = 0.0
                value = lower + 0.5 * (value - lower)
                needs_update = True

            self.colormap.points[idx].value = value

            if needs_update:
                self._update_later()

            if self._on_changed is not None:
                self._on_changed()

        def _on_selection_changed(self, item_id):
            self._update_buttons_enabled()

        def _on_delete(self):
            if len(self.colormap.points) > 2:
                idx = self._itemid2idx[self._edit.selected_item]
                self.colormap.points = self.colormap.points[:
                                                            idx] + self.colormap.points[
                                                                idx + 1:]
                del self._itemid2idx[self._edit.selected_item]
                self._update_later()
                if self._on_changed is not None:
                    self._on_changed()

        def _on_add(self):
            if self._edit.selected_item in self._itemid2idx:  # maybe no selection
                idx = self._itemid2idx[self._edit.selected_item]
                if idx < len(self.colormap.points) - 1:
                    lower = self.colormap.points[idx]
                    upper = self.colormap.points[idx + 1]
                else:
                    lower = self.colormap.points[len(self.colormap.points) - 2]
                    upper = self.colormap.points[len(self.colormap.points) - 1]
                add_idx = min(idx + 1, len(self.colormap.points) - 1)
                new_value = lower.value + 0.5 * (upper.value - lower.value)
                new_color = [
                    0.5 * lower.color[0] + 0.5 * upper.color[0],
                    0.5 * lower.color[1] + 0.5 * upper.color[1],
                    0.5 * lower.color[2] + 0.5 * upper.color[2]
                ]
                new_point = Colormap.Point(new_value, new_color)
                self.colormap.points = self.colormap.points[:add_idx] + [
                    new_point
                ] + self.colormap.points[add_idx:]
                self._update_later()
                if self._on_changed is not None:
                    self._on_changed()

        def _update_buttons_enabled(self):
            if self._edit.selected_item in self._itemid2idx:
                self._delete.enabled = len(self.colormap.points) > 2
                self._add.enabled = True
            else:
                self._delete.enabled = False
                self._add.enabled = False

        def _update_later(self):

            def update():
                self.update(self.colormap, self._min_value, self._max_value)
                self._window.post_redraw()  # need to manually request redraw

            gui.Application.instance.post_to_main_thread(self._window, update)

    class ProgressDialog:

        def __init__(self, title, window, n_items):
            self._window = window
            self._n_items = n_items

            em = window.theme.font_size
            self.dialog = gui.Dialog(title)
            self._label = gui.Label(title + "                    ")
            self._layout = gui.Vert(0, gui.Margins(em, em, em, em))
            self.dialog.add_child(self._layout)
            self._layout.add_child(self._label)
            self._layout.add_fixed(0.5 * em)
            self._progress = gui.ProgressBar()
            self._progress.value = 0.0
            self._layout.add_child(self._progress)

        def set_text(self, text):
            self._label.text = text + "                    "

        def post_update(self, text=None):
            if text is None:
                gui.Application.instance.post_to_main_thread(
                    self._window, self.update)
            else:

                def update_with_text():
                    self.update()
                    self._label.text = text

                gui.Application.instance.post_to_main_thread(
                    self._window, update_with_text)

        def update(self):
            value = min(1.0, self._progress.value + 1.0 / self._n_items)
            self._progress.value = value

    SOLID_NAME = "Solid Color"
    LABELS_NAME = "Label Colormap"
    RAINBOW_NAME = "Colormap (Rainbow)"
    GREYSCALE_NAME = "Colormap (Greyscale)"
    COLOR_NAME = "RGB"

    X_ATTR_NAME = "x position"
    Y_ATTR_NAME = "y position"
    Z_ATTR_NAME = "z position"

    def __init__(self):
        self._objects = None

        self._name2treenode = {}
        self._name2treeid = {}
        self._treeid2name = {}
        self._attrname2lut = {}
        self._colormaps = {}
        self._shadername2panelidx = {}
        self._gradient = rendering.Gradient()
        self._scalar_min = 0.0
        self._scalar_max = 1.0
        self._animation_frames = []
        self._last_animation_time = time.time()
        self._animation_delay_secs = 0.100
        self._dont_update_geometry = False

    def _init_dataset(self, dataset, split, indices):
        self._objects = DatasetModel(dataset, split, indices)

    def _init_data(self, data):
        self._objects = DataModel(data)

    def _init_user_interface(self, title, width, height):
        self.window = gui.Application.instance.create_window(
            title, width, height)
        self.window.set_on_layout(self._on_layout)

        em = self.window.theme.font_size

        self._3d = gui.SceneWidget()
        self._3d.enable_scene_caching(True)  # makes UI _much_ more responsive
        self._3d.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self._3d)

        self._panel = gui.Vert()
        self.window.add_child(self._panel)

        indented_margins = gui.Margins(em, 0, em, 0)

        # View controls
        ctrl = gui.CollapsableVert("Mouse Controls", 0, indented_margins)

        arcball = gui.Button("Arcball")
        arcball.set_on_clicked(self._on_arcball_mode)
        arcball.horizontal_padding_em = 0.5
        arcball.vertical_padding_em = 0
        fly = gui.Button("Fly")
        fly.set_on_clicked(self._on_fly_mode)
        fly.horizontal_padding_em = 0.5
        fly.vertical_padding_em = 0
        reset = gui.Button("Re-center")
        reset.set_on_clicked(self._on_reset_camera)
        reset.horizontal_padding_em = 0.5
        reset.vertical_padding_em = 0
        h = gui.Horiz(0.25 * em)
        h.add_stretch()
        h.add_child(arcball)
        h.add_child(fly)
        h.add_fixed(em)
        h.add_child(reset)
        h.add_stretch()
        ctrl.add_child(h)

        ctrl.add_fixed(em)
        self._panel.add_child(ctrl)

        # Dataset
        model = gui.CollapsableVert("Dataset", 0, indented_margins)

        vgrid = gui.VGrid(2, 0.25 * em)
        model.add_child(vgrid)
        model.add_fixed(0.5 * em)

        bgcolor = gui.ColorEdit()
        bgcolor.color_value = gui.Color(1, 1, 1)
        self._on_bgcolor_changed(bgcolor.color_value)
        bgcolor.set_on_value_changed(self._on_bgcolor_changed)
        vgrid.add_child(gui.Label("BG Color"))
        vgrid.add_child(bgcolor)

        view_tab = gui.TabControl()
        view_tab.set_on_selected_tab_changed(self._on_display_tab_changed)
        model.add_child(view_tab)

        # ... model list
        self._dataset = gui.TreeView()
        self._dataset.set_on_selection_changed(
            self._on_dataset_selection_changed)
        view_tab.add_tab("List", self._dataset)

        # ... animation slider
        v = gui.Vert()
        view_tab.add_tab("Animation", v)
        v.add_fixed(0.25 * em)
        grid = gui.VGrid(2)
        v.add_child(grid)

        self._slider = gui.Slider(gui.Slider.INT)
        self._slider.set_limits(0, len(self._objects.data_names))
        self._slider.set_on_value_changed(self._on_animation_slider_changed)
        grid.add_child(gui.Label("Index"))
        grid.add_child(self._slider)

        self._slider_current = gui.Label("")
        grid.add_child(gui.Label("Showing"))
        grid.add_child(self._slider_current)

        v.add_fixed(em)

        self._play = gui.Button("Play")
        self._play.horizontal_padding_em = 0.5
        self._play.vertical_padding_em = 0
        self._play.set_on_clicked(self._on_start_animation)
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(self._play)
        h.add_stretch()
        v.add_child(h)

        self._panel.add_child(model)

        # Coloring
        properties = gui.CollapsableVert("Properties", 0, indented_margins)

        grid = gui.VGrid(2, 0.25 * em)

        # ... data source
        self._datasource_combobox = gui.Combobox()
        self._datasource_combobox.set_on_selection_changed(
            self._on_datasource_changed)
        self._colormap_channel = gui.Combobox()
        self._colormap_channel.add_item("0")
        self._colormap_channel.set_on_selection_changed(
            self._on_channel_changed)
        h = gui.Horiz()
        h.add_child(self._datasource_combobox)
        h.add_fixed(em)
        h.add_child(gui.Label("Index"))
        h.add_child(self._colormap_channel)

        grid.add_child(gui.Label("Data"))
        grid.add_child(h)

        # ... shader
        self._shader = gui.Combobox()
        self._shader.add_item(self.SOLID_NAME)
        self._shader.add_item(self.LABELS_NAME)
        self._shader.add_item(self.RAINBOW_NAME)
        self._shader.add_item(self.GREYSCALE_NAME)
        self._shader.add_item(self.COLOR_NAME)
        self._colormaps[self.RAINBOW_NAME] = Colormap.make_rainbow()
        self._colormaps[self.GREYSCALE_NAME] = Colormap.make_greyscale()
        self._shader.selected_index = 0
        self._shader.set_on_selection_changed(self._on_shader_changed)
        grid.add_child(gui.Label("Shader"))
        grid.add_child(self._shader)

        properties.add_child(grid)

        # ... shader panels
        self._shader_panels = gui.StackedWidget()
        panel_idx = 0

        #     ... sub-panel: single color
        self._color_panel = gui.Vert()
        self._shader_panels.add_child(self._color_panel)
        self._shadername2panelidx[self.SOLID_NAME] = panel_idx
        panel_idx += 1
        self._color = gui.ColorEdit()
        self._color.color_value = gui.Color(0.5, 0.5, 0.5)
        self._color.set_on_value_changed(self._on_shader_color_changed)
        h = gui.Horiz()
        h.add_child(gui.Label("Color"))
        h.add_child(self._color)
        self._color_panel.add_child(h)

        #     ... sub-panel: labels
        self._labels_panel = gui.Vert()
        self._shader_panels.add_child(self._labels_panel)
        self._shadername2panelidx[self.LABELS_NAME] = panel_idx
        panel_idx += 1
        self._label_edit = self.LabelLUTEdit()
        self._label_edit.set_on_changed(self._on_labels_changed)
        self._labels_panel.add_child(gui.Label("Labels"))
        self._labels_panel.add_child(self._label_edit.widget)

        #     ... sub-panel: colormap
        self._colormap_panel = gui.Vert()
        self._shader_panels.add_child(self._colormap_panel)
        self._shadername2panelidx[self.RAINBOW_NAME] = panel_idx
        self._shadername2panelidx[self.GREYSCALE_NAME] = panel_idx
        panel_idx += 1
        self._colormap_edit = self.ColormapEdit(self.window, em)
        self._colormap_edit.set_on_changed(self._on_colormap_changed)
        self._colormap_panel.add_child(self._colormap_edit.widget)

        #     ... sub-panel: RGB
        self._rgb_panel = gui.Vert()
        self._shader_panels.add_child(self._rgb_panel)
        self._shadername2panelidx[self.COLOR_NAME] = panel_idx
        panel_idx += 1
        self._rgb_combo = gui.Combobox()
        self._rgb_combo.add_item("255")
        self._rgb_combo.add_item("1.0")
        self._rgb_combo.set_on_selection_changed(self._on_rgb_multiplier)
        h = gui.Horiz(0.5 * em)
        h.add_child(gui.Label("Max value"))
        h.add_child(self._rgb_combo)
        self._rgb_panel.add_child(h)

        properties.add_fixed(em)
        properties.add_child(self._shader_panels)
        self._panel.add_child(properties)

        # Populate tree, etc.
        for name in self._objects.data_names:
            self._add_tree_name(name)
        self._update_datasource_combobox()

    def set_lut(self, attr_name, lut):
        """Sets the LUT for a specific attribute
        
        Args:
            attr_name: Attribute name as string.
            lut: A LabelLUT object.
        """
        self._attrname2lut[attr_name] = lut

    def setup_camera(self):
        selected_names = self._get_selected_names()
        selected_bounds = [
            self._objects.calc_bounds_for(n) for n in selected_names
        ]
        min_val = [1e30, 1e30, 1e30]
        max_val = [-1e30, -1e30, -1e30]
        for b in selected_bounds:
            for i in range(0, 3):
                min_val[i] = min(min_val[i], b[0][i])
                max_val[i] = max(max_val[i], b[1][i])
        bounds = o3d.geometry.AxisAlignedBoundingBox(min_val, max_val)
        self._3d.setup_camera(60, bounds, bounds.get_center())

    def show_geometries_under(self, name, show):
        prefix = name
        for (n, node) in self._name2treenode.items():
            if n.startswith(prefix):
                self._3d.scene.show_geometry(n, show)
                node.checkbox.checked = show
        self._3d.force_redraw()

    def _add_tree_name(self, name):
        names = name.split("/")
        parent = self._dataset.get_root_item()
        for i in range(0, len(names) - 1):
            n = "/".join(names[:i + 1]) + "/"
            if n in self._name2treeid:
                parent = self._name2treeid[n]
            else:

                def on_parent_checked(checked):
                    self.show_geometries_under(n, checked)

                cell = gui.CheckableTextTreeCell(n, True, on_parent_checked)
                parent = self._dataset.add_item(parent, cell)
                self._name2treenode[n] = cell
                self._name2treeid[n] = parent
                self._treeid2name[parent] = n

        def on_checked(checked):
            self._3d.scene.show_geometry(name, checked)
            self._update_datasource_combobox()  # available attrs could change
            self._3d.force_redraw()

        cell = gui.CheckableTextTreeCell(names[-1], True, on_checked)
        cell.label.text_color = gui.Color(1.0, 0.0, 0.0, 1.0)
        node = self._dataset.add_item(parent, cell)
        self._name2treenode[name] = cell
        self._treeid2name[node] = name

        self._slider.set_limits(0, len(self._objects.data_names) - 1)
        if len(self._objects.data_names) == 1:
            self._slider_current.text = name

    def _load_geometry(self, name, ui_done_callback):
        progress_dlg = Visualizer.ProgressDialog("Loading...", self.window, 2)
        progress_dlg.set_text("Loading " + name + "...")

        def load_thread():
            result = self._objects.load(name)
            progress_dlg.post_update("Loading " + name + "...")

            gui.Application.instance.post_to_main_thread(
                self.window, ui_done_callback)
            gui.Application.instance.post_to_main_thread(
                self.window, self.window.close_dialog)

        self.window.show_dialog(progress_dlg.dialog)
        threading.Thread(target=load_thread).start()

    def _load_geometries(self, names, ui_done_callback):
        # Progress has: len(names) items + ui_done_callback
        progress_dlg = Visualizer.ProgressDialog("Loading...", self.window,
                                                 len(names) + 1)
        progress_dlg.set_text("Loading " + names[0] + "...")

        def load_thread():
            for i in range(0, len(names)):
                result = self._objects.load(names[i], True)
                if i + 1 < len(names):
                    text = "Loading " + names[i + 1] + "..."
                else:
                    text = "Creating GPU objects..."
                progress_dlg.post_update(text)
                if result:
                    self._name2treenode[names[i]].label.text_color = gui.Color(
                        0.0, 1.0, 0.0, 1.0)
                else:
                    break

            gui.Application.instance.post_to_main_thread(
                self.window, ui_done_callback)
            gui.Application.instance.post_to_main_thread(
                self.window, self.window.close_dialog)

        self.window.show_dialog(progress_dlg.dialog)
        threading.Thread(target=load_thread).start()

    def _update_geometry(self, check_unloaded=False):
        if check_unloaded:
            for name in self._objects.data_names:
                if not self._objects.is_loaded(name):
                    self._3d.scene.remove_geometry(name)

        material = self._get_material()
        for n, tcloud in self._objects.tclouds.items():
            self._update_point_cloud(n, tcloud, material)
            if not tcloud.is_empty():
                self._name2treenode[n].label.text_color = gui.Color(
                    0.0, 1.0, 0.0, 1.0)
            else:
                self._name2treenode[n].label.text_color = gui.Color(
                    1.0, 0.0, 0.0, 1.0)
                self._name2treenode[n].checkbox.checked = False
        self._3d.scene.update_material(material)
        self._3d.force_redraw()

    def _update_point_cloud(self, name, tcloud, material):
        if self._dont_update_geometry:
            return

        if tcloud.is_empty():
            return

        attr_name = self._datasource_combobox.selected_text
        attr = None
        flag = 0
        attr = self._objects.get_attr(name, attr_name)

        # Update scalar values
        if attr is not None:
            if len(attr.shape) == 1:
                scalar = attr
            else:
                channel = max(0, self._colormap_channel.selected_index)
                scalar = attr[:, channel]
        else:
            shape = [len(tcloud.point["points"].as_tensor().numpy())]
            scalar = np.zeros(shape, dtype='float32')
        tcloud.point["__visualization_scalar"] = Visualizer._make_tcloud_array(
            scalar)

        flag |= rendering.Scene.UPDATE_UV0_FLAG

        # Update RGB values
        if attr is not None and (len(attr.shape) == 2 and attr.shape[1] >= 3):
            max_val = float(self._rgb_combo.selected_text)
            if max_val <= 0:
                max_val = 255.0
            colors = attr[:, [0, 1, 2]] * (1.0 / max_val)
            tcloud.point["colors"] = Visualizer._make_tcloud_array(colors)
            flag |= rendering.Scene.UPDATE_COLORS_FLAG

        # Update geometry
        if self._3d.scene.scene.has_geometry(name):
            self._3d.scene.scene.update_geometry(name, tcloud, flag)
        else:
            self._3d.scene.add_geometry(name, tcloud, material)

        node = self._name2treenode[name]
        if node is not None:
            self._3d.scene.show_geometry(name, node.checkbox.checked)

    def _get_material(self):
        self._update_gradient()
        material = rendering.Material()
        if self._shader.selected_text == self.SOLID_NAME:
            material.shader = "unlitSolidColor"
            c = self._color.color_value
            material.base_color = [c.red, c.green, c.blue, 1.0]
        elif self._shader.selected_text == self.COLOR_NAME:
            material.shader = "defaultUnlit"
            material.base_color = [1.0, 1.0, 1.0, 1.0]
        else:
            material.shader = "unlitGradient"
            material.gradient = self._gradient
            material.scalar_min = self._scalar_min
            material.scalar_max = self._scalar_max

        return material

    def _update_gradient(self):
        if self._shader.selected_text == self.LABELS_NAME:
            colors = self._label_edit.get_colors()
            n = float(len(colors) - 1)
            if n >= 1:
                self._gradient.points = [
                    rendering.Gradient.Point(
                        float(i) / n, [
                            colors[i][0], colors[i][1], colors[i][2],
                            colors[i][3]
                        ]) for i in range(0, len(colors))
                ]
            else:
                self._gradient.points = [
                    rendering.Gradient.Point(0.0, [1.0, 0.0, 1.0, 1.0])
                ]
            self._gradient.mode = rendering.Gradient.LUT
        else:
            cmap = self._colormaps.get(self._shader.selected_text)
            if cmap is not None:
                self._gradient.points = [
                    rendering.Gradient.Point(
                        p.value, [p.color[0], p.color[1], p.color[2], 1.0])
                    for p in cmap.points
                ]
                self._gradient.mode = rendering.Gradient.GRADIENT

    def _update_geometry_colors(self):
        material = self._get_material()
        self._3d.scene.update_material(material)
        self._3d.force_redraw()

    def _update_datasource_combobox(self):
        current = self._datasource_combobox.selected_text
        self._datasource_combobox.clear_items()
        available_attrs = self._get_available_attrs()
        for attr_name in available_attrs:
            self._datasource_combobox.add_item(attr_name)
        if current in available_attrs:
            self._datasource_combobox.selected_text = current
        elif len(available_attrs) > 0:
            self._datasource_combobox.selected_text = available_attrs[0]
        else:
            # If no attributes, two possibilities:
            # 1) no geometries are selected: don't change anything
            # 2) geometries are selected: color solid
            has_checked = False
            for n, node in self._name2treenode.items():
                if node.checkbox.checked:
                    has_checked = True
                    break
            if has_checked:
                self._set_shader(self.SOLID_NAME)

    def _update_shaders_combobox(self):
        current_attr = self._datasource_combobox.selected_text
        current_shader = self._shader.selected_text
        has_lut = (current_attr in self._attrname2lut)
        is_scalar = True
        selected_names = self._get_selected_names()
        if len(selected_names) > 0 and len(
                self._objects.get_attr_shape(selected_names[0],
                                             current_attr)) > 1:
            is_scalar = False

        self._shader.clear_items()
        if not is_scalar:
            self._shader.add_item(self.COLOR_NAME)
        if has_lut:
            self._shader.add_item(self.LABELS_NAME)
            self._label_edit.set_labels(self._attrname2lut[current_attr])
        self._shader.add_item(self.RAINBOW_NAME)
        self._shader.add_item(self.GREYSCALE_NAME)
        self._shader.add_item(self.SOLID_NAME)

        if current_shader == self.LABELS_NAME and has_lut:
            self._set_shader(self.LABELS_NAME)
        elif is_scalar:
            self._set_shader(self.RAINBOW_NAME)

    def _update_attr_range(self):
        attr_name = self._datasource_combobox.selected_text
        current_channel = self._colormap_channel.selected_index
        self._scalar_min, self._scalar_max = self._objects.get_attr_minmax(
            attr_name, current_channel)

        if self._shader.selected_text in self._colormaps:
            cmap = self._colormaps[self._shader.selected_text]
            self._colormap_edit.update(cmap, self._scalar_min, self._scalar_max)

    def _set_shader(self, shader_name, force_update=False):
        # Disable channel if we are using a vector shader. Always do this to
        # ensure that the UI is consistent.
        if shader_name == Visualizer.COLOR_NAME:
            self._colormap_channel.enabled = False
        else:
            self._colormap_channel.enabled = True

        if shader_name == self._shader.selected_text and not force_update:
            return

        self._shader.selected_text = shader_name
        idx = self._shadername2panelidx[self._shader.selected_text]
        self._shader_panels.selected_index = idx

        if shader_name in self._colormaps:
            cmap = self._colormaps[shader_name]
            self._colormap_edit.update(cmap, self._scalar_min, self._scalar_max)

        self._update_geometry_colors()

    def _on_layout(self, theme):
        frame = self.window.content_rect
        em = theme.font_size
        panel_width = 20 * em
        panel_rect = gui.Rect(frame.get_right() - panel_width, frame.y,
                              panel_width, frame.height - frame.y)
        self._panel.frame = panel_rect
        self._3d.frame = gui.Rect(frame.x, frame.y, panel_rect.x - frame.x,
                                  frame.height - frame.y)

    def _on_arcball_mode(self):
        self._3d.set_view_controls(gui.SceneWidget.ROTATE_CAMERA)

    def _on_fly_mode(self):
        self._3d.set_view_controls(gui.SceneWidget.FLY)

    def _on_reset_camera(self):
        self.setup_camera()

    def _on_dataset_selection_changed(self, item):
        name = self._treeid2name[item]

        def ui_callback():
            self._update_attr_range()
            self._update_geometry(check_unloaded=True)

        if not self._objects.is_loaded(name):
            self._load_geometry(name, ui_callback)

    def _on_display_tab_changed(self, index):
        if index == 1:
            self._animation_frames = self._get_selected_names()
            self._slider.set_limits(0, len(self._animation_frames) - 1)
            self._on_animation_slider_changed(self._slider.int_value)
        else:
            for name, node in self._name2treenode.items():
                self._3d.scene.show_geometry(name, node.checkbox.checked)

    def _on_animation_slider_changed(self, new_value):
        idx = int(new_value)
        for i in range(0, len(self._animation_frames)):
            self._3d.scene.show_geometry(self._animation_frames[i], (i == idx))
            self._3d.force_redraw()
        self._slider_current.text = self._animation_frames[idx]
        r = self._slider_current.frame
        self._slider_current.frame = gui.Rect(r.x, r.y,
                                              self._slider.frame.get_right(),
                                              r.height)

    def _on_start_animation(self):

        def on_tick():
            return self._on_animate()

        self._play.text = "Stop"
        self._play.set_on_clicked(self._on_stop_animation)
        self._last_animation_time = 0.0
        self.window.set_on_tick_event(on_tick)

    def _on_animate(self):
        now = time.time()
        if now >= self._last_animation_time + self._animation_delay_secs:
            idx = (self._slider.int_value + 1) % len(self._animation_frames)
            self._slider.int_value = idx
            self._on_animation_slider_changed(idx)
            self._last_animation_time = now
            return True
        return False

    def _on_stop_animation(self):
        self.window.set_on_tick_event(None)
        self._play.text = "Play"
        self._play.set_on_clicked(self._on_start_animation)

    def _on_bgcolor_changed(self, new_color):
        bg_color = [
            new_color.red, new_color.green, new_color.blue, new_color.alpha
        ]
        self._3d.scene.set_background_color(bg_color)
        self._3d.force_redraw()

    def _on_datasource_changed(self, attr_name, idx):
        selected_names = self._get_selected_names()
        n_channels = 1
        if len(selected_names) > 0:
            shape = self._objects.get_attr_shape(selected_names[0], attr_name)
            if len(shape) <= 1:
                n_channels = 1
            else:
                n_channels = max(1, shape[1])
        current_channel = max(0, self._colormap_channel.selected_index)
        current_channel = min(n_channels - 1, current_channel)
        self._colormap_channel.clear_items()
        for i in range(0, n_channels):
            self._colormap_channel.add_item(str(i))
        self._colormap_channel.selected_index = current_channel

        self._update_attr_range()
        self._update_shaders_combobox()

        # Try to intelligently pick a shader.
        current_shader = self._shader.selected_text
        if current_shader == Visualizer.SOLID_NAME:
            pass
        elif attr_name in self._attrname2lut:
            self._set_shader(Visualizer.LABELS_NAME)
        elif attr_name == "colors":
            self._set_shader(Visualizer.COLOR_NAME)
        elif n_channels >= 3:
            self._set_shader(Visualizer.RAINBOW_NAME)
        elif current_shader == Visualizer.COLOR_NAME:  # vector -> scalar
            self._set_shader(Visualizer.RAINBOW_NAME)
        else:  # changing from one scalar to another, don't change
            pass

        self._update_geometry()

    def _on_channel_changed(self, name, idx):
        self._update_attr_range()
        self._update_geometry()  # need to recompute scalars array

    def _on_shader_changed(self, name, idx):
        # _shader.current_text is already name, so we need to force an update
        self._set_shader(name, force_update=True)

    def _on_shader_color_changed(self, color):
        self._update_geometry_colors()

    def _on_labels_changed(self):
        self._update_geometry_colors()

    def _on_colormap_changed(self):
        self._colormaps[
            self._shader.selected_text] = self._colormap_edit.colormap
        self._update_geometry_colors()

    def _on_rgb_multiplier(self, text, idx):
        self._update_geometry()

    def _get_selected_names(self):
        selected_names = []
        for n, node in self._name2treenode.items():
            if node.checkbox.checked:
                selected_names.append(n)
        return selected_names

    def _get_available_attrs(self):
        selected_names = self._get_selected_names()
        return self._objects.get_available_attrs(selected_names)

    @staticmethod
    def _make_tcloud_array(np_array, copy=False):
        if copy or not np_array.data.c_contiguous:
            t = o3d.core.Tensor(np_array)
        else:
            t = o3d.core.Tensor.from_numpy(np_array)
        return o3d.core.TensorList.from_tensor(t, inplace=True)

    def visualize_dataset(self,
                          dataset,
                          split,
                          indices=None,
                          width=1024,
                          height=768):
        """Visualizes a dataset

        Example:
            Minimal example for visualizing a dataset::
                import open3d.ml.torch as ml3d  # or open3d.ml.tf as ml3d

                dataset = ml3d.datasets.SemanticKITTI(dataset_path='/path/to/SemanticKITTI/')
                vis = ml3d.vis.Visualizer()
                vis.visualize_dataset(dataset, 'all', indices=range(100))

        Args:
            dataset: A dataset object.
            split: A string that identifies the split, e.g., 'test'.
            indices: An iterable with a subset of the data points to visualize.
                E.g., [0,2,3,4].
            width: window width.
            height: window height.
        """
        # Setup the labels
        lut = LabelLUT()
        for val in sorted(dataset.label_to_names.keys()):
            lut.add_label(dataset.label_to_names[val], val)
        self.set_lut("labels", lut)

        self._init_dataset(dataset, split, indices)
        self._visualize("Open3D - " + dataset.name, width, height)

    def visualize(self, data, width=1024, height=768):
        """Visualizes custom point cloud data

        Example:
            Minimal example for visualizing a single point cloud with an
            attribute::
                import numpy as np
                import open3d.ml.torch as ml3d
                # or import open3d.ml.tf as ml3d

                data = [ {
                    'name': 'my_point_cloud',
                    'points': np.random.rand(100,3).astype(np.float32),
                    'point_attr1': np.random.rand(100).astype(np.float32),
                    } ]

                vis = ml3d.vis.Visualizer()
                vis.visualize(data) 

        Args:
            data: A list of dictionaries. Each dictionary is a point cloud with
                attributes. Each dictionary must have the entries 'name' and
                'points'. Points and point attributes can be passed as numpy 
                arrays, PyTorch tensors or TensorFlow tensors.
            width: window width.
            height: window height.
        """
        self._init_data(data)
        self._visualize("Open3D", width, height)

    def _visualize(self, title, width, height):
        gui.Application.instance.initialize()
        self._init_user_interface(title, width, height)

        self._3d.scene.downsample_threshold = 400000

        # Turn all the objects off except the first one
        for name, node in self._name2treenode.items():
            node.checkbox.checked = False
            self._3d.scene.show_geometry(name, False)
        for name in [self._objects.data_names[0]]:
            self._name2treenode[name].checkbox.checked = True
            self._3d.scene.show_geometry(name, True)

        def on_done_ui():
            self._update_datasource_combobox()
            self._update_shaders_combobox()

            # Display "colors" by default if available, "points" if not
            available_attrs = self._get_available_attrs()
            self._set_shader(self.SOLID_NAME, force_update=True)
            if "colors" in available_attrs:
                self._datasource_combobox.selected_text = "colors"
            elif "points" in available_attrs:
                self._datasource_combobox.selected_text = "points"

            self._dont_update_geometry = True
            self._on_datasource_changed(
                self._datasource_combobox.selected_text,
                self._datasource_combobox.selected_index)
            self._update_geometry_colors()
            self._dont_update_geometry = False
            # _datasource_combobox was empty, now isn't, re-layout.
            self.window.set_needs_layout()

            self._update_geometry()
            self.setup_camera()

        self._load_geometries(self._objects.data_names, on_done_ui)
        gui.Application.instance.run()
