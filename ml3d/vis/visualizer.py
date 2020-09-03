import math  # debugging; remove
import numpy as np
import open3d as o3d
from open3d.visualization import gui
from open3d.visualization import rendering
from .colormap import *
from .labellut import *

import time

class Visualizer:
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
            return [self._label2color[label]
                    for label in sorted(self._label2color.keys())]

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
                cell = gui.LUTTreeCell(str(key) + ": " + lbl.name, True,
                                       color, None, None)
                cell.checkbox.set_on_checked(self._make_on_checked(key, self._on_label_checked))
                cell.color_edit.set_on_value_changed(self._make_on_color_changed(key, self._on_label_color_changed))
                self.widget.add_item(root, cell)

        def _make_on_color_changed(self, label, member_func):
            def on_changed(color):
                member_func(label, color)
            return on_changed
            
        def _on_label_color_changed(self, label, gui_color):
            self._label2color[label] = [gui_color.red,
                                        gui_color.green,
                                        gui_color.blue,
                                        self._label2color[label][3]]
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
                cell.color_edit.set_on_value_changed(self._make_on_color_changed(i, self._on_color_changed))
                cell.number_edit.set_on_value_changed(self._make_on_value_changed(i, self._on_value_changed))
                item_id = self._edit.add_item(root_id, cell)
                self._itemid2idx[item_id] = i
            self._update_buttons_enabled()

        def _make_on_color_changed(self, idx, member_func):
            def on_changed(color):
                member_func(idx, color)
            return on_changed
            
        def _on_color_changed(self, idx, gui_color):
            self.colormap.points[idx].color = [gui_color.red, gui_color.green,
                                               gui_color.blue]
            if self._on_changed is not None:
                self._on_changed()

        def _make_on_value_changed(self, idx, member_func):
            def on_changed(value):
                member_func(idx, value)
            return on_changed

        def _on_value_changed(self, idx, value):
            value = (value - self._min_value) / (self._max_value - self._min_value)
            needs_update = False
            value = min(1.0, max(0.0, value))

            if ((idx > 0 and value < self.colormap.points[idx-1].value) or
                (idx < len(self.colormap.points) - 1 and value > self.colormap.points[idx+1].value)):
                self.colormap.points[idx].value = value
                o = self.colormap.points[idx]
                self.colormap.points.sort(key=lambda cmap_pt: cmap_pt.value)
                for i in range(0, len(self.colormap.points)):
                    if self.colormap.points[i] is o:
                        idx = i
                        break
                needs_update = True
            if idx > 0 and value == self.colormap.points[idx-1].value:
                if idx < len(self.colormap.points):
                    upper = self.colormap.points[idx + 1].value
                else:
                    upper = 1.0
                value = value + 0.5 * (upper - value)
                needs_update = True
            if idx < len(self.colormap.points) - 1 and value == self.colormap.points[idx+1].value:
                if idx > 0:
                    lower = self.colormap.points[idx - 1].value
                else:
                    lower  = 0.0
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
                self.colormap.points = self.colormap.points[:idx] + self.colormap.points[idx+1:]
                del self._itemid2idx[self._edit.selected_item]
                self._update_later()
                if self._on_changed is not None:
                    self._on_changed()

        def _on_add(self):
            if self._edit.selected_item in self._itemid2idx: # maybe no selection
                idx = self._itemid2idx[self._edit.selected_item]
                if idx < len(self.colormap.points) - 1:
                    lower = self.colormap.points[idx]
                    upper = self.colormap.points[idx + 1]
                else:
                    lower = self.colormap.points[len(self.colormap.points) - 2]
                    upper = self.colormap.points[len(self.colormap.points) - 1]
                add_idx = min(idx + 1, len(self.colormap.points) - 1)
                new_value = lower.value + 0.5 * (upper.value - lower.value)
                new_color = [0.5 * lower.color[0] + 0.5 * upper.color[0],
                             0.5 * lower.color[1] + 0.5 * upper.color[1],
                             0.5 * lower.color[2] + 0.5 * upper.color[2]]
                new_point = Colormap.Point(new_value, new_color)
                self.colormap.points = self.colormap.points[:add_idx] + [new_point] + self.colormap.points[add_idx:]
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
                self._window.post_redraw() # need to manually request redraw

            gui.Application.instance.post_to_main_thread(self._window, update)

    SOLID_NAME = "Solid Color"
    LABELS_NAME = "Labels"
    RAINBOW_NAME = "Colormap (Rainbow)"
    GREYSCALE_NAME = "Colormap (Greyscale)"

    def __init__(self):
        self._data = {}
        self._data_names = []  # sets the order data will be displayed / animated
        self._known_attrs = set()
        self._name2treenode = {}
        self._name2treeid = {}
        self._colormaps = {}
        self._gradient = rendering.Gradient()
        self._attr2minmax = {}  # should only be accessed by _get_attr_minmax()
        self._scalar_min = 0.0
        self._scalar_max = 1.0
        self._last_animation_time = time.time()
        self._animation_delay_secs = 0.100
        self._dont_update_geometry = True
        self.window = gui.Window("ml3d.vis.Visualizer", 1024, 768)
        self.window.set_on_layout(self._on_layout)

        em = self.window.theme.font_size

        self._3d = gui.SceneWidget()
        self._3d.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self._3d)

        self._panel = gui.Vert()
        self.window.add_child(self._panel)

        indented_margins = gui.Margins(em, 0, em, 0)

        # View controls
        ctrl = gui.CollapsableVert("Mouse Controls", 0, indented_margins)

        arcball = gui.Button("Arcball")
        arcball.horizontal_padding_em = 0.5
        arcball.vertical_padding_em = 0
        fly = gui.Button("Fly")
        fly.horizontal_padding_em = 0.5
        fly.vertical_padding_em = 0
        reset = gui.Button("Reset")
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

        bgcolor = gui.ColorEdit()
        bgcolor.color_value = gui.Color(1, 1, 1)
        self._on_bgcolor_changed(bgcolor.color_value)
        bgcolor.set_on_value_changed(self._on_bgcolor_changed)
        h = gui.Horiz(em)
        h.add_child(gui.Label("BG Color"))
        h.add_child(bgcolor)
        model.add_child(h)
        model.add_fixed(0.5 * em)

        view_tab = gui.TabControl()
        view_tab.set_on_selected_tab_changed(self._on_display_tab_changed)
        model.add_child(view_tab)

        # ... model list
        self._dataset = gui.TreeView()
        view_tab.add_tab("List", self._dataset)

        # ... animation slider
        v = gui.Vert()
        view_tab.add_tab("Animation", v)
        v.add_fixed(0.25 * em)
        grid = gui.VGrid(2)
        v.add_child(grid)

        self._slider = gui.Slider(gui.Slider.INT)
        self._slider.set_limits(0, len(self._data))
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
        self._datasource_combobox.set_on_selection_changed(self._on_datasource_changed)
        grid.add_child(gui.Label("Source data"))
        grid.add_child(self._datasource_combobox)

        # ... shader
        self._shader = gui.Combobox()
        self._shader.add_item(self.SOLID_NAME)
        self._shader.add_item(self.LABELS_NAME)
        self._shader.add_item(self.RAINBOW_NAME)
        self._shader.add_item(self.GREYSCALE_NAME)
        self._colormaps[self.RAINBOW_NAME] = Colormap.make_rainbow()
        self._colormaps[self.GREYSCALE_NAME] = Colormap.make_greyscale()
        self._shader.selected_index = 0
        self._shader.set_on_selection_changed(self._on_shader_changed)
        grid.add_child(gui.Label("Shader"))
        grid.add_child(self._shader)

        properties.add_child(grid)

        # ... shader panels
        self._shader_panels = gui.StackedWidget()

        #     ... sub-panel: single color
        self._color_panel = gui.Vert()
        self._shader_panels.add_child(self._color_panel)
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
        self._label_edit = self.LabelLUTEdit()
        self._label_edit.set_on_changed(self._on_labels_changed)
        self._labels_panel.add_child(gui.Label("Labels"))
        self._labels_panel.add_child(self._label_edit.widget)

        #     ... sub-panel: colormap
        self._colormap_panel = gui.Vert()
        self._shader_panels.add_child(self._colormap_panel)
        self._colormap_edit = self.ColormapEdit(self.window, em)
        self._colormap_edit.set_on_changed(self._on_colormap_changed)
        self._colormap_panel.add_child(self._colormap_edit.widget)

        properties.add_fixed(em)
        properties.add_child(self._shader_panels)
        self._panel.add_child(properties)

    def set_labels(self, labellut):
        self._label_edit.set_labels(labellut)

    def clear():
        self._data = {}
        self._data_names = []
        self._name2treenode = {}
        self._name2treeid = {}
        self._dataset.clear()
        self._attr2minmax = {}
        self._label_edit.clear()

    def add(self, name, cloud):
        self._data[name] = cloud
        self._data_names.append(name)

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

        def on_checked(checked):
            self._3d.scene.show_geometry(name, checked)

        cell = gui.CheckableTextTreeCell(names[-1], True, on_checked)
        node = self._dataset.add_item(parent, cell)
        self._name2treenode[name] = cell

        for attr_name,_ in cloud.point.items():
            if attr_name == "points":
                continue
            if attr_name not in self._known_attrs:
                self._datasource_combobox.add_item(attr_name)
                self._known_attrs.add(attr_name)

        self._slider.set_limits(0, len(self._data) - 1)
        if len(self._data) == 1:
            self._slider_current.text = name

    def setup_camera(self):
        bounds = self._3d.scene.bounding_box
        self._3d.setup_camera(60, bounds, bounds.get_center())

        old_dont_update = self._dont_update_geometry
        self._dont_update_geometry = True
        self._on_datasource_changed(self._datasource_combobox.selected_text,
                                    self._datasource_combobox.selected_index)
        self._update_geometry_colors()
        self._dont_update_geometry = old_dont_update

    def show_geometries_under(self, name, show):
        prefix = name
        for (n,node) in self._name2treenode.items():
            if n.startswith(prefix):
                self._3d.scene.show_geometry(n, show)
                node.checkbox.checked = show

    def _get_attr_minmax(self, attr_name):
        if attr_name not in self._attr2minmax:
            attr_min = 1e30
            attr_max = -1e30
            for _,tcloud in self._data.items():
                if attr_name in tcloud.point:
                    attr = tcloud.point[attr_name].as_tensor().numpy()
                    attr_min = min(attr_min, min(attr))
                    attr_max = max(attr_max, max(attr))
                    maxattr = max(attr)
            self._attr2minmax[attr_name] = (attr_min, attr_max)
        return self._attr2minmax[attr_name]

    def _update_geometry(self):
        material = self._get_material()
        for n,tcloud in self._data.items():
            self._update_point_cloud(n, tcloud, material)
        
    def _update_point_cloud(self, name, tcloud, material):
        if self._dont_update_geometry:
            return

        self._3d.scene.remove_geometry(name)
        attr_name = self._datasource_combobox.selected_text
        if attr_name in tcloud.point:
            attr = tcloud.point[attr_name].as_tensor().numpy()
            uv = np.column_stack((attr, [0.0] * len(attr)))
        else:
            uv = [[0.0, 0.0]] * len(tcloud.point["points"].as_tensor().numpy())
        tcloud.point["uv"] = o3d.core.TensorList.from_tensor(o3d.core.Tensor(np.array(uv, dtype='float32')), inplace=True)

        self._3d.scene.add_geometry(name, tcloud, material)
        node = self._name2treenode[name]
        if node is not None:
            self._3d.scene.show_geometry(name, node.checkbox.checked)

    def _get_material(self):
        self._update_gradient()
        material = rendering.Material()
        if self._shader.selected_text == self.SOLID_NAME:
            material.shader = "defaultUnlit"
            c = self._color.color_value
            material.base_color = [c.red, c.green, c.blue, 1.0]
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
                self._gradient.points = [rendering.Gradient.Point(
                                         float(i) / n,
                                         [colors[i][0], colors[i][1],
                                          colors[i][2], colors[i][3]])
                                     for i in range(0, len(colors))]
            else:
                self._gradient.points = [rendering.Gradient.Point(0.0, [1.0, 0.0, 1.0, 1.0])]
            self._gradient.mode = rendering.Gradient.LUT
        else:
            cmap = self._colormaps.get(self._shader.selected_text)
            if cmap is not None:
                self._gradient.points = [rendering.Gradient.Point(p.value,
                                                          [p.color[0],
                                                           p.color[1],
                                                           p.color[2],
                                                           1.0])
                                         for p in cmap.points]
                self._gradient.mode = rendering.Gradient.GRADIENT

    def _update_geometry_colors(self):
        material = self._get_material()
        self._3d.scene.update_material(material)

    def _on_layout(self, theme):
        frame = self.window.content_rect
        em = theme.font_size
        panel_width = 20 * em
        panel_rect = gui.Rect(frame.get_right() - panel_width, frame.y,
                              panel_width, frame.height - frame.y)
        self._panel.frame = panel_rect
        self._3d.frame = gui.Rect(frame.x, frame.y,
                                  panel_rect.x - frame.x, frame.height - frame.y)

    def _on_display_tab_changed(self, index):
        if index == 1:
            self._on_animation_slider_changed(self._slider.int_value)
        else:
            for name,node in self._name2treenode.items():
                self._3d.scene.show_geometry(name, node.checkbox.checked)

    def _on_animation_slider_changed(self, new_value):
        idx = int(new_value)
        for i in range(0, len(self._data_names)):
            self._3d.scene.show_geometry(self._data_names[i], (i == idx))
        self._slider_current.text = self._data_names[idx]
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
            idx = (self._slider.int_value + 1) % (len(self._data_names) - 1)
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
        self._3d.set_background_color(new_color)

    def _on_datasource_changed(self, attr_name, idx):
        self._scalar_min, self._scalar_max = self._get_attr_minmax(attr_name)

        if self._shader.selected_text in self._colormaps:
            cmap = self._colormaps[self._shader.selected_text]
            self._colormap_edit.update(cmap, self._scalar_min, self._scalar_max)

        self._update_geometry()

    def _on_shader_changed(self, name, idx):
        # Last items are all colormaps, so just clamp to n_children - 1
        idx = min(idx, len(self._shader_panels.get_children()) - 1)
        self._shader_panels.selected_index = idx

        if name in self._colormaps:
            cmap = self._colormaps[name]
            self._colormap_edit.update(cmap, self._scalar_min, self._scalar_max)

        self._update_geometry_colors()

    def _on_shader_color_changed(self, color):
        self._update_geometry_colors()
            
    def _on_labels_changed(self):
        self._update_geometry_colors()

    def _on_colormap_changed(self):
        self._colormaps[self._shader.selected_text] = self._colormap_edit.colormap
        self._update_geometry_colors()

    @staticmethod
    def _make_tcloud_array(np_array, copy=False):
        if copy or not np_array.data.c_contiguous:
            t = o3d.core.Tensor(np_array)
        else:
            t = o3d.core.Tensor.from_numpy(np_array)
        return o3d.core.TensorList.from_tensor(t, inplace=True)

    @staticmethod
    def visualize(dataset, idx_or_list):
        gui.Application.instance.initialize()

        mlvis = Visualizer()
        mlvis.window.title = dataset.dataset.name

        # Setup the labels
        lut = LabelLUT()
        for val in sorted(dataset.dataset.label_to_names.keys()):
            lut.add_label(dataset.dataset.label_to_names[val], val)
        mlvis.set_labels(lut)

        # We don't want to be recreating the geometry all the time, so defer
        # it until we are ready.
        mlvis._dont_update_geometry = True

        # Add the requested data
        indices = idx_or_list
        if not isinstance(idx_or_list, list):
            indices = [idx_or_list]
        for i in indices:
            info = dataset.get_attr(i)
            data = dataset.get_data(i)

            # Create tpointcloud
            pts = data["point"]
            tcloud = o3d.tgeometry.PointCloud(o3d.core.Dtype.Float32,
                                              o3d.core.Device("CPU:0"))
            if pts.shape[1] == 4:
                # We can't use inplace Tensor creation (e.g. from_numpy())
                # because the resulting arrays won't be contiguous. However,
                # TensorList can be inplace.
                xyz = pts[:,[0,1,2]]
                tcloud.point["intensity"] = Visualizer._make_tcloud_array(pts[:,3], copy=True)
                tcloud.point["points"] = Visualizer._make_tcloud_array(xyz, copy=True)
            else:
                tcloud.point["points"] = Visualizer._make_tcloud_array(pts)
            # Only add scalar attributes for now
            for k,v in data.items():
                if len(v.shape) == 1 or (len(v.shape) == 2 and v.shape[1] == 1):
                    isint = v.dtype.name.startswith('int')
                    tcloud.point[k] = Visualizer._make_tcloud_array(v, copy=isint)

            # ---- Debugging ----
            # dist = [math.sqrt(pt[0]*pt[0]+pt[1]*pt[1]+pt[2]*pt[2]) for pt in tcloud.point["points"].as_tensor().numpy()]
            # dist = np.array(dist, dtype='float32')
            # tcloud.point["distance"] = Visualizer._make_tcloud_array(dist, copy=True)
            # ----

            mlvis.add(info["name"], tcloud)

        # Display labels by default, if available
        for attr_name in ["label", "labels"]:
            if attr_name in mlvis._known_attrs:
                mlvis._datasource_combobox.selected_text = attr_name
                if not mlvis._label_edit.is_empty():
                    mlvis._shader.selected_text = "Labels"
                    mlvis._on_shader_changed(mlvis._shader.selected_text,
                                             mlvis._shader.selected_index)
                break

        # Turn all the objects off except the first one
        for name,node in mlvis._name2treenode.items():
            node.checkbox.checked = False
            mlvis._3d.scene.show_geometry(name, False)
        for name in [mlvis._data_names[0]]:
            mlvis._name2treenode[name].checkbox.checked = True
            mlvis._3d.scene.show_geometry(name, True)

        # Ok, now we can create our geometry
        mlvis._dont_update_geometry = False
        mlvis._update_geometry()

        mlvis.setup_camera()

        gui.Application.instance.add_window(mlvis.window)
        gui.Application.instance.run()
