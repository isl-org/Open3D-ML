class LabelLUT:

    class Label:

        def __init__(self, name, value, color):
            self.name = name
            self.value = value
            self.color = color

    Colors = [[1.0, 0.0, 0.0], [1.0, 0.5, 0.0], [1.0, 1.0,
                                                 0.0], [0.5, 1.0, 0.0],
              [0.0, 1.0, 0.0], [0.0, 1.0, 0.5], [0.0, 1.0,
                                                 1.0], [0.0, 0.5, 1.0],
              [0.0, 0.0, 1.0], [0.5, 0.0, 0.0], [0.5, 0.25, 0.0],
              [0.5, 0.5, 0.0], [0.25, 0.5, 0.0], [0.0, 0.5, 0.0],
              [0.0, 0.5, 0.25], [0.0, 0.5, 0.5], [0.0, 0.25, 0.5],
              [0.0, 0.0, 0.5], [1.0, 0.25, 0.25], [1.0, 0.5, 0.25],
              [1.0, 1.0, 0.25], [0.5, 1.0, 0.25], [0.25, 1.0, 0.25],
              [0.25, 1.0, 0.5], [0.25, 1.0, 1.25], [0.25, 0.5, 1.25],
              [0.25, 0.25, 1.0], [0.125, 0.125, 0.125], [0.25, 0.25, 0.25],
              [0.375, 0.375, 0.375], [0.5, 0.5, 0.5], [0.625, 0.625, 0.625],
              [0.75, 0.75, 0.75], [0.875, 0.875, 0.875]]

    def __init__(self):
        self._next_color = 0
        self.labels = {}

    def add_label(self, name, value, color=None):
        if color is None:
            if self._next_color >= len(self.Colors):
                color = [0.85, 1.0, 1.0]
            else:
                color = self.Colors[self._next_color]
                self._next_color += 1
        self.labels[value] = self.Label(name, value, color)
