class LabelLUT:

    class Label:

        def __init__(self, name, value, color):
            self.name = name
            self.value = value
            self.color = color

    Colors = [[0. ,        0. ,        0.       ],
              [0.96078431, 0.58823529 ,0.39215686],
              [0.96078431, 0.90196078 ,0.39215686],
              [0.58823529, 0.23529412 ,0.11764706],
              [0.70588235, 0.11764706 ,0.31372549],
              [1.        , 0.         ,0.        ],
              [0.11764706, 0.11764706 ,1.        ],
              [0.78431373, 0.15686275 ,1.        ],
              [0.35294118, 0.11764706 ,0.58823529],
              [1.        , 0.         ,1.        ],
              [1.        , 0.58823529 ,1.        ],
              [0.29411765, 0.         ,0.29411765],
              [0.29411765, 0.         ,0.68627451],
              [0.        , 0.78431373 ,1.        ],
              [0.19607843, 0.47058824 ,1.        ],
              [0.        , 0.68627451 ,0.        ],
              [0.        , 0.23529412 ,0.52941176],
              [0.31372549, 0.94117647 ,0.58823529],
              [0.58823529, 0.94117647 ,1.        ],
              [0.        , 0.         ,1.        ],
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
