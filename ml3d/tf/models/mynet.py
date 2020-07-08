"""Dummy network with just 1 convolution"""
# use absolute imports for using code defined in the main open3d repo
import open3d.ml.tf as ml3d
import tensorflow as tf

def mynet(num_classes, radius, pretrained=False):
    model = MyNet(num_classes, radius)

    if pretrained:
        # this makes probably only sense for the pipeline
        # and the weight loading code could go there as well
        pass

    return model


class MyNet(tf.keras.Model):

    def __init__(self, num_classes, radius, name=None):
        super().__init__(name=name)
        self.num_classes = num_classes
        self.radius = radius

        self.conv = ml3d.layers.ContinuousConv(name='conv1', filters=num_classes, kernel_size=[3,3,3])
        

    def call(self, feats, points):

        radius = self.radius

        ans = self.conv(feats, points, points, extents=2*radius)
        return ans

