# Model Zoo

## Pretrained weights

For the task of semantic segmentation, we measure the performance of different methods using the mean intersection-over-union (mIoU) over all classes.
The table shows the available models and datasets for the segmentation task and the respective scores. Each score links to the respective weight file.


| Model / Dataset    | SemanticKITTI | Toronto 3D | S3DIS | Semantic3D | Paris-Lille3D |
|--------------------|---------------|----------- |-------|--------------|-------------|
| RandLA-Net (tf)    | [53.7](https://storage.googleapis.com/open3d-releases/model-zoo/randlanet_semantickitti_202010091306.zip) |  [69.0](https://storage.googleapis.com/open3d-releases/model-zoo/randlanet_toronto3d_202010091250.zip) |  [67.0](https://storage.googleapis.com/open3d-releases/model-zoo/randlanet_s3dis_202010091238.zip)    | [76.0](https://storage.googleapis.com/open3d-releases/model-zoo/randlanet_semantic3d_202012120312utc.zip) |  [70.0](https://storage.googleapis.com/open3d-releases/model-zoo/randlanet_parislille3d_202012160654utc.zip) |
| RandLA-Net (torch) | [52.8](https://storage.googleapis.com/open3d-releases/model-zoo/randlanet_semantickitti_202009090354utc.pth)        |     [71.2](https://storage.googleapis.com/open3d-releases/model-zoo/randlanet_toronto3D_202010091306.pth)  |  [67.0](https://storage.googleapis.com/open3d-releases/model-zoo/randlanet_s3dis_202010091238.pth)  | [76.0](https://storage.googleapis.com/open3d-releases/model-zoo/randlanet_semantic3d_202012120312utc.pth) |  [70.0](https://storage.googleapis.com/open3d-releases/model-zoo/randlanet_parislille3d_202012160654utc.pth) |
| KPConv     (tf)    | [58.7](https://storage.googleapis.com/open3d-releases/model-zoo/kpconv_semantickitti_202010021102utc.zip)         |     [65.6](https://storage.googleapis.com/open3d-releases/model-zoo/kpconv_toronto3d_202012221551utc.zip)  |  [65.0](https://storage.googleapis.com/open3d-releases/model-zoo/kpconv_s3dis_202010091238.zip) | - |  [76.7](https://storage.googleapis.com/open3d-releases/model-zoo/kpconv_parislille3d_202011241550utc.zip) |
| KPConv     (torch) | [58.0](https://storage.googleapis.com/open3d-releases/model-zoo/kpconv_semantickitti_202009090354utc.pth)          |     [65.6](https://storage.googleapis.com/open3d-releases/model-zoo/kpconv_toronto3d_202012221551utc.pth) |  [60.0](https://storage.googleapis.com/open3d-releases/model-zoo/kpconv_s3dis_202010091238.pth)  | - | [76.7](https://storage.googleapis.com/open3d-releases/model-zoo/kpconv_parislille3d_202011241550utc.pth) |

[md5 checksum file](https://storage.googleapis.com/open3d-releases/model-zoo/integrity.txt)


## Models
The followings are the models we implemented in this model zoo.
* KPConv ([github](https://github.com/HuguesTHOMAS/KPConv)): [KPConv: Flexible and Deformable Convolution for Point Clouds](https://arxiv.org/abs/1904.08889).
* RandLA-Net ([github](https://github.com/QingyongHu/RandLA-Net)) [RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds](https://arxiv.org/abs/1911.11236).

## Datasets

The following is a list of datasets for which we provide dataset reader classes.

* SemanticKITTI ([project page](http://semantic-kitti.org/))
* Toronto 3D ([github](https://github.com/WeikaiTan/Toronto-3D))
* Semantic 3D ([project-page](http://www.semantic3d.net/))
* S3DIS ([project-page](http://3dsemantics.stanford.edu/))
* Paris-Lille 3D ([project-page](https://npm3d.fr/paris-lille-3d))

For downloading these datasets visit the respective webpages and have a look at the scripts in [`scripts/download_datasets`](https://github.com/intel-isl/Open3D-ML/tree/master/scripts/download_datasets).

