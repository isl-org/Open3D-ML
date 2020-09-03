from ml3d.datasets import (SemanticKITTI, ParisLille3D, Semantic3D, 
                                S3DIS, Toronto3D)
from ml3d.torch.pipelines import SemanticSegmentation 
from ml3d.torch.models import RandLANet
from ml3d.utils import Config, get_module

def demo_dataset():
    # read data from datasets

    # dataset = SemanticKITTI(dataset_path="../dataset/SemanticKITTI",
    #                         use_cahe=False)
    datasets = []
    # datasets.append(ParisLille3D(dataset_path="../dataset/Paris_Lille3D",
    #                         use_cahe=False))
    # datasets.append(Toronto3D(dataset_path="../dataset/Toronto_3D",
    #                         use_cahe=False))
    # datasets.append(S3DIS(dataset_path="../dataset/S3DIS",
    #                         use_cahe=False))
    datasets.append(Semantic3D(dataset_path="../dataset/Semantic3D",
                            use_cahe=False))

    for dataset in datasets:
        print(dataset.label_to_names)

        # print names of all pointcould
        split = dataset.get_split('test')
        for i in range(len(split)):
            attr = split.get_attr(i)
            print(attr['name'])

        split = dataset.get_split('train')
        for i in range(len(split)):
            data = split.get_data(i)
            print(data['point'].shape)

if __name__ == '__main__':
    demo_dataset()