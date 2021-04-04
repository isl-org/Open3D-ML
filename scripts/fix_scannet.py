from pathlib import Path
import traceback
import logging as log
import numpy as np
from tqdm import tqdm
# import open3d as o3d
from concurrent.futures import ThreadPoolExecutor

def compress_vertices(vertices_file):
    # print(vertices_file)
    point = np.load(vertices_file)
    np.savez_compressed(scannet_frames_out / (vertices_file.name[:-1] + 'z'), point=point)

scannet_frames = Path(
    "/mnt/beegfs/tier1/vcl-nfs-work/ssheorey/Open3D/Datasets/ScanNet-frames-cache/frames"
)
scannet_frames_out = Path(
    "/mnt/beegfs/tier1/vcl-nfs-work/ssheorey/Open3D/Datasets/ScanNet-frames-cache/frames_npz"
)
scannet_frames_out.mkdir(exist_ok=True)
# errors = []

with ThreadPoolExecutor(max_workers=64) as executor:
    for errs in tqdm(executor.map(compress_vertices, Path(scannet_frames).glob("*_vert.npy"))):
        pass
# for vert_file in tqdm(Path(scannet_frames).glob("*_vert.npy")):
#     try:
#         o3d_bboxes = np.load(bbox_file, allow_pickle=True)
#         np_bboxes = np.vstack(list(box.to_xyzwlhyc() for box in o3d_bboxes))
#         bbox_out_file = Path(scannet_frames_out) / bbox_file.name
#         np.save(bbox_out_file, np_bboxes)
#     except Exception:
#         errmsg = "\n".join(bbox_file, traceback.format_exc())
#         log.warning(errmsg)
#         errors.append(errmsg)
# if errors:
#     with open(scannet_frames_out / "errors.txt") as errfile:
#         errfile.write("\n".join(errors))
