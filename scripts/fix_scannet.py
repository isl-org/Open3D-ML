from pathlib import Path
import traceback
import logging as log
import sys
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
from tqdm import tqdm

# import open3d as o3d
from open3d.ml.datasets import utils
BEVBox3D = utils.bev_box.BEVBox3D

scannet_frames = Path(
    # "/Users/ssheorey/Documents/Open3D/Data/ScanNet/npy-test/frames_npz"
    "/mnt/beegfs/tier1/vcl-nfs-work/ssheorey/Open3D/Datasets/ScanNet-frames-df/frames"
)
scannet_frames_out = Path(
    # "/Users/ssheorey/Documents/Open3D/Data/ScanNet/npy-test/frames_npz_df"
    "/mnt/beegfs/tier1/vcl-nfs-work/ssheorey/Open3D/Datasets/ScanNet-frames-df/frames_easy"
)


def easy_bbox(bbox_file):
    bboxes = np.load(bbox_file)
    bboxes_easy = bboxes[
        np.logical_and(bboxes[:, 8] <= 0.25, bboxes[:, 9] >= 16384), :]
    if len(bboxes_easy) > 0:
        np.save(scannet_frames_out / bbox_file.name, bboxes_easy)
    return len(bboxes_easy), len(bboxes)


def easy_subset():
    with ThreadPoolExecutor(max_workers=1) as executor:
        easy = 0
        total = 0
        easy_frames = 0
        total_frames = 0
        for this_easy, this_total in tqdm(
                executor.map(easy_bbox,
                             Path(scannet_frames).glob("*_bbox.npy"))):
            easy += this_easy
            total += this_total
            total_frames += 1
            easy_frames += 1 if this_easy > 0 else 0
    log.info(
        f"Saved {easy} / {total} bounding boxes in {easy_frames} / {total_frames} frames"
    )


def compress_vertices_file(vertices_file):
    # print(vertices_file)
    point = np.load(vertices_file)
    np.savez_compressed(scannet_frames_out / (vertices_file.name[:-1] + 'z'),
                        point=point)


def compress_vertex_files():
    with ThreadPoolExecutor(max_workers=64) as executor:
        for errs in tqdm(
                executor.map(compress_vertices_file,
                             Path(scannet_frames).glob("*_vert.npy"))):
            pass


def convert_bbox_obj_to_bbox_array():

    scannet_frames_out.mkdir(exist_ok=True)
    errors = []
    for bbox_file in tqdm(Path(scannet_frames).glob("*_bbox.npy")):
        try:
            o3d_bboxes = np.load(bbox_file, allow_pickle=True)
            np_bboxes = np.vstack(list(box.to_xyzwlhyc() for box in o3d_bboxes))
            bbox_out_file = Path(scannet_frames_out) / bbox_file.name
            np.save(bbox_out_file, np_bboxes)
        except Exception:
            errmsg = "\n".join(bbox_file, traceback.format_exc())
            log.warning(errmsg)
            errors.append(errmsg)
    if errors:
        with open(scannet_frames_out / "errors.txt") as errfile:
            errfile.write("\n".join(errors))


if __name__ == "__main__":
    if sys.platform.startswith('linux'):
        multiprocessing.set_start_method('forkserver')
    scannet_frames_out.mkdir(exist_ok=True)
    easy_subset()
