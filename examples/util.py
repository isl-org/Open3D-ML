# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
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
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------
from os import listdir
from os.path import isfile, join, dirname, abspath, exists
import urllib.request
import tempfile
import shutil


def ensure_demo_data():
    """Check if demo data is present. Else download from github."""
    # Very basic check for demo data. Delete the folder and re-run if data is
    # corrupted.
    DEMO_DATA_DIR = join(dirname(abspath(__file__)), "demo_data")
    DEMO_DATA_URL = "https://github.com/isl-org/open3d_downloads/releases/download/open3d-ml/open3dml_demo_data.zip"
    if exists(DEMO_DATA_DIR) and {'KITTI', 'SemanticKITTI'}.issubset(
            listdir(DEMO_DATA_DIR)):
        return DEMO_DATA_DIR
    print(f"Demo data not found in {DEMO_DATA_DIR}. Downloading...")
    with tempfile.TemporaryDirectory() as dl_dir:
        dl_filename = join(dl_dir, "demo_data.zip")
        with urllib.request.urlopen(DEMO_DATA_URL) as response, open(
                dl_filename, 'wb') as dl_file:
            shutil.copyfileobj(response, dl_file)
        shutil.unpack_archive(dl_filename, DEMO_DATA_DIR)

    return DEMO_DATA_DIR
