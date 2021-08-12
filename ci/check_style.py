import argparse
import multiprocessing
import os
import sys
import time
from functools import partial
from pathlib import Path

# Yapf requires Python 3.6.4+
if sys.version_info < (3, 6, 4):
    raise RuntimeError(
        "Requires Python 3.6.4+, currently using Python {}.{}.{}.".format(
            *sys.version_info))

# Check and import yapf
# > not found: throw exception
# > version mismatch: throw exception
try:
    import yapf
except ImportError:
    raise ImportError(
        "yapf not found. Install with `pip install yapf==0.31.0`.")
if yapf.__version__ != "0.31.0":
    raise RuntimeError(
        "yapf 0.31.0 required. Install with `pip install yapf==0.31.0`.")
print(f"Using yapf version {yapf.__version__}")

# Check and import nbformat
# > not found: throw exception
try:
    import nbformat
except ImportError:
    raise ImportError(
        "nbformat not found. Install with `pip install nbformat`.")
print(f"Using nbformat version {nbformat.__version__}")

PYTHON_FORMAT_DIRS = ["."]

JUPYTER_FORMAT_DIRS = ["."]

# Note: also modify CPP_FORMAT_DIRS in check_cpp_style.cmake.
CPP_FORMAT_DIRS = ["."]


def _glob_files(directories, extensions):
    """Find files with certain extensions in directories recursively.

    Args:
        directories: list of directories, relative to the root Open3D repo directory.
        extensions: list of extensions, e.g. ["cpp", "h"].

    Return:
        List of file paths.
    """
    pwd = Path(os.path.dirname(os.path.abspath(__file__)))
    repo_dir = pwd.parent

    file_paths = []
    for directory in directories:
        directory = repo_dir / directory
        for extension in extensions:
            extension_regex = "*." + extension
            file_paths.extend(directory.rglob(extension_regex))
    return sorted(set(str(file_path) for file_path in file_paths))


class PythonFormatter:

    def __init__(self, file_paths, style_config):
        self.file_paths = file_paths
        self.style_config = style_config

    @staticmethod
    def _check_style(file_path, style_config):
        """Returns true if style is valid."""
        _, _, changed = yapf.yapflib.yapf_api.FormatFile(
            file_path, style_config=style_config, in_place=False)
        return not changed

    @staticmethod
    def _apply_style(file_path, style_config):
        _, _, _ = yapf.yapflib.yapf_api.FormatFile(file_path,
                                                   style_config=style_config,
                                                   in_place=True)

    def run(self, do_apply_style, no_parallel, verbose):
        print(f"{'Applying' if do_apply_style else 'Checking'} Python style...")

        if verbose:
            print("To format:")
            print("\n".join(f"> {file_path}" for file_path in self.file_paths))

        start_time = time.time()
        if no_parallel:
            is_valid_files = map(
                partial(self._check_style, style_config=self.style_config),
                self.file_paths)
        else:
            with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
                is_valid_files = pool.map(
                    partial(self._check_style, style_config=self.style_config),
                    self.file_paths)

        changed_files = [file_path for is_valid, file_path
                         in zip(is_valid_files, self.file_paths)
                         if not is_valid]
        if do_apply_style:
            for file_path in changed_files:
                self._apply_style(file_path, self.style_config)
        print(f"Formatting takes {time.time() - start_time:.2f}s")

        return changed_files


class JupyterFormatter:

    def __init__(self, file_paths, style_config):
        self.file_paths = file_paths
        self.style_config = style_config

    @staticmethod
    def _check_or_apply_style(file_path, style_config, do_apply_style):
        """Returns true if style is valid.

        Since there are common code for check and apply style, the two functions
        are merged into one.
        """
        # Ref: https://gist.github.com/oskopek/496c0d96c79fb6a13692657b39d7c709
        with open(file_path) as f:
            notebook = nbformat.read(f, as_version=nbformat.NO_CONVERT)
        nbformat.validate(notebook)

        changed = False
        for cell in notebook.cells:
            if cell["cell_type"] != "code":
                continue
            src = cell["source"]
            lines = src.split("\n")
            if len(lines) <= 0 or "# noqa" in lines[0]:
                continue
            # yapf will puts a `\n` at the end of each cell, and if this is the
            # only change, cell_changed is still False.
            formatted_src, cell_changed = yapf.yapflib.yapf_api.FormatCode(
                src, style_config=style_config)
            if formatted_src.endswith("\n"):
                formatted_src = formatted_src[:-1]
            if cell_changed:
                cell["source"] = formatted_src
                changed = True

        if do_apply_style:
            with open(file_path, "w") as f:
                nbformat.write(notebook, f, version=nbformat.NO_CONVERT)

        return not changed

    def run(self, do_apply_style, no_parallel, verbose):
        print(f"{'Applying' if do_apply_style else 'Checking'} Jupyter style...")

        if verbose:
            print("To format:")
            print("\n".join(f"> {file_path}" for file_path in self.file_paths))

        start_time = time.time()
        if no_parallel:
            is_valid_files = map(
                partial(self._check_or_apply_style,
                        style_config=self.style_config,
                        do_apply_style=False), self.file_paths)
        else:
            with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
                is_valid_files = pool.map(
                    partial(self._check_or_apply_style,
                            style_config=self.style_config,
                            do_apply_style=False), self.file_paths)

        changed_files = [file_path for is_valid, file_path
                         in zip(is_valid_files, self.file_paths)
                         if not is_valid]
        if do_apply_style:
            for file_path in changed_files:
                self._check_or_apply_style(file_path,
                                           style_config=self.style_config,
                                           do_apply_style=True)
        print(f"Formatting takes {time.time() - start_time:.2f}s")

        return changed_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--do_apply_style",
        dest="do_apply_style",
        action="store_true",
        default=False,
        help="Apply style to files in-place.",
    )
    parser.add_argument(
        "--no_parallel",
        dest="no_parallel",
        action="store_true",
        default=False,
        help="Disable parallel execution.",
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="If true, prints file names while formatting.",
    )
    args = parser.parse_args()

    # Check formatting libs
    pwd = Path(os.path.dirname(os.path.abspath(__file__)))
    python_style_config = str(pwd.parent / ".style.yapf")

    # Check or apply style
    python_formatter = PythonFormatter(_glob_files(PYTHON_FORMAT_DIRS, ["py"]),
                                       style_config=python_style_config)
    jupyter_formatter = JupyterFormatter(_glob_files(JUPYTER_FORMAT_DIRS,
                                                     ["ipynb"]),
                                         style_config=python_style_config)

    changed_files = []
    changed_files.extend(
        python_formatter.run(do_apply_style=args.do_apply_style,
                             no_parallel=args.no_parallel,
                             verbose=args.verbose))
    changed_files.extend(
        jupyter_formatter.run(do_apply_style=args.do_apply_style,
                              no_parallel=args.no_parallel,
                              verbose=args.verbose))

    if changed_files:
        if args.do_apply_style:
            print("Style applied to the following files:")
            print("\n".join(changed_files))
        else:
            print("Style error found in the following files:")
            print("\n".join(changed_files))
            exit(1)
    else:
        print("All files passed style check.")
