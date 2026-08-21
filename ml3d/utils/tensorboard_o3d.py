"""Register Open3D TensorBoard helpers when the GUI build is available."""


def ensure_tensorboard_plugin():
    """Import the Open3D TensorBoard plugin (registers SummaryWriter.add_3d).

    Headless Open3D builds (BUILD_GUI=OFF) omit Filament rendering; skip import
    so ML pipelines and tests can run without visualization support.
    """
    import open3d as o3d

    if not o3d._build_config.get("BUILD_GUI"):
        return False
    from open3d.visualization.tensorboard_plugin import summary  # noqa: F401
    return True
