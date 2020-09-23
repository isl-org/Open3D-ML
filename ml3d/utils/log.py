import logging
import os
from pathlib import Path


class LogRecord(logging.LogRecord):

    def getMessage(self):
        msg = self.msg
        if self.args:
            if isinstance(self.args, dict):
                msg = msg.format(**self.args)
            else:
                msg = msg.format(*self.args)
        return msg


def get_tb_hash(path):
    name = Path(path).name
    if not os.path.exists(Path(path).parent):
        return 1
    files = os.listdir(Path(path).parent)
    hsh = 0
    for f in files:
        id, val = f.split('_', 1)
        if name == val:
            hsh = max(hsh, int(id))

    return hsh + 1
