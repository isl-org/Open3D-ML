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
        return '00001'
    files = os.listdir(Path(path).parent)
    hsh = 0
    for f in files:
        id, val = f.split('_', 1)
        hsh = max(hsh, int(id))

    hsh = str(hsh + 1)
    hsh = '0' * (5 - len(hsh)) + hsh

    return hsh
