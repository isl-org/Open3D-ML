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


def get_runid(path):
    name = Path(path).name
    if not os.path.exists(Path(path).parent):
        return '00001'
    files = os.listdir(Path(path).parent)
    runid = 0
    for f in files:
        try:
            id, val = f.split('_', 1)
            runid = max(runid, int(id))
        except:
            pass
    runid = str(runid + 1)
    runid = '0' * (5 - len(runid)) + runid

    return runid
