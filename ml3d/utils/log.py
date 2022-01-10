import logging
import os
from pathlib import Path


class LogRecord(logging.LogRecord):
    """Class for logging information."""

    def getMessage(self):
        msg = self.msg
        if self.args:
            if isinstance(self.args, dict):
                msg = msg.format(**self.args)
            else:
                msg = msg.format(*self.args)
        return msg


def get_runid(path):
    """Get runid for an experiment."""
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


def code2md(code_text, language=None):
    """Format code as markdown for display (eg in tensorboard)"""
    four_spaces = '    '
    code_md = four_spaces + code_text.replace(os.linesep,
                                              os.linesep + four_spaces)
    return code_md[:-4]
