import logging
from os import makedirs
from os.path import exists, join, isfile, dirname, abspath

def make_dir(folder_name):
	if not exists(folder_name):
		makedirs(folder_name)  

class LogRecord(logging.LogRecord):
    def getMessage(self):
        msg = self.msg
        if self.args:
            if isinstance(self.args, dict):
                msg = msg.format(**self.args)
            else:
                msg = msg.format(*self.args)
        return msg
