from os import makedirs
from os.path import exists, join, isfile, dirname, abspath

def make_dir(folder_name):
	if not exists(folder_name):
		makedirs(folder_name)  
