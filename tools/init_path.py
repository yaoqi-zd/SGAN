import os, sys
import os.path as osp

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


# add workspace path
cur_path = osp.dirname(__file__)
add_path(osp.join(cur_path, ".."))