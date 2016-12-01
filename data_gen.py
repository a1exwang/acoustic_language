import argparse

from data.cqt_frame_transformer import Transformer
from data.maps import Maps, RedisCache

parser = argparse.ArgumentParser(description='MAPS Database importer')
parser.add_argument('--maps-dir',
                    help='MAPS database root directory')

args = parser.parse_args()
root_dir = args.maps_dir
if not root_dir:
    raise RuntimeError("Wrong Argument")

t = Transformer()
cache = RedisCache('audio')
maps = Maps(root_dir=root_dir, transformer=t, cache=cache)
maps.warm_up()
for (file_path, file_data) in maps.pieces(shuffle=False):
    print(file_path)
