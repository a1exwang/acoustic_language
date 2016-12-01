from redis import StrictRedis as Redis
from data.maps_constants import *
import os
import _pickle as cPickle
import math
import numpy as np
import random


class FileScanner:
    Pieces = 'MUS'

    def __init__(self, maps_root_path, data_type, type_params = None):
        self.mapsdb_root = maps_root_path
        self.type = data_type
        self.type_params = type_params

    def scan(self):
        for instr_name in os.listdir(self.mapsdb_root):
            instr_dir = os.path.join(self.mapsdb_root, instr_name)
            if not os.path.isdir(instr_dir):
                continue
            type_dirs = os.listdir(instr_dir)
            if self.type not in type_dirs:
                print("*** Warning! Cannot find %s in %s instrument" % (self.type, instr_name))
            full_path = os.path.join(instr_dir, self.type)
            for file_name in os.listdir(full_path):
                if file_name.endswith('.wav'):
                    wav_file_path = os.path.join(full_path, file_name)
                    yield wav_file_path


class Cache:
    class CacheNotSatisfiedException(RuntimeError):
        pass

    def get(self, key: str):
        raise self.CacheNotSatisfiedException()

    def put(self, key: str, py_obj):
        pass


class RedisCache(Cache):
    CACHE_REDIS_VERSION_KEY_NAME = '__cache_redis_version_key__'

    def __init__(self, version: str):
        self.redis = Redis(db=1)
        key_count = len(self.redis.keys())
        db_version_bytes = self.redis.get(self.CACHE_REDIS_VERSION_KEY_NAME)
        db_version = db_version_bytes.decode('utf-8') if db_version_bytes else None
        if key_count > 0 and db_version != version:
            raise RuntimeError("Current Redis is at an older version, please backup your redis database and flush db")
        elif key_count == 0:
            self.redis.set(self.CACHE_REDIS_VERSION_KEY_NAME, version)

    def get(self, key: str):
        result = self.redis.get(key)
        if result:
            return cPickle.loads(result)
        else:
            raise self.CacheNotSatisfiedException()

    def put(self, key: str, python_obj):
        self.redis.set(key, cPickle.dumps(python_obj))


class ObjCache(Cache):
    def __init__(self, upstream: Cache):
        self.obj_cache = {}
        self.upstream = upstream

    def get(self, key: str):
        if key not in self.obj_cache:
            file_data = self.upstream.get(key)
            self.obj_cache[key] = file_data

        return self.obj_cache[key]

    def put(self, key: str, py_obj):
        self.upstream.put(key, py_obj)
        self.obj_cache[key] = py_obj


class Maps:
    def __init__(self, root_dir: str, cache, transformer):
        self.maps_root_path = root_dir
        self.transformer = transformer
        self.file_list = []
        self.cache = cache

    def warm_up(self):
        scanner = FileScanner(self.maps_root_path, FileScanner.Pieces)
        for file_path in scanner.scan():
            self.file_list.append(file_path)

    def total_files(self):
        return len(self.file_list)

    def pieces(self, shuffle=False):
        # iterate file scanner, query in cache and pull into memory
        if len(self.file_list) == 0:
            raise RuntimeError("Cache not warm up!")

        index_list = list(range((len(self.file_list))))
        if shuffle:
            random.shuffle(index_list)

        for i in index_list:
            file_path = self.file_list[i]
            try:
                file_data = self.cache.get(file_path)
            except Cache.CacheNotSatisfiedException:
                file_data = self.transformer.get_data(file_path)
                self.cache.put(file_path, file_data)

            yield (file_path, file_data)

    @staticmethod
    def make_batch(x, y, batch_size):
        assert(x.shape[0] == y.shape[0])
        data_count = x.shape[1] // batch_size

        # x_result = np.zeros([x.shape[0], batch_size, data_count])
        # y_result = np.zeros([SEMITONES_ON_PIANO, batch_size, data_count])
        for i in range(data_count):
            x_batch = x[i*batch_size:(i+1)*batch_size, :]
            y_batch = y[i*batch_size:(i+1)*batch_size, :]

            yield (x_batch, y_batch)
