from __future__ import absolute_import
import os
import tempfile
from shutil import copyfile
import logging
from profiler16_un.datasets.file_utils import mkdir_p

logger = logging.getLogger(__name__)


def get_remote_path(env_var='DATASETS'):
    """Returns the location of the remote folder, where the datasets are stored.
       This is done by reading the environment variable env_var with default value PRDATASETS_REMOTE.
       If the latter is not set, a ValueError is thrown
    """
    value = os.environ.get(env_var, None)
    if value is None:
        raise ValueError(
            'You need to set an environment variable {}, containing the path to corpora folder.'.format(env_var))
    return value


def get_local_path():
    """Returns the location of the local cache folding for storing the datsets files"""
    return os.path.expanduser('~/.datasets')


# The location of the remote files
PRDATASETS_REMOTE = get_remote_path()

# The location where fetched files are stored
PRDATASETS_LOCAL = get_local_path()


def download_file(remote_file, remote_path=PRDATASETS_REMOTE):
    """Downloads a remote file to a local temp folder. Returns the pair (dest_file, dest_folder)."""
    tmp_dir = tempfile.mkdtemp()
    dest = os.path.join(tmp_dir, remote_file)
    source = os.path.join(remote_path, remote_file)
    logger.info('Copy remote file: {} -> {}', source, dest)
    mkdir_p(os.path.dirname(dest))
    copyfile(source, dest)
    return dest, tmp_dir


def get_file(name, fill_cache=True, cache_folder=PRDATASETS_LOCAL, remote_folder=PRDATASETS_REMOTE):
    """Gets a file from the local cache, or downloads it from the remote location."""
    dest = os.path.join(cache_folder, name)
    logger.info('Getting {}'.format(dest))
    if os.path.isfile(dest):
        return dest
    else:
        source = os.path.join(remote_folder, name)
        if fill_cache:
            # download file from remote to cache and return cache location
            logger.info(
                'Cache miss! Copy remote file to cache: {} -> {}', source, dest)
            mkdir_p(os.path.dirname(dest))
            copyfile(source, dest)
            return dest
        else:
            # don't download and only return the remote location
            return source
