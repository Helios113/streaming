# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Register or look up the prefix to use for all shared resources.

The prefix is used by all workers using this StreamingDataset of this training job. This is used to
prevent shared resources like shared memory from colliding.
"""

import os
import uuid
from collections import Counter
from tempfile import gettempdir
from time import sleep
from typing import Iterator, Union

import numpy as np
from torch import distributed as dist

from streaming.base.constant import BARRIER_FILELOCK, CACHE_FILELOCK, LOCALS, SHM_TO_CLEAN, TICK
from streaming.base.shared import SharedMemory
from streaming.base.world import World


def _each_prefix_int() -> Iterator[int]:
    """Get each possible prefix int to check in order.

    Returns:
        Iterator[int]: Each prefix int.
    """
    prefix_int = 0
    while True:
        yield prefix_int
        prefix_int += 1


def _get_path(prefix_int: int, name: str) -> str:
    """Get the name of the shared memory.

    Args:
        prefix (int): The prefix int.
        name (str): The name of the shared memory.

    Returns:
        str: Unique shared memory name.
    """
    return f'{prefix_int:06}_{name}'


def _pack_locals(dirnames: list[str], prefix_int: int) -> bytes:
    """Pack local dirnames and prefix int.

    Args:
        dirnames (List[str]): Unpacked local dirnames.
        prefix_int (int): Prefix int.

    Returns:
        bytes: Packed local dirnames and prefix int.
    """
    text = '\0'.join(dirnames) + f'\0{prefix_int}'
    data = text.encode('utf-8')
    size = 4 + len(data)
    return b''.join([np.int32(size).tobytes(), data])


def _unpack_locals(data: bytes) -> tuple[list[str], int]:
    """Unpack local dirnames and prefix int.

    Args:
        data (bytes): Packed local dirnames and prefix int.

    Returns:
        List[str]: Unpacked local dirnames and prefix int.
    """
    size = np.frombuffer(data[:4], np.int32)[0]
    text = data[4:size].decode('utf-8')
    text = text.split('\0')
    return text[:-1], int(text[-1] or 0)


def _check_self(streams_local: list[str]) -> None:
    """Check our local working directories for overlap.

    Args:
        streams_local (List[str]): Local dirs.

    Raises:
        ValueError: If there is overlap.
    """
    occurrences = Counter(streams_local)
    duplicate_local_dirs = [dirname for dirname, count in occurrences.items() if count > 1]
    if duplicate_local_dirs:
        raise ValueError(
            f'Reused local directory: {duplicate_local_dirs}. Provide a different one.')


def _make_unique_local_dirs(streams_local: list[str], their_locals: list[str]) -> list[str]:
    """Make local directories unique by adding a suffix to avoid conflicts.
    
    Args:
        streams_local (List[str]): Original local directories.
        their_locals (List[str]): Existing local directories that conflict.
        
    Returns:
        List[str]: Modified local directories with unique suffixes.
    """
    unique_suffix = str(uuid.uuid4())[:8]
    modified_dirs = []
    
    for local_dir in streams_local:
        if local_dir and local_dir in their_locals:
            # Add unique suffix to conflicting directory
            modified_dirs.append(f"{local_dir}_{unique_suffix}")
        else:
            modified_dirs.append(local_dir)
    
    return modified_dirs


def _check_and_find(streams_local: list[str], streams_remote: list[Union[str, None]],
                    shm_name: str) -> tuple[int, list[str]]:
    """Find the next available prefix and resolve any directory conflicts.

    Local leader walks the existing shm prefixes starting from zero, resolving any
    local working directory conflicts by creating unique directories. When attaching to an existing
    shm fails, we have reached the end of the existing shms. We will register the next one.

    Args:
        streams_local (List[str]): Our local working directories.
        streams_remote (List[Union[str, None]]): Our remote working directories.
        shm_name (str): The shared memory file name, e.g., LOCALS, BARRIER etc.

    Returns:
        tuple[int, List[str]]: Next available prefix int and potentially modified local directories.
    """
    prefix_int = 0
    current_streams_local = streams_local.copy()

    for prefix_int in _each_prefix_int():

        name = _get_path(prefix_int, shm_name)

        # Check if any shared memory filelocks exist for the current prefix
        try:
            filelock_exists = any(
                os.path.exists(os.path.join(gettempdir(), _get_path(prefix_int, filelock_name)))
                for filelock_name in [BARRIER_FILELOCK, CACHE_FILELOCK])
            if filelock_exists:
                continue
        except PermissionError:
            continue

        # Attempt to access shared memory by name. Use prefix_int if files do not exist
        try:
            shm = SharedMemory(name, False)
        except PermissionError:
            continue
        except FileNotFoundError:
            break  # This prefix is available - no existing shared memory

        if shm_name != LOCALS:
            continue

        their_locals, _ = _unpack_locals(bytes(shm.buf))

        # Check for conflicting local directories and resolve them
        if any(streams_remote):
            # Get the indices of the local directories which matches with the current
            # shared memory.
            matching_index = np.where(np.isin(current_streams_local, their_locals))[0]
            if matching_index.size > 0:
                has_conflicts = any(streams_remote[idx] is not None for idx in matching_index)
                if has_conflicts:
                    print("SHM conflict detected.", flush=True)
                    # Instead of raising error, create unique directories
                    current_streams_local = _make_unique_local_dirs(current_streams_local, their_locals)
                    # Continue to next prefix with modified directories
                    continue
    
    return prefix_int, current_streams_local


def _check_and_find_retrying(streams_local: list[str], streams_remote: list[Union[str, None]],
                             shm_name: str, retry: int) -> tuple[int, list[str]]:
    """Find the next available prefix while resolving directory conflicts.

    If an overlap is found, modifies directories and tries again, up to "retry" times. We allow
    this grace period because modifying python shared memory in a destructor intermediated through
    a numpy array appears to be racy.

    Args:
        streams_local (List[str]): Our local working directories.
        streams_remote (List[Union[str, None]]): Our remote working directories.
        shm_name (str): The shared memory file name, e.g., LOCALS, BARRIER etc.
        retry (int): Number of retries upon failure before raising an exception.

    Returns:
        tuple[int, List[str]]: Next available prefix int and potentially modified local directories.
    """
    if retry < 0:
        raise ValueError(f'Specify at least zero retries (provided {retry}).')
    
    errs = []
    for _ in range(1 + retry):
        try:
            return _check_and_find(streams_local, streams_remote, shm_name)
        except ValueError as err:
            errs.append(err)
            sleep(TICK)
    raise errs[-1]


def get_shm_prefix(streams_local: list[str],
                   streams_remote: list[Union[str, None]],
                   world: World,
                   retry: int = 100) -> tuple[int, SharedMemory]:
    """Register or lookup our shared memory prefix.

    Args:
        streams_local (List[str]): Local working dir of each stream, relative to /.
            We need to verify that there is no overlap with any other currently
            running StreamingDataset.
        streams_remote (List[Union[str, None]]): Remote working dir of each stream.
        world (World): Information about nodes, ranks, and workers.
        retry (int): Number of retries upon failure before raising an exception.
            Defaults to ``100``.

    Returns:
        Tuple[int, SharedMemory]: Shared memory integer prefix and object. The name
            is required to be very short due to limitations of Python on Mac OSX.
    """
    # Check my locals for overlap.
    _check_self(streams_local)

    # Find prefix and get potentially modified local directories
    results = [
        _check_and_find_retrying(streams_local, streams_remote, shm_name=shm_name, retry=retry)
        for shm_name in SHM_TO_CLEAN
    ]
    
    # Use the maximum prefix and the local directories from that result
    prefix_int = max(result[0] for result in results)
    # Get the modified streams_local from the result that gave us the max prefix
    final_streams_local = next(result[1] for result in results if result[0] == prefix_int)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    # First, the local leader registers the first available shm prefix, recording its locals.
    if world.is_local_leader:
        name = _get_path(prefix_int, LOCALS)
        data = _pack_locals(final_streams_local, prefix_int)
        shm = SharedMemory(name, True, len(data))
        shm.buf[:len(data)] = data

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    # Non-local leaders go next, searching for match.
    if not world.is_local_leader:
        name = _get_path(prefix_int, LOCALS)
        try:
            shm = SharedMemory(name, False)
        except FileNotFoundError:
            raise RuntimeError(f'Internal error: shared memory prefix={prefix_int} was not ' +
                               f'registered by local leader. This may be because you specified ' +
                               f'different ``local`` parameters from different ranks.')

        their_locals, their_prefix_int = _unpack_locals(bytes(shm.buf))
        if final_streams_local != their_locals or prefix_int != their_prefix_int:
            raise RuntimeError(f'Internal error: shared memory registered does not match ' +
                               f'local leader as streams_local or prefix_int not match. ' +
                               f'local leader: {their_locals} and {their_prefix_int}. ' +
                               f'expected: {final_streams_local} and {prefix_int}.')
    return prefix_int, shm  # pyright: ignore
