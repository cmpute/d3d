import gc
import os
import time
import weakref
import logging

import torch
from py3nvml import py3nvml

_timers = {}
def tap_time(name, clean=False, report=False):
    '''
    This function start a timer with certain name. The second call of this function with
    same name will stop the timer and report the time.
    '''

    torch.cuda.synchronize()
    if name not in _timers:
        _timers[name] = time.time()
        return 0
    else:
        elapse = time.time() - _timers[name]
        if clean:
            del _timers[name]
        if report:
            logging.getLogger('d3d.profiler').debug("Elapsed time for %s: %.4f" % (name, elapse))
        return elapse

class TensorRef:
    def __init__(self, tensor):
        self._ref = weakref.ref(tensor)
        self._summary = f"<Tensor, type={type(tensor).__name__}, shape={list(tensor.shape)}, device={tensor.device}>"
    def __hash__(self):
        return hash(self._ref)
    def __eq__(self, other):
        return self._ref() is other
    def __str__(self):
        return self._summary
    def released(self):
        return self._ref() is None

_tensors = set()
def tap_tensors(report=False):
    '''
    Used for memory leak debugging
    '''
    tensor_new = [obj for obj in gc.get_objects() if torch.is_tensor(obj) and obj not in _tensors]
    tensor_del = [obj for obj in _tensors if obj.released()]
    logger = logging.getLogger('d3d.profiler')

    if report:
        logger.debug(f'========== {len(tensor_new)} new tensors, {len(tensor_del)} released tensors ==========')

    if len(tensor_new) > 50:
        logger.debug("(Tensor list suppressed)")
        report = False # prevent list being too long
        
    for tensor in tensor_new:
        ref = TensorRef(tensor)
        if report:
            logger.debug("+" + str(ref))
        _tensors.add(ref)
    for ref in tensor_del:
        if report:
            logger.debug("-" + str(ref))
        _tensors.remove(ref)

    return tensor_new, tensor_del

_gpu_connected = False
def report_gpumem(gpus=None):
    global _gpu_connected
    if not _gpu_connected:
        py3nvml.nvmlInit()
        _gpu_connected = True

    gpus = gpus or range(py3nvml.nvmlDeviceGetCount())
    total_mem = 0
    for gidx in gpus:
        handle = py3nvml.nvmlDeviceGetHandleByIndex(gidx)
        procs = py3nvml.nvmlDeviceGetComputeRunningProcesses(handle)
        cproc = next(p for p in procs if p.pid == os.getpid())
        total_mem += cproc.usedGpuMemory

    return total_mem
