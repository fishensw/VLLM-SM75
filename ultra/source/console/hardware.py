"""Passive NVML sampling: never imports a CUDA framework or creates a context."""
import json
import sys
import time
import subprocess
import pynvml as n

def value(fn, *args):
    try:
        v = fn(*args)
        return v.decode() if isinstance(v, bytes) else v
    except n.NVMLError:
        return None

n.nvmlInit()
rows = []
handles = [n.nvmlDeviceGetHandleByIndex(i) for i in range(n.nvmlDeviceGetCount())]
for i, h in enumerate(handles):
    mem = value(n.nvmlDeviceGetMemoryInfo, h)
    util = value(n.nvmlDeviceGetUtilizationRates, h)
    pci = value(n.nvmlDeviceGetPciInfo, h)
    power = value(n.nvmlDeviceGetPowerUsage, h)
    state = value(n.nvmlDeviceGetPerformanceState, h)
    rows.append(dict(index=i, uuid=value(n.nvmlDeviceGetUUID, h), name=value(n.nvmlDeviceGetName, h),
        usedMiB=mem.used / 2**20 if mem else None, totalMiB=mem.total / 2**20 if mem else None,
        util=util.gpu if util else None, temp=value(n.nvmlDeviceGetTemperature, h, n.NVML_TEMPERATURE_GPU),
        power=power / 1000 if power is not None else None, pstate=f'P{state}' if state is not None else None,
        bdf=(pci.busId.decode() if isinstance(pci.busId, bytes) else pci.busId) if pci else None,
        coreMHz=value(n.nvmlDeviceGetClockInfo, h, n.NVML_CLOCK_GRAPHICS),
        memoryMHz=value(n.nvmlDeviceGetClockInfo, h, n.NVML_CLOCK_MEM),
        pcieGen=value(n.nvmlDeviceGetCurrPcieLinkGeneration, h), pcieMaxGen=value(n.nvmlDeviceGetMaxPcieLinkGeneration, h),
        pcieWidth=value(n.nvmlDeviceGetCurrPcieLinkWidth, h), pcieMaxWidth=value(n.nvmlDeviceGetMaxPcieLinkWidth, h),
        rxKBps=value(n.nvmlDeviceGetPcieThroughput, h, n.NVML_PCIE_UTIL_RX_BYTES),
        txKBps=value(n.nvmlDeviceGetPcieThroughput, h, n.NVML_PCIE_UTIL_TX_BYTES)))
result = {'time': time.time(), 'gpus': rows, 'pcieSampling': 'NVML short-window sample, KB/s; RX enters GPU, TX leaves GPU'}
if '--p2p' in sys.argv:
    result['p2p'] = [dict(source=i, target=j,
        read=value(n.nvmlDeviceGetP2PStatus, a, b, (n.NVML_P2P_CAPS_INDEX_READ[0] if isinstance(n.NVML_P2P_CAPS_INDEX_READ, tuple) else n.NVML_P2P_CAPS_INDEX_READ)),
        write=value(n.nvmlDeviceGetP2PStatus, a, b, (n.NVML_P2P_CAPS_INDEX_WRITE[0] if isinstance(n.NVML_P2P_CAPS_INDEX_WRITE, tuple) else n.NVML_P2P_CAPS_INDEX_WRITE)))
        for i, a in enumerate(handles) for j, b in enumerate(handles) if i != j]
    result['topology'] = {}
    for name, flags in [('matrix', ['-m']), ('read', ['-p2p', 'r']), ('write', ['-p2p', 'w']), ('pcie', ['-p2p', 'p'])]:
        cmd = ['nvidia-smi', 'topo', *flags]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
            result['topology'][name] = {'command': ' '.join(cmd), 'output': proc.stdout, 'error': proc.stderr, 'exitCode': proc.returncode}
        except (OSError, subprocess.TimeoutExpired) as exc:
            result['topology'][name] = {'command': ' '.join(cmd), 'error': str(exc)}
    result['p2pMeaning'] = 'Driver capability status: 0=OK; not a measured bandwidth or proof of engine transport'
n.nvmlShutdown()
print(json.dumps(result))
