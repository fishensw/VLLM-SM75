# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# vllm-sm75 overlay: 运行中动态开/关投机解码的 scheduler 子类。
# 通过 --scheduler-cls vllm.v1.core.sched.scheduler_sm75.SM75Scheduler 启用。
#
# 设计:把父类的 num_spec_tokens 实例属性改成 property。关闭时 getter 返回 0,
# 于是父类里所有 self.num_spec_tokens 的读取(主 gate / pad 路径 / prefill
# lookahead / KV budget / 统计)自动变 0,无需 override schedule()。不碰上游
# scheduler.py / async_scheduler.py,上游随便改只跟 num_spec_tokens 这个稳定
# 机制耦合。
#
# 继承 AsyncScheduler(非 Scheduler):本项目投机 variant 都开 --async-scheduling,
# 运行时基类就是 AsyncScheduler;继承它可保住 async 调度,不被 get_scheduler_cls
# 的 warning 降级。

import os

from vllm.v1.core.sched.async_scheduler import AsyncScheduler

# 关值集合: VLLM_SPEC_DECODE 命中即"启动时关闭投机", 否则默认开。
# 与 monitor.py 的 VLLM_MONITOR 同款风格(直读环境变量, 不依赖 vllm.envs)。
_OFF_VALUES = ("0", "off", "false", "no")


def _initial_spec_decode_enabled() -> bool:
    """启动时的初始投机状态, 由 VLLM_SPEC_DECODE 决定。

    默认开(未设 = 启动即开); 显式设 0/off/false/no = 启动即关(先跑纯 target
    decode, 运行中经 /monitor 按钮开启)。运行中仍可经 API 随时切换, 覆盖 env 值。
    """
    return os.environ.get("VLLM_SPEC_DECODE", "1").strip().lower() not in _OFF_VALUES


class SM75Scheduler(AsyncScheduler):
    """AsyncScheduler + 运行中投机解码 on/off(只跳草稿计算, 不卸显存)。"""

    @property
    def num_spec_tokens(self) -> int:
        # 关闭时所有读取变 0(纯 target decode); 开启时返回启动时配置的原值。
        return 0 if not self._spec_decode_enabled else self._num_spec_tokens

    @num_spec_tokens.setter
    def num_spec_tokens(self, value: int) -> None:
        # 父类 __init__ 的 self.num_spec_tokens = N 走这里, 存原始配置值(不 gate)。
        self._num_spec_tokens = value

    def __init__(self, *args, **kwargs) -> None:
        # 必须在 super().__init__ 之前: 父类 __init__ 执行期间就会读
        # self.num_spec_tokens(触发 getter, 读这个 flag)。
        # 初始值由 VLLM_SPEC_DECODE 决定, 默认开(未设 = 启动即开)。
        self._spec_decode_enabled = _initial_spec_decode_enabled()
        super().__init__(*args, **kwargs)

    def set_spec_decode_enabled(self, enabled: bool) -> None:
        self._spec_decode_enabled = bool(enabled)

    def is_spec_decode_enabled(self) -> bool:
        return self._spec_decode_enabled

    def is_spec_decode_configured(self) -> bool:
        # 启动时是否配了投机(读原始 N, 不走 gate, 否则"已配置但关闭"会误判 False)。
        return self._num_spec_tokens > 0
