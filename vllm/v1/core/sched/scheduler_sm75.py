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

from vllm.v1.core.sched.async_scheduler import AsyncScheduler


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
        self._spec_decode_enabled = True
        super().__init__(*args, **kwargs)

    def set_spec_decode_enabled(self, enabled: bool) -> None:
        self._spec_decode_enabled = bool(enabled)

    def is_spec_decode_enabled(self) -> bool:
        return self._spec_decode_enabled

    def is_spec_decode_configured(self) -> bool:
        # 启动时是否配了投机(读原始 N, 不走 gate, 否则"已配置但关闭"会误判 False)。
        return self._num_spec_tokens > 0
