"""请求上下文的分支托管。

处理器在请求期间派生的异步工作分为两类：

* 普通子任务共享当前请求上下文。请求结束（响应已发送或处理被取消）后，
  这类工作继续持有的请求句柄会被标记失效，``Request.get_current()``
  对它们抛出与“无请求”相同的错误，从而无法读到复用连接上的下一位
  用户身份。
* 需要比请求活得更久的工作必须显式 ``request.spawn_branch()`` 领取一份
  最小化的不可变快照（:class:`RequestSnapshot`），并声明完成时必须执行
  的清理责任。

分支的取消、异常、任务转交与 worker 停机都遵循同一条规则：业务体最多
执行一次，清理体最多执行一次，二者都不会在中途把原始请求暴露出去。
"""

from __future__ import annotations

from asyncio import CancelledError
from contextlib import suppress
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    NamedTuple,
)

from sanic.log import error_logger


if TYPE_CHECKING:
    from asyncio import Task

    from sanic.app import Sanic
    from sanic.request.types import Request


BranchBody = Callable[["RequestSnapshot"], Awaitable[Any]]
BranchCleanup = Callable[
    ["RequestSnapshot", BaseException | None], Awaitable[Any]
]


class BranchState(str, Enum):
    """分支生命周期状态。"""

    RUNNING = "running"
    TRANSFERRED = "transferred"
    CANCELLED = "cancelled"
    FAILED = "failed"
    DONE = "done"

    def __str__(self) -> str:
        return self.value


class BranchDiagnostic(NamedTuple):
    """诊断视图：只说明分支来源，不含任何请求敏感值。"""

    name: str | None
    state: str
    request_id: str | None
    method: str
    path: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state,
            "request_id": self.request_id,
            "method": self.method,
            "path": self.path,
        }


# 请求结束后仍可能被诊断接口展示的终态。
_TERMINAL_STATES = frozenset(
    (
        BranchState.CANCELLED,
        BranchState.FAILED,
        BranchState.DONE,
    )
)


class RequestSnapshot:
    """脱离请求生命周期的最小化身份快照。

    快照只复制补写审计摘要所必需的非敏感标识：请求 ID、方法、路径与
    路由端点名。请求头、cookie、token、请求体、查询串以及可变的
    ``request.ctx`` 均不在快照内，因此请求对象随后被复用时，分支既不会
    读到下一位用户的值，也无法通过快照回溯任何凭据。
    """

    __slots__ = ("_endpoint", "_id", "_method", "_path")

    def __init__(self, request: Request):
        # 读取发生在领取快照的瞬间：此时请求仍属于当前用户，取到的是
        # 稳定的标量值；之后原始请求如何复用都与快照无关。
        self._id = str(request.id) if request.id is not None else None
        self._method = request.method
        self._path = request.path
        self._endpoint = request.name

    def __repr__(self) -> str:
        return (
            f"<RequestSnapshot: {self._method} {self._path} "
            f"request={self._id}>"
        )

    @property
    def request_id(self) -> str | None:
        return self._id

    @property
    def method(self) -> str:
        return self._method

    @property
    def path(self) -> str:
        return self._path

    @property
    def endpoint(self) -> str | None:
        return self._endpoint


class RequestContextBranch:
    """一个脱离请求、自带快照与清理责任的上下文分支。"""

    __slots__ = (
        "_app",
        "_body",
        "_cleanup",
        "_exception",
        "_name",
        "_owner",
        "_snapshot",
        "_state",
        "_task",
    )

    def __init__(
        self,
        request: Request,
        body: BranchBody,
        cleanup: BranchCleanup | None,
        name: str | None,
    ):
        self._app = request.app
        self._snapshot = RequestSnapshot(request)
        self._body = body
        self._cleanup = cleanup
        self._name = name
        self._state: BranchState = BranchState.RUNNING
        self._task: Task | None = None
        self._exception: BaseException | None = None
        self._owner: object | None = None

    def __repr__(self) -> str:
        return (
            f"<RequestContextBranch: {self._state} "
            f"{self._snapshot.method} {self._snapshot.path}>"
        )

    # --------------------------------------------------------------- #
    # 对外只读视图
    # --------------------------------------------------------------- #

    @property
    def name(self) -> str | None:
        return self._name

    @property
    def state(self) -> BranchState:
        return self._state

    @property
    def snapshot(self) -> RequestSnapshot:
        return self._snapshot

    @property
    def task(self) -> Task | None:
        return self._task

    @property
    def exception(self) -> BaseException | None:
        return self._exception

    @property
    def custodian(self) -> object | None:
        """责任转交后的接收方；未转交时为 None。"""
        return self._owner

    @property
    def done(self) -> bool:
        return self._state in _TERMINAL_STATES

    def diagnostic(self) -> BranchDiagnostic:
        return BranchDiagnostic(
            name=self._name,
            state=str(self._state),
            request_id=self._snapshot.request_id,
            method=self._snapshot.method,
            path=self._snapshot.path,
        )

    # --------------------------------------------------------------- #
    # 生命周期
    # --------------------------------------------------------------- #

    def bind(self, task: Task) -> None:
        """登记承载分支的 asyncio 任务（用于停机时取消）。"""
        self._task = task

    async def run(self) -> Any:
        """执行分支业务体，并保证清理责任恰好执行一次。

        无论业务体正常返回、抛出异常还是被取消，清理回调都会带着业务体
        的终局原因运行一次；清理本身抛出的异常会被记录而不会掩盖业务体
        的原始异常，清理也不会被重复执行。
        """
        exc: BaseException | None = None
        try:
            return await self._body(self._snapshot)
        except BaseException as e:  # 包括 CancelledError
            exc = e
            raise
        finally:
            # 先落定状态再清理：即使清理过程被二次取消，状态也已记录。
            self._settle(exc)
            await self._run_cleanup(exc)

    async def cancel(self, msg: str | None = None) -> None:
        """取消承载任务并等待分支收尾（清理执行且只执行一次）。"""
        task = self._task
        if task is None or task.done():
            # 没有可等待的承载任务（正常托管流程下不会发生）：直接确保
            # 清理恰好执行一次。已收尾的分支是空操作。
            if not self.done:
                self._settle(CancelledError(msg or "branch cancelled"))
                await self._run_cleanup(self._exception)
            return

        if msg:
            task.cancel(msg)
        else:
            task.cancel()
        # 业务异常与取消都已在 run() 内完成清理，这里取回结果只是等待
        # 收尾，不能让表层异常阻止其他分支的停机流程。
        with suppress(BaseException):
            await task

    def claim(self, owner: object) -> None:
        """将完成责任转交给新的宿主（如持久队列、worker 对端）。

        转交后框架不再把该分支计为存活请求分支：诊断接口不再展示它，
        本机停机也不会取消它，由接收方负责让执行走到终点。正在执行的
        任务与已声明的清理回调仍附着在实际执行体上，清理因此仍然恰好
        执行一次；接收方不应再登记第二份清理。转交是一次性的，重复
        转交或对已终态分支转交都会报错，防止责任悬空后双方都认为该
        由对方清理。
        """
        if self._state is not BranchState.RUNNING:
            raise RuntimeError(
                f"Cannot transfer a {self._state} branch; only a running "
                "branch can change custody."
            )
        if owner is None:
            raise ValueError("Transferred branch must name a new custodian.")
        self._owner = owner
        self._state = BranchState.TRANSFERRED
        # 本机不再追踪其生命周期，诊断中也不再显示为存活分支。
        branch_registry.discard(self, self._app)

    async def _run_cleanup(self, exc: BaseException | None) -> None:
        cleanup = self._cleanup
        if cleanup is None:
            return
        # 清空引用，保证清理最多执行一次（cancel 与 run 的 finally
        # 发生竞态时不会重入）。
        self._cleanup = None
        try:
            await cleanup(self._snapshot, exc)
        except CancelledError:
            # 停机期间清理被打断：清理引用已清空，不会被重复执行，
            # 按取消而非错误记录。
            error_logger.warning(
                "Branch cleanup interrupted by cancellation for %s",
                self._label(),
            )
        except BaseException as cleanup_exc:  # noqa: BLE001
            error_logger.exception(
                "Branch cleanup failed for %s: %s",
                self._label(),
                cleanup_exc,
            )

    def _settle(self, exc: BaseException | None) -> None:
        if self._state is BranchState.TRANSFERRED:
            # 责任已转交，本机不再声明终态。
            return
        self._exception = exc
        if isinstance(exc, CancelledError):
            self._state = BranchState.CANCELLED
        elif exc is not None:
            self._state = BranchState.FAILED
        else:
            self._state = BranchState.DONE

    def _label(self) -> str:
        return self._name or repr(self._snapshot)


# --------------------------------------------------------------------- #
# 进程级注册表：诊断与停机收尾（按应用隔离）
# --------------------------------------------------------------------- #


class _BranchRegistry:
    """登记各应用的存活分支，供诊断查询与 worker 停机统一收尾。"""

    def __init__(self) -> None:
        self._branches: dict[Any, set[RequestContextBranch]] = {}

    def add(self, branch: RequestContextBranch, app: Sanic) -> None:
        self._branches.setdefault(app, set()).add(branch)

    def discard(self, branch: RequestContextBranch, app: Sanic) -> None:
        with suppress(KeyError):
            group = self._branches[app]
            group.discard(branch)
            if not group:
                del self._branches[app]

    def alive(self, app: Sanic | None = None) -> list[RequestContextBranch]:
        groups = (
            (self._branches.get(app, set()),)
            if app is not None
            else tuple(self._branches.values())
        )
        return [
            branch
            for group in groups
            for branch in tuple(group)
            if branch.state is BranchState.RUNNING
        ]

    async def shutdown(self, app: Sanic) -> None:
        """取消指定应用全部仍在运行的分支并等待清理完成。

        已转交（TRANSFERRED）的分支由接收方负责，不在此取消（转交时已
        出册）；每个分支的取消各自独立，单个分支收尾异常不影响其他
        分支。
        """
        for branch in tuple(self._branches.get(app, set())):
            try:
                await branch.cancel()
            except BaseException as exc:  # noqa: BLE001
                error_logger.exception(
                    "Error while shutting down branch %s: %s",
                    branch._label(),
                    exc,
                )
            finally:
                self.discard(branch, app)

    def __len__(self) -> int:
        return sum(len(group) for group in self._branches.values())


branch_registry = _BranchRegistry()


def attach_branch(
    branch: RequestContextBranch, task: Task, app: Sanic
) -> None:
    """绑定承载任务并登记存活分支；任务终态后自动出册。"""
    branch.bind(task)
    branch_registry.add(branch, app)

    def _on_done(done_task: Task) -> None:
        branch_registry.discard(branch, app)
        # 取回业务异常，避免“从未检索的任务异常”警告；异常本身已经由
        # run() 的清理路径与 branch.exception 记录在案。
        if not done_task.cancelled():
            with suppress(BaseException):
                done_task.exception()

    task.add_done_callback(_on_done)
