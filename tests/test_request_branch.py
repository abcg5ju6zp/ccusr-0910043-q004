"""请求上下文分支托管的回归测试。

覆盖安全主线：处理器结束后，普通子任务无法再读取（可能已被下一请求
复用的）请求上下文；延后工作只能领取最小化快照；取消、异常、转交、
worker 停机都不泄漏身份、不重复执行清理；诊断信息不含敏感值。
"""

import asyncio
import logging

from unittest.mock import Mock

import pytest

from sanic_testing.reusable import ReusableClient

from sanic import Sanic, response
from sanic.exceptions import ServerError
from sanic.request import (
    BranchState,
    Request,
    RequestSnapshot,
)
from sanic.request.branch import branch_registry


def make_request(app=None, path=b"/", headers=None, method="POST", rid="r1"):
    if app is None:
        app = Mock()
        app.config.REQUEST_ID_HEADER = "x-request-id"
    request = Request(
        path,
        headers or {"authorization": "Bearer secret-token"},
        "1.1",
        method,
        Mock(),
        app,
    )
    request._id = rid
    return request


# --------------------------------------------------------------------- #
# 普通子任务随请求结束失效
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_get_current_denied_after_finalize():
    request = make_request(rid="alice")
    token = Request._current.set(request)
    try:
        observed = {}

        async def straggler():
            # 跨越请求结束时刻后再读取上下文
            await asyncio.sleep(0.01)
            try:
                observed["request"] = Request.get_current()
            except ServerError:
                observed["denied"] = True

        task = asyncio.create_task(straggler())
        await asyncio.sleep(0)
        request.finalize()
        await task

        assert observed == {"denied": True}
        assert request.finalized is True
    finally:
        Request._current.reset(token)


@pytest.mark.asyncio
async def test_get_current_compatible_while_request_alive(app):
    """请求存活期间，同步读取方式保持原样可用。"""
    token = Request._current.set(None)
    request = make_request(app=app, rid="live")
    Request._current.set(request)
    try:
        assert Request.get_current() is request
    finally:
        Request._current.reset(token)


def test_get_current_without_request_still_raises():
    with pytest.raises(ServerError, match="No current request"):
        Request.get_current()


@pytest.mark.asyncio
async def test_finalize_is_idempotent():
    request = make_request()
    request.finalize()
    request.finalize()  # 不得抛错或改变其他状态
    assert request.finalized


# --------------------------------------------------------------------- #
# 最小化快照
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_snapshot_is_minimal_and_carries_no_secret():
    request = make_request(
        path=b"/audit/alice",
        headers={
            "authorization": "Bearer secret-token",
            "cookie": "session=super-secret",
            "x-api-key": "key-1234",
        },
        rid="req-alice",
    )
    request.ctx.audit_identity = "alice"

    snapshot = RequestSnapshot(request)

    assert snapshot.request_id == "req-alice"
    assert snapshot.method == "POST"
    assert snapshot.path == "/audit/alice"

    # 快照上不存在任何可以回溯凭据或可变上下文的入口
    for attr in (
        "headers",
        "token",
        "credentials",
        "cookies",
        "body",
        "json",
        "form",
        "args",
        "query_string",
        "url",
        "ctx",
        "transport",
    ):
        assert not hasattr(snapshot, attr), attr

    # __slots__ 保证快照不可变
    with pytest.raises(AttributeError):
        snapshot.token = "x"  # type: ignore[misc]


# --------------------------------------------------------------------- #
# 清理责任：成功 / 异常 / 取消各执行一次
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_branch_cleanup_runs_once_on_success(app):
    request = make_request(app=app, rid="ok")
    calls = []

    async def body(snapshot):
        await asyncio.sleep(0)
        return "summary"

    async def cleanup(snapshot, exc):
        calls.append(exc)

    branch = request.spawn_branch(body, cleanup=cleanup, name="audit-success")
    result = await branch.task

    assert result == "summary"
    assert branch.state is BranchState.DONE
    assert calls == [None]
    # 任务终态后自动出册
    assert branch_registry.alive(app) == []


@pytest.mark.asyncio
async def test_branch_cleanup_runs_once_on_exception(app):
    request = make_request(app=app, rid="fail")
    calls = []
    failure = ValueError("write failed")

    async def body(snapshot):
        raise failure

    async def cleanup(snapshot, exc):
        calls.append(exc)

    branch = request.spawn_branch(body, cleanup=cleanup)
    with pytest.raises(ValueError, match="write failed"):
        await branch.task

    assert branch.state is BranchState.FAILED
    assert branch.exception is failure
    assert calls == [failure]


@pytest.mark.asyncio
async def test_branch_cleanup_runs_once_on_cancel(app):
    request = make_request(app=app, rid="cancel")
    calls = []

    async def body(snapshot):
        await asyncio.sleep(10)

    async def cleanup(snapshot, exc):
        calls.append(type(exc).__name__)
        await asyncio.sleep(0)

    branch = request.spawn_branch(body, cleanup=cleanup)
    await asyncio.sleep(0.01)
    await branch.cancel()
    # 二次取消必须是空操作，不能再触发一次清理
    await branch.cancel()

    assert branch.state is BranchState.CANCELLED
    assert calls == ["CancelledError"]


@pytest.mark.asyncio
async def test_branch_cleanup_failure_is_logged_not_raised(app, caplog):
    request = make_request(app=app, rid="bad-cleanup")

    async def body(snapshot):
        await asyncio.sleep(0)

    async def cleanup(snapshot, exc):
        raise RuntimeError("cleanup boom")

    branch = request.spawn_branch(body, cleanup=cleanup)
    with caplog.at_level(logging.ERROR):
        # 清理异常不得向外掩盖业务结果
        result = await branch.task

    assert result is None
    assert "cleanup boom" in caplog.text
    assert branch.state is BranchState.DONE


# --------------------------------------------------------------------- #
# 派生与转交规则
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_spawn_after_finalize_is_rejected(app):
    request = make_request(app=app)
    request.finalize()
    with pytest.raises(ServerError, match="finalized request"):
        request.spawn_branch(lambda s: None)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_transfer_removes_branch_from_local_custody(app):
    request = make_request(app=app, path=b"/audit/a", rid="r-a")
    queue = object()

    branch = request.spawn_branch(
        lambda s: asyncio.sleep(10),  # type: ignore[arg-type,return-value]
        name="handoff",
    )
    assert Request.live_branches(app)

    request.accept_branch(branch, queue)

    assert branch.state is BranchState.TRANSFERRED
    assert branch.custodian is queue
    # 转交后不再被本机诊断/停机追踪
    assert branch_registry.alive(app) == []

    # 重复转交、对外人转交、转交终态分支都必须失败
    with pytest.raises(RuntimeError, match="transferred branch"):
        request.accept_branch(branch, object())

    other = make_request(app=app, path=b"/audit/b", rid="r-b")
    with pytest.raises(ServerError, match="not owned"):
        other.accept_branch(branch, object())

    branch.task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await branch.task


# --------------------------------------------------------------------- #
# 诊断接口：说明来源但不泄露敏感值
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_diagnostics_show_origin_without_secrets(app):
    request = make_request(
        app=app,
        path=b"/audit/alice",
        headers={"authorization": "Bearer secret-token"},
        rid="diag-1",
    )
    branch = request.spawn_branch(
        lambda s: asyncio.sleep(10),  # type: ignore[arg-type,return-value]
        name="audit",
    )
    try:
        live = Request.live_branches(app)
        assert len(live) == 1
        info = live[0].as_dict()
        assert info == {
            "name": "audit",
            "state": "running",
            "request_id": "diag-1",
            "method": "POST",
            "path": "/audit/alice",
        }
        rendered = repr(info)
        assert "secret-token" not in rendered
    finally:
        await branch.cancel()


# --------------------------------------------------------------------- #
# 端到端：keep-alive 连接复用，身份不串、延后工作归属正确
# --------------------------------------------------------------------- #


def test_branch_identity_survives_reused_connection(port):
    app = Sanic("test-branch-reuse")
    summaries = []
    straggler_reads = []

    @app.post("/audit/<user>")
    async def audit(request, user):
        # 旧审计中间件的错误用法：普通子任务在请求结束后才补写。
        async def straggler():
            await asyncio.sleep(0.05)
            try:
                current = Request.get_current()
            except ServerError:
                straggler_reads.append("denied")
            else:
                straggler_reads.append(f"leak:{current.path}")

        asyncio.create_task(straggler())

        # 正确用法：领取最小化快照后再延后执行。
        async def delayed_summary(snapshot):
            await asyncio.sleep(0.04)
            summaries.append((user, snapshot.path, snapshot.request_id))

        request.spawn_branch(delayed_summary, name="audit-summary")
        return response.text("ok")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    client = ReusableClient(app, loop=loop, port=port)
    with client:
        req1, resp1 = client.post(
            "/audit/alice",
            headers={"Connection": "keep-alive"},
        )
        req2, resp2 = client.post("/audit/bob")
        assert resp1.status == resp2.status == 200
        # 两个请求跑在同一条 keep-alive 连接上
        assert req1.protocol is req2.protocol
        assert req2.protocol.state["requests_count"] == 2

        # 等延后工作跨过第二个请求的生命周期完成
        loop.run_until_complete(asyncio.sleep(0.15))

    # 快照归属的是各自的发起请求，没有把 bob 记到 alice 那笔
    by_request = {rid: (user, path) for user, path, rid in summaries}
    assert by_request[str(req1.id)] == ("alice", "/audit/alice")
    assert by_request[str(req2.id)] == ("bob", "/audit/bob")
    # 普通子任务在请求结束后一律读不到请求上下文
    assert straggler_reads == ["denied", "denied"]


# --------------------------------------------------------------------- #
# 端到端：worker 停机取消存活分支并执行清理；转交分支不动
# --------------------------------------------------------------------- #


def test_shutdown_cancels_alive_branch_and_runs_cleanup(port):
    app = Sanic("test-branch-shutdown")
    cleanup_seen = []
    handle = {}

    @app.post("/audit")
    async def audit(request):
        async def work(snapshot):
            await asyncio.sleep(30)

        async def cleanup(snapshot, exc):
            cleanup_seen.append(type(exc).__name__)

        handle["branch"] = request.spawn_branch(
            work, cleanup=cleanup, name="long-audit"
        )
        return response.text("ok")

    client = ReusableClient(app, port=port)
    with client:
        _, resp = client.post("/audit")
        assert resp.status == 200
        branch = handle["branch"]
        assert branch.state is BranchState.RUNNING
        # 退出 with 即触发 before_server_stop
    assert branch.state is BranchState.CANCELLED
    assert cleanup_seen == ["CancelledError"]


def test_shutdown_does_not_cancel_transferred_branch(port):
    app = Sanic("test-branch-transfer-shutdown")
    external_queue = object()
    handle = {}

    @app.post("/audit")
    async def audit(request):
        async def work(snapshot):
            await asyncio.sleep(30)

        branch = request.spawn_branch(work, name="transferred-audit")
        request.accept_branch(branch, external_queue)
        handle["branch"] = branch
        return response.text("ok")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    client = ReusableClient(app, loop=loop, port=port)
    with client:
        _, resp = client.post("/audit")
        assert resp.status == 200
        branch = handle["branch"]

    # 责任已转交：停机不取消它
    assert branch.state is BranchState.TRANSFERRED
    assert not branch.task.cancelled()
    branch.task.cancel()
    loop.run_until_complete(
        asyncio.gather(branch.task, return_exceptions=True)
    )


# --------------------------------------------------------------------- #
# ASGI 路径同样在请求结束后失效
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_asgi_finalizes_request(app):
    reads = []
    finished = asyncio.Event()

    @app.post("/audit/<user>")
    async def audit(request, user):
        async def straggler():
            await asyncio.sleep(0.03)
            try:
                Request.get_current()
                reads.append("leak")
            except ServerError:
                reads.append("denied")
            finished.set()

        asyncio.create_task(straggler())
        return response.text("ok")

    # asgi_client 会在类上 monkeypatch Sanic.__call__ 且不还原，残留后
    # 会让后续真正的 ASGI 服务器收不到 lifespan 事件。测试结束时还原。
    native_call = Sanic.__call__
    try:
        _, resp = await app.asgi_client.post("/audit/alice")
        assert resp.status == 200
        await finished.wait()
        assert reads == ["denied"]
    finally:
        Sanic.__call__ = native_call  # type: ignore[method-assign]
        app.asgi = False
