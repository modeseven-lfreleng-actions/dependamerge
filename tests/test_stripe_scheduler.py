# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Tests for the owner-wide striped merge scheduler.

The load-bearing guarantee of the striped scheduler is that **no two PRs
from the same repository are ever in flight simultaneously**, while
distinct repositories are still processed concurrently.  These tests
assert that invariant with a deterministic fake that records the set of
in-flight repositories at every dispatch.
"""

import asyncio

import pytest

from dependamerge.merge_manager import (
    AsyncMergeManager,
    MergeResult,
    MergeStatus,
)
from dependamerge.models import ComparisonResult, PullRequestInfo

# Work-item type accepted by ``merge_prs_parallel``.  Annotating the test
# fixtures with this (rather than letting the literal narrow to
# ``tuple[PullRequestInfo, None]``) keeps the invariant ``list`` element
# type assignable to the method's parameter.
PRPair = tuple[PullRequestInfo, ComparisonResult | None]


def _make_pr(number: int, repo: str) -> PullRequestInfo:
    """Build a minimal PullRequestInfo for scheduler testing."""
    return PullRequestInfo(
        number=number,
        title=f"Bump dep in {repo} #{number}",
        body="Automated dependency update",
        author="dependabot[bot]",
        head_sha="abc123",
        base_branch="main",
        head_branch=f"dependabot/{repo}/{number}",
        state="open",
        mergeable=True,
        mergeable_state="clean",
        behind_by=0,
        files_changed=[],
        repository_full_name=repo,
        html_url=f"https://github.com/{repo}/pull/{number}",
    )


class _Recorder:
    """Instrumented replacement for ``_merge_single_pr``.

    Records, for every PR dispatched, the set of repositories that were
    in flight at that moment, so tests can assert the per-repo
    single-flight invariant and observe cross-repo concurrency.
    """

    def __init__(self, hold: float = 0.02) -> None:
        self._hold = hold
        self.inflight_by_repo: dict[str, int] = {}
        self.max_concurrent_per_repo: dict[str, int] = {}
        self.max_global_inflight = 0
        self.dispatch_order: list[tuple[str, int]] = []

    async def __call__(self, pr_info: PullRequestInfo) -> MergeResult:
        repo = pr_info.repository_full_name
        self.inflight_by_repo[repo] = self.inflight_by_repo.get(repo, 0) + 1
        self.max_concurrent_per_repo[repo] = max(
            self.max_concurrent_per_repo.get(repo, 0),
            self.inflight_by_repo[repo],
        )
        self.max_global_inflight = max(
            self.max_global_inflight,
            sum(self.inflight_by_repo.values()),
        )
        self.dispatch_order.append((repo, pr_info.number))
        # Hold the "in flight" state long enough that any scheduling
        # violation (two PRs of one repo, or more than `concurrency`
        # repos at once) would overlap and be observed.
        await asyncio.sleep(self._hold)
        self.inflight_by_repo[repo] -= 1
        return MergeResult(pr_info=pr_info, status=MergeStatus.MERGED)


def _manager(concurrency: int = 10) -> AsyncMergeManager:
    return AsyncMergeManager(
        token="test_token",
        concurrency=concurrency,
        preview_mode=True,
    )


@pytest.mark.asyncio
async def test_striped_single_flight_per_repo():
    """No repository ever has more than one PR in flight at a time."""
    pr_list: list[PRPair] = [
        (_make_pr(1, "owner/a"), None),
        (_make_pr(2, "owner/a"), None),
        (_make_pr(3, "owner/a"), None),
        (_make_pr(1, "owner/b"), None),
        (_make_pr(2, "owner/b"), None),
        (_make_pr(1, "owner/c"), None),
    ]
    mgr = _manager(concurrency=10)
    recorder = _Recorder()
    mgr._merge_single_pr = recorder  # type: ignore[assignment]

    results = await mgr.merge_prs_parallel(pr_list, stripe=True)

    # Load-bearing invariant: at most one in-flight PR per repository.
    assert all(count == 1 for count in recorder.max_concurrent_per_repo.values()), (
        recorder.max_concurrent_per_repo
    )
    # Every PR produced a result.
    assert len(results) == len(pr_list)
    assert all(r.status is MergeStatus.MERGED for r in results)


@pytest.mark.asyncio
async def test_striped_runs_distinct_repos_concurrently():
    """Distinct repositories ARE processed concurrently (not serialised)."""
    pr_list: list[PRPair] = [
        (_make_pr(1, "owner/a"), None),
        (_make_pr(1, "owner/b"), None),
        (_make_pr(1, "owner/c"), None),
        (_make_pr(2, "owner/a"), None),
        (_make_pr(2, "owner/b"), None),
        (_make_pr(2, "owner/c"), None),
    ]
    mgr = _manager(concurrency=10)
    recorder = _Recorder()
    mgr._merge_single_pr = recorder  # type: ignore[assignment]

    await mgr.merge_prs_parallel(pr_list, stripe=True)

    # Three distinct repos with available concurrency should overlap.
    assert recorder.max_global_inflight >= 2
    # But still never two PRs of the same repo at once.
    assert all(c == 1 for c in recorder.max_concurrent_per_repo.values())


@pytest.mark.asyncio
async def test_striped_respects_global_concurrency_bound():
    """Global in-flight never exceeds the configured concurrency."""
    pr_list: list[PRPair] = [(_make_pr(1, f"owner/r{i}"), None) for i in range(8)]
    mgr = _manager(concurrency=3)
    recorder = _Recorder()
    mgr._merge_single_pr = recorder  # type: ignore[assignment]

    await mgr.merge_prs_parallel(pr_list, stripe=True)

    assert recorder.max_global_inflight <= 3, recorder.max_global_inflight


@pytest.mark.asyncio
async def test_striped_processes_repo_prs_in_order():
    """Within a repository, PRs are dispatched in first-seen order."""
    pr_list: list[PRPair] = [
        (_make_pr(10, "owner/a"), None),
        (_make_pr(20, "owner/a"), None),
        (_make_pr(30, "owner/a"), None),
    ]
    mgr = _manager(concurrency=5)
    recorder = _Recorder()
    mgr._merge_single_pr = recorder  # type: ignore[assignment]

    await mgr.merge_prs_parallel(pr_list, stripe=True)

    a_order = [num for repo, num in recorder.dispatch_order if repo == "owner/a"]
    assert a_order == [10, 20, 30]


@pytest.mark.asyncio
async def test_striped_preserves_result_ordering():
    """Results are returned in the caller's original input order."""
    pr_list: list[PRPair] = [
        (_make_pr(1, "owner/a"), None),
        (_make_pr(1, "owner/b"), None),
        (_make_pr(2, "owner/a"), None),
        (_make_pr(1, "owner/c"), None),
    ]
    mgr = _manager(concurrency=10)
    recorder = _Recorder()
    mgr._merge_single_pr = recorder  # type: ignore[assignment]

    results = await mgr.merge_prs_parallel(pr_list, stripe=True)

    returned = [(r.pr_info.repository_full_name, r.pr_info.number) for r in results]
    expected = [(pr.repository_full_name, pr.number) for pr, _ in pr_list]
    assert returned == expected


@pytest.mark.asyncio
async def test_striped_isolates_per_pr_failures():
    """A crash on one PR does not lose results for siblings in its repo."""
    pr_list: list[PRPair] = [
        (_make_pr(1, "owner/a"), None),
        (_make_pr(2, "owner/a"), None),
        (_make_pr(3, "owner/a"), None),
    ]
    mgr = _manager(concurrency=5)

    async def flaky(pr_info: PullRequestInfo) -> MergeResult:
        if pr_info.number == 2:
            raise RuntimeError("boom")
        return MergeResult(pr_info=pr_info, status=MergeStatus.MERGED)

    mgr._merge_single_pr = flaky  # type: ignore[assignment]

    results = await mgr.merge_prs_parallel(pr_list, stripe=True)

    assert len(results) == 3
    by_number = {r.pr_info.number: r for r in results}
    assert by_number[1].status is MergeStatus.MERGED
    assert by_number[2].status is MergeStatus.FAILED
    assert by_number[2].error and "boom" in by_number[2].error
    # PR #3 still ran even though #2 (earlier in the same repo) failed.
    assert by_number[3].status is MergeStatus.MERGED


@pytest.mark.asyncio
async def test_striped_handles_duplicate_work_item_objects():
    """Reused tuple objects do not collapse into a single result.

    Results are keyed by input position, not ``id(item)``, so passing the
    same tuple object multiple times must still yield one result per slot
    rather than overwriting earlier entries.
    """
    shared: PRPair = (_make_pr(1, "owner/a"), None)
    pr_list: list[PRPair] = [shared, shared, shared]
    mgr = _manager(concurrency=5)
    recorder = _Recorder()
    mgr._merge_single_pr = recorder  # type: ignore[assignment]

    results = await mgr.merge_prs_parallel(pr_list, stripe=True)

    # One result per input slot, despite the shared tuple identity.
    assert len(results) == len(pr_list)
    assert all(r.status is MergeStatus.MERGED for r in results)
    # The repository's three (identical) PRs were each dispatched.
    assert len(recorder.dispatch_order) == 3


class TestCancellationIsNotAResult:
    """Cancellation must tear the run down, not enter the results list.

    ``asyncio.CancelledError`` derives from ``BaseException`` rather than
    ``Exception``, so it misses the ``isinstance(result, Exception)``
    branch in ``_collect_results``.  The ``cast`` that follows is a no-op
    at runtime, so without an explicit guard the exception object itself
    is appended to the results list and only fails later --- and further
    away --- when a caller touches ``.status``, with nothing connecting
    the error to the PR that was cancelled.

    The two schedulers must also agree here: ``_run_striped`` has always
    re-raised cancellation explicitly, so interrupting an owner-wide run
    and interrupting a single-repository run should fail the same way.
    """

    @pytest.mark.asyncio
    async def test_flat_propagates_cancellation(self) -> None:
        pr_list: list[PRPair] = [
            (_make_pr(1, "owner/a"), None),
            (_make_pr(2, "owner/b"), None),
        ]
        mgr = _manager(concurrency=5)

        async def _cancel_one(pr_info: PullRequestInfo) -> MergeResult:
            if pr_info.repository_full_name == "owner/b":
                raise asyncio.CancelledError
            return MergeResult(pr_info=pr_info, status=MergeStatus.MERGED)

        mgr._merge_single_pr = _cancel_one  # type: ignore[assignment]

        with pytest.raises(asyncio.CancelledError):
            await mgr.merge_prs_parallel(pr_list)

    @pytest.mark.asyncio
    async def test_striped_propagates_cancellation(self) -> None:
        """The pre-existing behaviour the flat path now matches."""
        pr_list: list[PRPair] = [
            (_make_pr(1, "owner/a"), None),
            (_make_pr(2, "owner/b"), None),
        ]
        mgr = _manager(concurrency=5)

        async def _cancel_one(pr_info: PullRequestInfo) -> MergeResult:
            if pr_info.repository_full_name == "owner/b":
                raise asyncio.CancelledError
            return MergeResult(pr_info=pr_info, status=MergeStatus.MERGED)

        mgr._merge_single_pr = _cancel_one  # type: ignore[assignment]

        with pytest.raises(asyncio.CancelledError):
            await mgr.merge_prs_parallel(pr_list, stripe=True)

    @pytest.mark.asyncio
    async def test_flat_still_converts_ordinary_failures_in_order(self) -> None:
        """Narrowing the guard must not disturb ordinary per-PR failures."""
        pr_list: list[PRPair] = [
            (_make_pr(1, "owner/a"), None),
            (_make_pr(2, "owner/b"), None),
            (_make_pr(3, "owner/c"), None),
        ]
        mgr = _manager(concurrency=5)

        async def _fail_middle(pr_info: PullRequestInfo) -> MergeResult:
            if pr_info.number == 2:
                raise RuntimeError("boom")
            return MergeResult(pr_info=pr_info, status=MergeStatus.MERGED)

        mgr._merge_single_pr = _fail_middle  # type: ignore[assignment]

        results = await mgr.merge_prs_parallel(pr_list)

        assert [r.pr_info.number for r in results] == [1, 2, 3]
        assert results[1].status is MergeStatus.FAILED
        assert results[1].error and "boom" in results[1].error
        assert results[0].status is MergeStatus.MERGED
        assert results[2].status is MergeStatus.MERGED


@pytest.mark.asyncio
async def test_run_deadline_reset_between_runs_on_reused_manager():
    """A reused manager must not carry a stale ``_run_deadline``.

    A previous owner-wide run with ``max_wait`` sets ``_run_deadline``.
    A subsequent run without a ceiling (``max_wait=None``) must clear it
    so its per-PR waits are not clamped (or instantly expired) by the
    earlier run's deadline.
    """
    pr_list: list[PRPair] = [(_make_pr(1, "owner/a"), None)]

    # First run with a wall-clock ceiling sets a deadline.
    mgr = AsyncMergeManager(
        token="test_token",
        concurrency=5,
        preview_mode=True,
        max_wait=900.0,
    )
    mgr._merge_single_pr = _Recorder()  # type: ignore[assignment]
    await mgr.merge_prs_parallel(pr_list, stripe=True)
    assert mgr._run_deadline is not None

    # Simulate the manager being reconfigured for an uncapped run.
    mgr._max_wait = None
    mgr._merge_single_pr = _Recorder()  # type: ignore[assignment]
    await mgr.merge_prs_parallel(pr_list, stripe=True)

    # The stale deadline from the first run was cleared.
    assert mgr._run_deadline is None


@pytest.mark.asyncio
async def test_no_wait_flag_reset_between_runs_on_reused_manager():
    """A reused manager must not carry a stale ``_no_wait`` flag.

    ``_no_wait`` (the fire-and-forget ``max_wait == 0`` mode) is derived
    from ``_max_wait`` in ``__init__`` but must be recomputed at the
    start of every run: if the same instance is later reconfigured for a
    blocking run (``max_wait=None`` or ``> 0``), a stale ``_no_wait=True``
    would otherwise wrongly skip all waits.
    """
    pr_list: list[PRPair] = [(_make_pr(1, "owner/a"), None)]

    # First run in fire-and-forget mode sets ``_no_wait`` True.
    mgr = AsyncMergeManager(
        token="test_token",
        concurrency=5,
        preview_mode=True,
        max_wait=0,
    )
    mgr._merge_single_pr = _Recorder()  # type: ignore[assignment]
    await mgr.merge_prs_parallel(pr_list, stripe=True)
    assert mgr._no_wait is True

    # Reconfigure for an uncapped (blocking) run.
    mgr._max_wait = None
    mgr._merge_single_pr = _Recorder()  # type: ignore[assignment]
    await mgr.merge_prs_parallel(pr_list, stripe=True)

    # The stale fire-and-forget flag from the first run was cleared.
    assert mgr._no_wait is False
