# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Tests for the in-run adaptive first-poll delay.

Polling a repository every ten seconds from t=0 when its checks reliably
take four minutes spends roughly twenty requests learning nothing. Once
one PR in a repository has shown how long its checks take, its siblings
can sleep most of that time before their first poll --- and because the
striped scheduler runs a repository's PRs one after another, the
observation always exists by the second PR.

``docs/BULK_RUN_PERFORMANCE_AUDIT.md`` ties this to the persistent
record in §4, which would carry the figure between runs. This is the
in-run half: it needs no storage and does not pre-commit that design.

The tests weight towards the cases where a head start must *not* be
taken, since sleeping through a resolution is the failure that costs a
merge.
"""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock

import pytest

from dependamerge.merge_manager import MergeResult, MergeStatus
from dependamerge.models import PullRequestInfo
from tests.conftest import make_merge_manager

REPO = "lfreleng-actions/slow-repo"


def _pr(mergeable_state: str = "blocked") -> PullRequestInfo:
    return PullRequestInfo(
        number=1,
        title="t",
        body=None,
        author="dependabot[bot]",
        head_sha="a" * 40,
        base_branch="main",
        head_branch="x",
        state="open",
        mergeable=True,
        mergeable_state=mergeable_state,
        behind_by=None,
        files_changed=[],
        repository_full_name=REPO,
        html_url=f"https://github.com/{REPO}/pull/1",
        reviews=[],
        review_comments=[],
    )


def _mgr(interval: float = 10.0):
    mgr, client = make_merge_manager()
    mgr._merge_recheck_interval = interval
    return mgr, client


class TestRecordWaitDuration:
    def test_records_a_positive_duration(self) -> None:
        mgr, _ = _mgr()
        mgr._record_wait_duration(REPO, 240.0)
        assert mgr._repo_wait_seconds[REPO] == [240.0]

    @pytest.mark.parametrize("value", [0.0, -1.0])
    def test_ignores_non_positive(self, value: float) -> None:
        mgr, _ = _mgr()
        mgr._record_wait_duration(REPO, value)
        assert REPO not in mgr._repo_wait_seconds


class TestHeadStart:
    def test_nothing_known_means_no_head_start(self) -> None:
        mgr, _ = _mgr()
        assert mgr._wait_head_start(REPO, budget=300.0) == 0.0

    def test_slow_repository_earns_a_head_start(self) -> None:
        mgr, _ = _mgr(interval=10.0)
        mgr._record_wait_duration(REPO, 240.0)

        head_start = mgr._wait_head_start(REPO, budget=300.0)

        # 80% of the observed median, capped at half the budget.
        assert head_start == pytest.approx(min(240.0 * 0.8, 150.0))

    def test_fast_repository_earns_none(self) -> None:
        """Below a few poll intervals the normal cadence is already cheap."""
        mgr, _ = _mgr(interval=10.0)
        mgr._record_wait_duration(REPO, 12.0)

        assert mgr._wait_head_start(REPO, budget=300.0) == 0.0

    def test_never_sleeps_more_than_half_the_budget(self) -> None:
        """A repository that has become faster must still be observed."""
        mgr, _ = _mgr(interval=10.0)
        mgr._record_wait_duration(REPO, 600.0)

        assert mgr._wait_head_start(REPO, budget=100.0) == pytest.approx(50.0)

    def test_head_start_is_never_negative(self) -> None:
        mgr, _ = _mgr(interval=10.0)
        mgr._record_wait_duration(REPO, 240.0)

        assert mgr._wait_head_start(REPO, budget=0.0) == 0.0

    def test_uses_the_median_not_the_worst_case(self) -> None:
        """One slow outlier must not strand every sibling behind it."""
        mgr, _ = _mgr(interval=10.0)
        for value in (60.0, 60.0, 900.0):
            mgr._record_wait_duration(REPO, value)

        # Median is 60, not 900.
        assert mgr._wait_head_start(REPO, budget=1000.0) == pytest.approx(48.0)

    def test_even_sample_count_averages_the_middle_pair(self) -> None:
        """A true median, not the upper middle.

        Taking ``ordered[len//2]`` would return 900 for ``[60, 900]``,
        letting a single outlier set the figure --- precisely what
        choosing the median is supposed to prevent.
        """
        mgr, _ = _mgr(interval=10.0)
        mgr._record_wait_duration(REPO, 60.0)
        mgr._record_wait_duration(REPO, 900.0)

        # median 480 -> 80% = 384, under half of a 2000s budget
        assert mgr._wait_head_start(REPO, budget=2000.0) == pytest.approx(384.0)

    def test_repositories_do_not_share_observations(self) -> None:
        mgr, _ = _mgr(interval=10.0)
        mgr._record_wait_duration(REPO, 240.0)

        assert mgr._wait_head_start("lfreleng-actions/other", budget=300.0) == 0.0

    def test_scales_with_the_configured_interval(self) -> None:
        """A slower cadence raises the bar for a head start to be worthwhile."""
        mgr, _ = _mgr(interval=60.0)
        mgr._record_wait_duration(REPO, 100.0)

        # 100s is under 3 x 60s, so no head start despite being slow.
        assert mgr._wait_head_start(REPO, budget=600.0) == 0.0


class TestHeadStartRespectsStopConditions:
    """A head start must never sleep through a state the caller wants.

    The conflict path calls back with ``stop_on_clean`` after a rebase
    may already have left the PR ``clean``. Sleeping first would burn up
    to half the shared deadline before the loop noticed it should return
    immediately.
    """

    @pytest.mark.asyncio
    async def test_skipped_when_already_clean_and_stopping_on_clean(self) -> None:
        mgr, _ = _mgr(interval=10.0)
        mgr._record_wait_duration(REPO, 240.0)
        pr = _pr(mergeable_state="clean")
        slept: list[float] = []

        async def _sleep(seconds: float) -> None:
            slept.append(seconds)

        import dependamerge.merge_manager as mod

        original = mod.asyncio.sleep
        mod.asyncio.sleep = _sleep  # type: ignore[assignment]
        try:
            await mgr._apply_wait_head_start(
                pr, "k", 300.0, ("blocked", "unstable"), True, True
            )
        finally:
            mod.asyncio.sleep = original  # type: ignore[assignment]

        assert slept == []

    @pytest.mark.asyncio
    async def test_skipped_when_state_is_outside_continue_states(self) -> None:
        mgr, _ = _mgr(interval=10.0)
        mgr._record_wait_duration(REPO, 240.0)
        pr = _pr(mergeable_state="dirty")
        slept: list[float] = []

        async def _sleep(seconds: float) -> None:
            slept.append(seconds)

        import dependamerge.merge_manager as mod

        original = mod.asyncio.sleep
        mod.asyncio.sleep = _sleep  # type: ignore[assignment]
        try:
            await mgr._apply_wait_head_start(
                pr, "k", 300.0, ("blocked", "unstable"), False, True
            )
        finally:
            mod.asyncio.sleep = original  # type: ignore[assignment]

        assert slept == []

    @pytest.mark.asyncio
    async def test_applied_when_the_state_is_one_being_waited_through(self) -> None:
        mgr, _ = _mgr(interval=10.0)
        mgr._record_wait_duration(REPO, 240.0)
        pr = _pr(mergeable_state="blocked")
        slept: list[float] = []

        async def _sleep(seconds: float) -> None:
            slept.append(seconds)

        import dependamerge.merge_manager as mod

        original = mod.asyncio.sleep
        mod.asyncio.sleep = _sleep  # type: ignore[assignment]
        try:
            await mgr._apply_wait_head_start(
                pr, "k", 300.0, ("blocked", "unstable"), False, True
            )
        finally:
            mod.asyncio.sleep = original  # type: ignore[assignment]

        assert slept and slept[0] > 0


class TestPerRunState:
    """Observations belong to a run, not to the manager instance.

    The manager supports reuse --- ``merge_prs_parallel`` resets the run
    deadline for exactly that reason, and a non-confirmed invocation runs
    the whole batch once as a preview before the real pass. Carrying
    observations across would let an earlier run's finding skip a
    workflow lookup, or size a head start from stale latency.
    """

    @pytest.mark.asyncio
    async def test_a_new_run_clears_observations(self) -> None:
        mgr, client = _mgr()
        mgr._repo_wait_seconds[REPO] = [240.0]
        mgr._semantic_title_aligned.add(f"{REPO}#1")
        mgr._last_merge_exception[f"{REPO}#1"] = Exception("not satisfied")
        mgr._last_merge_exception_head[f"{REPO}#1"] = "a" * 40

        await mgr.merge_prs_parallel([])

        assert mgr._repo_wait_seconds == {}
        assert mgr._semantic_title_aligned == set()
        # A rejection is run-scoped evidence too: the head it names can
        # be unchanged while the reason has moved on.
        assert mgr._last_merge_exception == {}
        assert mgr._last_merge_exception_head == {}
        # The block-reason memo lives on the client but is run-scoped
        # for the same reason, and its window can outlast the gap
        # between two runs.
        client.clear_block_reasons.assert_called_once()

    @pytest.mark.asyncio
    async def test_an_empty_batch_clears_the_previous_run_results(self) -> None:
        """A batch filtered down to nothing must not report the last run.

        ``_results`` is run-scoped exactly like the observations above,
        and it is what ``get_results_summary``, ``get_failed_prs`` and
        ``get_successful_prs`` read.  Reuse is explicitly supported, so a
        caller that filters a batch to empty --- everything already
        merged, or everything excluded --- would otherwise be shown the
        *previous* batch's outcomes as though they belonged to this run.
        """
        mgr, _ = _mgr()
        mgr._results = [
            MergeResult(pr_info=_pr(), status=MergeStatus.FAILED, error="boom"),
            MergeResult(pr_info=_pr(), status=MergeStatus.MERGED),
        ]

        assert await mgr.merge_prs_parallel([]) == []

        assert mgr._results == []
        assert mgr.get_failed_prs() == []
        assert mgr.get_successful_prs() == []
        assert mgr.get_results_summary()["total"] == 0


class TestTheClockStopsBeforeTheSlotIsReacquired:
    """Scheduler queue time is not check latency.

    The poll loop runs inside ``parked()``, which releases this worker's
    concurrency slot. Leaving that block re-acquires it, and on a busy
    run that acquire queues behind other work. Sampling the clock after
    the block charges the queue to the repository's checks, and every
    sibling is then taught to sleep through a delay that belongs to the
    scheduler rather than to CI.
    """

    @pytest.mark.asyncio
    async def test_slot_contention_is_not_counted(self, monkeypatch) -> None:
        import dependamerge.merge_manager as mod

        mgr, _ = _mgr()
        pr = _pr(mergeable_state="blocked")
        clock = {"now": 1000.0}

        @contextlib.asynccontextmanager
        async def _slow_to_reacquire():
            try:
                yield
            finally:
                # Stand in for ``lease.acquire()`` queueing behind other
                # workers on the way out of the park.
                clock["now"] += 500.0

        async def _advance(_seconds: float) -> None:
            # Thirty seconds of genuine check latency per poll, without
            # spending it: the fake clock is what the code reads.
            clock["now"] += 30.0

        monkeypatch.setattr(mod, "parked", _slow_to_reacquire)
        monkeypatch.setattr(mod.asyncio, "sleep", _advance)
        loop = asyncio.get_running_loop()
        monkeypatch.setattr(loop, "time", lambda: clock["now"])
        mgr._fetch_pr_state = AsyncMock(  # type: ignore[method-assign]
            return_value={"mergeable_state": "clean", "mergeable": True}
        )

        await mgr._wait_for_auto_merge(
            pr,
            "o",
            "r",
            continue_states=("blocked", "unknown", ""),
            deadline=clock["now"] + 100_000.0,
            measures_checks=True,
        )

        # The checks took 30 s; the 500 s re-acquire belongs to the
        # scheduler and must not reach the repository's median.
        assert mgr._repo_wait_seconds == {REPO: [30.0]}

    @pytest.mark.asyncio
    async def test_a_wait_that_never_polled_records_nothing(self) -> None:
        """Entering already ``clean`` measured no checks at all.

        The required-workflow retry loop re-enters this method after
        every attempt, and a stale ``clean`` snapshot exits it before a
        single poll. Recording those near-zero samples would drag the
        median to nothing and quietly disable the head start for every
        sibling in the repository.
        """
        mgr, _ = _mgr()

        await mgr._wait_for_auto_merge(
            _pr(mergeable_state="clean"),
            "o",
            "r",
            continue_states=("blocked", "unknown", ""),
            measures_checks=True,
        )

        assert mgr._repo_wait_seconds == {}

    @pytest.mark.asyncio
    async def test_a_terminal_state_exit_records_nothing(self, monkeypatch) -> None:
        """Turning ``dirty`` did not time a check run.

        Such exits tend to come in seconds, so recording them as check
        latency would pull the median down and disable the head start
        just as surely as a zero-poll sample.
        """
        import dependamerge.merge_manager as mod

        mgr, _ = _mgr()
        clock = {"now": 1000.0}

        async def _advance(_seconds: float) -> None:
            clock["now"] += 5.0

        monkeypatch.setattr(mod.asyncio, "sleep", _advance)
        loop = asyncio.get_running_loop()
        monkeypatch.setattr(loop, "time", lambda: clock["now"])
        mgr._fetch_pr_state = AsyncMock(  # type: ignore[method-assign]
            return_value={"mergeable_state": "dirty", "mergeable": False}
        )

        await mgr._wait_for_auto_merge(
            _pr(mergeable_state="blocked"),
            "o",
            "r",
            continue_states=("blocked", "unknown", ""),
            deadline=clock["now"] + 100_000.0,
            measures_checks=True,
        )

        assert mgr._repo_wait_seconds == {}


class TestOnlyCheckWaitsAreRecorded:
    """A rebase turnaround is not check latency.

    ``_wait_for_auto_merge`` also waits for dependabot rebases and for an
    armed auto-merge to close. Recording those would let the conflict
    path's rebase wait hand the same PR's next phase a head start worth
    half its remaining budget.
    """

    @pytest.mark.asyncio
    async def test_unmarked_wait_is_not_recorded(self) -> None:
        mgr, client = make_merge_manager()
        mgr._no_wait = True  # return immediately; we only assert recording
        pr = _pr()

        await mgr._wait_for_auto_merge(
            pr, "o", "r", continue_states=("dirty", "unknown", "")
        )

        assert mgr._repo_wait_seconds == {}

    @pytest.mark.asyncio
    async def test_the_rebase_wait_call_site_is_unmarked(self) -> None:
        """Guards the wiring, not just the parameter.

        ``_handle_merge_conflict`` waits for a rebase with
        ``continue_states=("dirty", "unknown", "")``; that call must not
        pass ``measures_checks``.
        """
        import inspect

        from dependamerge.merge_manager import AsyncMergeManager

        source = inspect.getsource(AsyncMergeManager._handle_merge_conflict)
        rebase_wait = source.split('continue_states=("dirty"', 1)[1].split(")", 2)[1]
        assert "measures_checks" not in rebase_wait


class TestTheConflictPathsCheckWait:
    """Only the wait that stops at ``clean`` measures check latency.

    Once a rebase clears the conflict, ``_handle_merge_conflict`` waits
    again --- but for two different things depending on whether
    auto-merge could be armed. Without it the wait stops at ``clean``
    and so times the checks. With it armed the wait deliberately runs
    *through* ``clean`` until GitHub closes the PR, so the duration also
    carries however long the merge queue took. Recording that figure
    would hand every sibling in the repository a head start sized from
    a queue wait they will not have.
    """

    @staticmethod
    async def _conflict_path(auto_ok: bool) -> list[bool]:
        """Drive the handler past the rebase; return each wait's flag."""
        from unittest.mock import AsyncMock, patch

        from dependamerge.merge_manager import MergeResult, MergeStatus

        mgr, _ = _mgr()
        pr = _pr(mergeable_state="dirty")
        # Skips the ``@dependabot rebase`` macro: the branch is already
        # being rebased, which is not what this test is about.
        pr.body = "Dependabot is rebasing this PR"
        marks: list[bool] = []

        async def _wait(*_args: object, **kwargs: object) -> tuple[bool, bool]:
            marks.append(bool(kwargs.get("measures_checks", False)))
            # The first call is the rebase wait; clearing the conflict
            # lets the handler reach the check wait under test.
            pr.mergeable_state = "blocked"
            return False, False

        with (
            patch.object(mgr, "_wait_for_auto_merge", side_effect=_wait),
            patch.object(mgr, "_approve_pr", new_callable=AsyncMock),
            patch.object(
                mgr,
                "_enable_auto_merge_for_pr",
                new_callable=AsyncMock,
                return_value=auto_ok,
            ),
        ):
            await mgr._handle_merge_conflict(
                pr,
                "lfreleng-actions",
                "slow-repo",
                MergeResult(pr, MergeStatus.PENDING),
            )

        return marks

    @pytest.mark.asyncio
    async def test_the_check_wait_is_measured_without_auto_merge(self) -> None:
        rebase_wait, check_wait = await self._conflict_path(auto_ok=False)

        assert rebase_wait is False
        assert check_wait is True

    @pytest.mark.asyncio
    async def test_waiting_through_clean_is_not_measured(self) -> None:
        rebase_wait, check_wait = await self._conflict_path(auto_ok=True)

        assert rebase_wait is False
        assert check_wait is False


class TestHeadStartOnlyForCheckWaits:
    """The figure measures checks, so only check waits may spend it.

    A rebase often lands in seconds; a head start sized from check
    latency could sleep clean through one.
    """

    @pytest.mark.asyncio
    async def test_not_applied_to_a_wait_that_does_not_measure_checks(self) -> None:
        mgr, _ = _mgr(interval=10.0)
        mgr._record_wait_duration(REPO, 240.0)
        pr = _pr(mergeable_state="dirty")
        slept: list[float] = []

        async def _sleep(seconds: float) -> None:
            slept.append(seconds)

        import dependamerge.merge_manager as mod

        original = mod.asyncio.sleep
        mod.asyncio.sleep = _sleep  # type: ignore[assignment]
        try:
            await mgr._apply_wait_head_start(
                pr, "k", 300.0, ("dirty", "unknown", ""), False, False
            )
        finally:
            mod.asyncio.sleep = original  # type: ignore[assignment]

        assert slept == []
