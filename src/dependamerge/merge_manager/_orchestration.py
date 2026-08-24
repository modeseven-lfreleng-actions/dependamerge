# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""
Top-level orchestration of a parallel merge run.

The flat and striped schedulers, the semaphore that bounds
concurrency, and the per-repository dispatch lock that serialises
the moment of merge within a repository.
"""

from __future__ import annotations

import asyncio

from ..models import ComparisonResult, PullRequestInfo
from ..slot_lease import holding_slot
from ._base import _MergeManagerBase
from ._types import MergeResult, MergeStatus


class _OrchestrationMixin(_MergeManagerBase):
    """Scheduling and dispatch for a whole merge run."""

    async def merge_prs_parallel(
        self,
        pr_list: list[tuple[PullRequestInfo, ComparisonResult | None]],
        *,
        stripe: bool = False,
    ) -> list[MergeResult]:
        """
        Merge multiple PRs in parallel.

        Args:
            pr_list: List of ``(PullRequestInfo, ComparisonResult | None)``
                tuples.  The comparison element is ``None`` for owner-wide
                and repository-wide runs (no source PR to compare
                against) and a ``ComparisonResult`` for similar-PR runs.
            stripe: When True, schedule the batch with the striped
                scheduler (see :meth:`_run_striped`): PRs are grouped by
                repository and at most one PR per repository is processed
                at a time, while distinct repositories run concurrently.
                Used for owner-wide runs where the flat list mixes many
                repositories and back-to-back same-repo merges would race
                GitHub's mergeability propagation.  When False (default)
                the flat scheduler is used (one task per PR), suitable for
                single-PR and single-repository batches.

        Returns:
            List of MergeResult objects with operation results
        """
        # Per-run observations, cleared before anything else so the
        # invariant is simply "a run starts with none".  This manager
        # supports reuse, and a non-confirmed invocation runs the whole
        # batch once as a preview before the real pass.  Carrying them
        # over would let an earlier run's finding skip a workflow lookup
        # entirely, or size a head start from stale latency.
        #
        # A stored rejection is run-scoped evidence for the same reason.
        # The head it was raised against can be unchanged while the
        # *reason* has moved on, so an earlier run's "not satisfied"
        # could send a fresh failure down the undispatched-workflow path
        # on a commit whose workflows have since run.
        self._repo_wait_seconds.clear()
        self._semantic_title_aligned.clear()
        self._last_merge_exception.clear()
        self._last_merge_exception_head.clear()
        # The block-reason memo lives on the client rather than here,
        # but it is run-scoped for the same reason and its expiry window
        # can outlast the gap between two runs.
        if self._github_client is not None:
            self._github_client.clear_block_reasons()

        # The accumulated results are run-scoped for the same reason, and
        # are read back by ``get_results_summary``, ``get_failed_prs`` and
        # ``get_successful_prs``.  Clearing them here rather than only on
        # the normal path means a batch that filters down to nothing ---
        # everything already merged, or everything excluded --- reports an
        # empty run instead of the previous run's outcomes.
        self._results = []

        # Resolve the owner-wide global wait ceiling for this run.  A
        # positive ``max_wait`` becomes a monotonic wall-clock deadline
        # that every per-PR wait is clamped to (see
        # ``_wait_for_auto_merge``); ``0`` (``_no_wait``) skips waiting
        # entirely; ``None`` leaves each per-PR ``merge_timeout``
        # uncapped (repository / similar-PR runs).  Reset both first so a
        # reused manager instance never carries a stale deadline — or a
        # stale ``_no_wait`` flag — from a previous run into this one
        # (``_max_wait`` may differ between runs on the same instance).
        self._run_deadline = None
        self._no_wait = self._max_wait is not None and self._max_wait <= 0
        if self._max_wait is not None and self._max_wait > 0:
            self._run_deadline = asyncio.get_running_loop().time() + self._max_wait

        # Every piece of run-scoped state is now reset, so an empty batch
        # leaves the manager in the same condition a real run would.  The
        # guard sits here rather than earlier because nothing above needs
        # a PR, while everything below does --- the org-approval lookup
        # reads ``pr_list[0]``.
        if not pr_list:
            return []

        if self.preview_mode:
            self.log.info(f"🔍 PREVIEW: Would merge {len(pr_list)} PRs")
        else:
            self.log.debug(f"Starting parallel merge of {len(pr_list)} PRs")
            # Enumerate the org's approval requirement once, up-front, so
            # the "organization mandates an approving review" line is shown
            # before merging begins rather than mid-run.  All PRs in a run
            # share the same org owner; the result is cached and reused by
            # the per-PR proactive-approval check.
            owner = pr_list[0][0].repository_full_name.split("/", 1)[0]
            await self._org_approval_rulesets(owner)

        # Background ticker that surfaces a single-line countdown
        # whenever one or more workers are waiting for required
        # checks to complete (auto-merge wait loop). The countdown
        # uses the worst-case (latest) deadline across all waiting
        # PRs so the user sees the longest remaining wait.
        ticker_task = asyncio.create_task(
            self._wait_status_ticker(),
            name="merge-wait-ticker",
        )

        try:
            if stripe:
                final_results = await self._run_striped(pr_list)
            else:
                final_results = await self._run_flat(pr_list)
        finally:
            ticker_task.cancel()
            try:
                await ticker_task
            except asyncio.CancelledError:
                # Expected during normal shutdown.
                pass
            except Exception as ticker_exc:
                # Unexpected: log so we can debug ticker crashes
                # without swallowing them silently. The merge run
                # itself has already completed at this point, so
                # we still continue to results processing.
                self.log.warning(
                    "wait-status ticker exited unexpectedly: %s",
                    ticker_exc,
                    exc_info=True,
                )

        self._results = final_results
        return final_results

    async def _run_flat(
        self,
        pr_list: list[tuple[PullRequestInfo, ComparisonResult | None]],
    ) -> list[MergeResult]:
        """Flat scheduler: one task per PR, bounded by the merge semaphore.

        Suitable for single-PR and single-repository batches where there
        is no need to avoid stacking same-repo merges (the per-repo merge
        dispatch lock already serialises the final API call).
        """
        tasks = []
        for pr_info, _comparison in pr_list:
            task = asyncio.create_task(
                self._merge_single_pr_with_semaphore(pr_info),
                name=f"merge-{pr_info.repository_full_name}#{pr_info.number}",
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        return self._collect_results(pr_list, results)

    async def _run_striped(
        self,
        pr_list: list[tuple[PullRequestInfo, ComparisonResult | None]],
    ) -> list[MergeResult]:
        """Striped scheduler: one serial worker per repository.

        Owner-wide batches frequently contain several PRs targeting the
        same repository (e.g. dependabot and pre-commit-ci both opening
        PRs in one repo).  Merging two PRs in the same repository
        back-to-back races GitHub's branch-protection / mergeability
        propagation.  To avoid that structurally (no injected delays, no
        random retries) this scheduler:

        - Groups PRs by repository, preserving first-seen order.
        - Runs one worker coroutine per repository that processes that
          repository's PRs strictly sequentially, so **at most one PR per
          repository is ever in flight**.
        - Lets distinct repositories run concurrently, bounded by the
          shared merge semaphore.  When a repository releases its slot
          between PRs it re-acquires the semaphore for its next PR,
          competing afresh with other repositories' waiting PRs.  In
          practice CPython wakes semaphore waiters in roughly the order
          they blocked, so this tends to round-robin ("stripe") work
          across repositories — but that ordering is only a best-effort
          optimisation, not a correctness property.  Fairness is **not**
          part of the public ``asyncio.Semaphore`` contract and is not
          relied upon: the single-flight-per-repository invariant above
          comes solely from each repository's serial worker, regardless
          of the order in which the semaphore admits waiters.

        Combined with ``repo_scoped`` mergeability refresh, when a
        repository's second PR finally starts it re-reads state the first
        PR's merge may have invalidated.
        """
        # Group by repository, preserving first-seen order so the stripe
        # ordering is deterministic.  Each item is carried with its index
        # in ``pr_list`` so results reassemble in the caller's order.
        grouped: dict[
            str, list[tuple[int, tuple[PullRequestInfo, ComparisonResult | None]]]
        ] = {}
        for index, item in enumerate(pr_list):
            grouped.setdefault(item[0].repository_full_name, []).append((index, item))

        # Results are keyed by each work item's position in ``pr_list``.
        # Positional keys (rather than ``id(item)``) stay correct even if
        # the caller passes duplicate tuple objects (e.g. ``[item] * n``),
        # where ``id()`` would collide and later results would overwrite
        # earlier ones.
        result_by_index: dict[int, MergeResult] = {}

        async def _repo_worker(
            items: list[tuple[int, tuple[PullRequestInfo, ComparisonResult | None]]],
        ) -> None:
            for index, item in items:
                pr_info = item[0]
                try:
                    res = await self._merge_single_pr_with_semaphore(pr_info)
                except asyncio.CancelledError:
                    # Cancellation must propagate so the gather below can
                    # tear the run down.  On Python >= 3.10 CancelledError
                    # already derives from BaseException (not Exception),
                    # so the handler below would not catch it; this
                    # explicit re-raise documents that intent and guards
                    # against the broad handler ever being widened.
                    raise
                except Exception as e:
                    # Defensive: a crash on one PR must not lose results
                    # for the remaining PRs in the same repository.
                    res = MergeResult(
                        pr_info=pr_info,
                        status=MergeStatus.FAILED,
                        error=str(e),
                    )
                    # The exception escaped before the semaphore
                    # wrapper could record a terminal outcome.
                    self._record_terminal_outcome(pr_info, MergeStatus.FAILED)
                    self.log.error(
                        "Unexpected error merging PR %s#%s: %s",
                        pr_info.repository_full_name,
                        pr_info.number,
                        e,
                    )
                result_by_index[index] = res

        tasks = [
            asyncio.create_task(
                _repo_worker(items),
                name=f"merge-repo-{repo}",
            )
            for repo, items in grouped.items()
        ]

        # Workers swallow every per-PR exception (each PR is wrapped
        # defensively above), so the only thing they can propagate is
        # ``asyncio.CancelledError`` on shutdown — which ``gather``
        # re-raises to tear the whole run down.
        await asyncio.gather(*tasks)

        return [result_by_index[i] for i in range(len(pr_list))]

    async def _merge_single_pr_with_semaphore(
        self, pr_info: PullRequestInfo
    ) -> MergeResult:
        """Merge a single PR with concurrency control.

        The slot is leased, not pinned: any wait loop inside
        ``_merge_single_pr`` that wraps itself in ``parked()`` (the
        auto-merge wait, post-rebase polls, recreate waits, …)
        releases the slot for the duration of the wait and re-acquires
        it before resuming active work, so PRs waiting on external
        events (dependabot rebases, CI) never starve runnable PRs.
        See ``slot_lease.py`` and ``docs/MERGE_ENGINE_DESIGN.md``.
        """
        async with holding_slot(self._merge_semaphore):
            result = await self._merge_single_pr(pr_info)
            # Single terminal-accounting point: map the result status
            # onto the tracker counters (see _record_terminal_outcome).
            # Uses the *original* pr_info so the transitory state keyed
            # on it is cleared even when the result carries a
            # recreated PR.
            self._record_terminal_outcome(pr_info, result.status)
            return result

    async def _merge_single_pr(self, pr_info: PullRequestInfo) -> MergeResult:
        """Merge a single PR, then confirm any failure is real.

        Thin wrapper over :meth:`_merge_single_pr_impl`.  It exists
        because a reported failure is frequently not one: GitHub
        auto-merge routinely completes a PR moments after this tool
        stops waiting for it.  In the 503-PR run analysed in
        ``docs/BULK_RUN_PERFORMANCE_AUDIT.md``, **21 of the 34 reported
        failures had in fact merged**, most within two minutes of being
        reported.

        Wrapping rather than editing the end of ``_merge_single_pr_impl``
        is deliberate: that method has several early ``return`` paths
        (permission denied, already merged, conflict handling), and a
        check placed before its final ``return`` would miss them.
        """
        result = await self._merge_single_pr_impl(pr_info)
        return await self._confirm_failure(pr_info, result)

    async def _get_merge_dispatch_lock(self, owner: str, repo: str) -> asyncio.Lock:
        """Return the ``asyncio.Lock`` that serialises merge dispatch for ``owner/repo``.

        The lock is created lazily on first request and shared by
        every worker targeting the same repository.  Workers
        targeting different repositories receive distinct locks and
        can dispatch in parallel.

        Holding this lock around the actual ``merge_pull_request``
        API call (and its retry loop) prevents back-to-back merges
        on the same repo from racing GitHub's branch-protection
        propagation, while leaving every other phase of the merge
        flow — approve, rebase polling, Step 5.5's auto-merge wait —
        free to run in parallel across workers.
        """
        key = f"{owner}/{repo}"
        async with self._merge_dispatch_locks_lock:
            lock = self._merge_dispatch_locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._merge_dispatch_locks[key] = lock
            return lock
