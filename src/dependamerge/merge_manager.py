# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

from __future__ import annotations

import asyncio
import fnmatch
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote

from rich.console import Console

from . import rebase
from .bot_identity import is_dependabot
from .copilot_handler import CopilotCommentHandler
from .gerrit import (
    GerritAuthError,
    GerritRestError,
    create_gerrit_service,
    create_submit_manager,
)
from .github2gerrit_detector import (
    GitHub2GerritDetectionResult,
    GitHub2GerritMapping,
    build_gerrit_change_url_from_mapping,
    build_gerrit_skip_message,
    build_gerrit_submission_comment,
    detect_github2gerrit_comments,
    fetch_gitreview_from_github,
)
from .github_async import GitHubAsync
from .github_async import PermissionError as GitHubPermissionError
from .github_service import GitHubService
from .models import ComparisonResult, PullRequestInfo
from .netrc import NetrcParseError, resolve_gerrit_credentials
from .output_utils import log_and_print
from .progress_tracker import MergeProgressTracker
from .slot_lease import holding_slot, parked

# Centralised timing constants for all async merge operations.
#
# Every polling loop in this module (post-rebase status checks,
# pre-commit.ci re-runs, @dependabot recreate, recreated-PR readiness)
# derives its iteration count from these two values so that the timeout
# is consistent and easy to adjust from a single place or via the
# ``--merge-timeout`` CLI flag.
DEFAULT_MERGE_TIMEOUT: float = 300.0  # seconds (5 minutes)
DEFAULT_MERGE_RECHECK_INTERVAL: float = 10.0  # seconds between polls

# After a sibling PR merges (or a concurrent dependamerge run lands a
# change), GitHub recomputes a PR's mergeability asynchronously and
# briefly reports ``mergeable=null`` / ``mergeable_state="unknown"`` —
# typically for a few seconds.  Before dispatching a merge in a
# repo-scoped batch we re-read the PR and, when GitHub is still
# computing, poll up to this many seconds for a concrete value so the
# merge decision is made against fresh state rather than the
# (possibly stale) fetch-time snapshot.
MERGEABILITY_REFRESH_TIMEOUT_SECONDS: float = 10.0

# First-poll delay for the auto-merge wait loop.  The loop's steady
# cadence is ``DEFAULT_MERGE_RECHECK_INTERVAL`` (10s), but the *first*
# refresh happens after this much shorter delay: when auto-merge fires
# the moment it is armed (checks were already green and approval was
# the only blocker) a full-interval first sleep would discover the
# merge ~8 seconds late — per PR, serialized per repository in striped
# runs.  One extra lightweight GET is a fair trade for that.
MERGE_WAIT_FIRST_POLL_SECONDS: float = 2.0

# Required verification checks (DCO, lint, build, etc.) normally
# start reporting within a few seconds.  When a *required* check has
# been pending for longer than this on a PR that itself was created
# / last updated more than this many seconds ago, the check is
# treated as stuck.  Used by ``_detect_stuck_required_check`` to
# decide whether to ask dependabot to recreate the PR.
STUCK_CHECK_THRESHOLD_SECONDS: float = 60.0

# pre-commit.ci normally reports back within a few minutes.  A
# ``pre-commit.ci - pr`` status stuck in ``pending`` for longer than
# this is treated as a hung run that needs a fresh ``pre-commit.ci
# run`` trigger.  Kept deliberately generous so a slow-but-normal run
# is never interrupted; used by ``_trigger_stale_precommit_ci``.
PRECOMMIT_CI_STUCK_PENDING_SECONDS: float = 300.0


class MergeStatus(Enum):
    """Status of a PR merge operation."""

    PENDING = "pending"
    APPROVING = "approving"
    APPROVED = "approved"
    MERGING = "merging"
    MERGED = "merged"
    AUTO_MERGE_PENDING = "auto_merge_pending"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"
    # Terminal: the PR was closed without merging (dependabot decided
    # the update is no longer needed after sibling merges, the PR was
    # superseded, or a human closed it mid-run).  Distinct from FAILED
    # because there is nothing for the operator to follow up on.
    CLOSED = "closed"


@dataclass
class MergeResult:
    """Result of a PR merge operation."""

    pr_info: PullRequestInfo
    status: MergeStatus
    error: str | None = None
    # Non-fatal note attached to a *successful* (or otherwise non-error)
    # outcome — e.g. a preview MERGED result for a PR that is behind its
    # base branch and would be rebased first. Kept separate from ``error``
    # so a MERGED status never carries a contradictory error message.
    warning: str | None = None
    attempts: int = 0
    duration: float = 0.0


class AsyncMergeManager:
    """
    Manages parallel approval and merging of pull requests.

    This class handles:
    - Concurrent approval of PRs
    - Concurrent merging with retry logic
    - Progress tracking and error handling
    - Rate limit-aware processing
    """

    def __init__(
        self,
        token: str,
        merge_method: str = "merge",
        max_retries: int = 2,
        concurrency: int = 5,
        fix_out_of_date: bool = False,
        merge_timeout: float = DEFAULT_MERGE_TIMEOUT,
        progress_tracker: MergeProgressTracker | None = None,
        preview_mode: bool = False,
        dismiss_copilot: bool = False,
        force_level: str = "code-owners",
        github2gerrit_mode: str = "submit",
        no_netrc: bool = False,
        netrc_file: Path | None = None,
        rebase_local: bool = True,
        repo_scoped: bool = False,
        max_wait: float | None = None,
    ):
        self.token = token
        self.default_merge_method = merge_method
        self.max_retries = max_retries
        self.concurrency = concurrency
        self.fix_out_of_date = fix_out_of_date
        self.progress_tracker = progress_tracker
        self.preview_mode = preview_mode
        self.dismiss_copilot = dismiss_copilot
        self.force_level = force_level
        self.github2gerrit_mode = github2gerrit_mode
        self.no_netrc = no_netrc
        self.netrc_file = netrc_file
        # When True (the default), Step 5's rebase path uses a local
        # ``git`` clone + rebase + force-push-with-lease workflow
        # for PRs whose verification status would otherwise be lost
        # by the GitHub REST ``update-branch`` endpoint. The local
        # workflow inherits the user's ``~/.gitconfig`` and so
        # respects ``commit.gpgsign`` / ``gpg.format`` /
        # ``user.signingkey`` automatically. Set to False to force
        # the legacy REST-only path (simpler but loses signature
        # verification on signed branches).
        self.rebase_local = rebase_local
        # When True, a worker refreshes its PR's live merge state just
        # before dispatching, because a sibling merge can make a PR
        # ``dirty`` / ``behind`` between the up-front fetch and this
        # worker's dispatch (see ``_refresh_pr_mergeability``).  Enabled
        # both for single-repository batches and for owner-wide striped
        # runs, where each repository's PRs are serialised so an earlier
        # merge can invalidate a later sibling.  Left False only for
        # similar-PR runs spread across unrelated repositories, where our
        # own merges do not invalidate the snapshot.
        self._repo_scoped = repo_scoped
        # Owner-wide global wait ceiling (seconds), or ``None`` for
        # repository / similar-PR runs which keep the legacy per-PR
        # ``merge_timeout`` behaviour with no overall cap.  Semantics:
        #   * ``None``  — no global ceiling (per-PR ``merge_timeout``
        #                 governs each wait independently).
        #   * ``> 0``   — a wall-clock ceiling for the whole run; every
        #                 per-PR wait deadline is clamped to it, so the
        #                 run cannot block past this bound.  Anything
        #                 still in flight when it elapses keeps auto-merge
        #                 armed and is reported AUTO_MERGE_PENDING.
        #   * ``0``     — fire-and-forget: never block.  Approve, arm
        #                 auto-merge, report AUTO_MERGE_PENDING, move on.
        # ``_run_deadline`` (the resolved monotonic ceiling) and
        # ``_no_wait`` (the ``0`` case) are set when a run starts.
        self._max_wait = max_wait
        self._run_deadline: float | None = None
        self._no_wait: bool = max_wait is not None and max_wait <= 0
        self.log = logging.getLogger(__name__)

        # Centralised merge-operation timing
        # Coerce merge_timeout to float and validate, guarding against Typer
        # OptionInfo objects that leak through when the CLI function is called
        # directly (e.g. from tests) without the Typer argument parser.
        try:
            _mt = float(merge_timeout)
            if not math.isfinite(_mt) or _mt <= 0:
                raise ValueError(f"out of range: {_mt}")
            self._merge_timeout = _mt
        except (TypeError, ValueError):
            self.log.warning(
                "Invalid merge_timeout=%r; falling back to default of %.0f seconds",
                merge_timeout,
                DEFAULT_MERGE_TIMEOUT,
            )
            self._merge_timeout = DEFAULT_MERGE_TIMEOUT
        # Clamp the per-iteration sleep so a small ``merge_timeout``
        # (< DEFAULT_MERGE_RECHECK_INTERVAL) does not over-sleep and
        # blow past the user-specified total timeout. For typical
        # values (>= 10s), this is a no-op and keeps the default
        # 10s polling cadence.
        self._merge_recheck_interval = min(
            DEFAULT_MERGE_RECHECK_INTERVAL, self._merge_timeout
        )
        # Use math.ceil so the effective poll window is at least
        # the configured ``merge_timeout`` — plain ``int()`` would
        # truncate (e.g. 301/10 -> 30 attempts -> only 300s).
        self._merge_poll_max_attempts = max(
            1, math.ceil(self._merge_timeout / self._merge_recheck_interval)
        )

        # Track merge operations
        self._merge_semaphore = asyncio.Semaphore(concurrency)
        self._results: list[MergeResult] = []
        self._github_client: GitHubAsync | None = None
        self._github_service: GitHubService | None = None
        self._copilot_handler: CopilotCommentHandler | None = None
        # Reuse the progress tracker's Rich Console (when one is
        # provided) so per-PR ✅/❌ lines emitted during a merge run
        # interleave cleanly with the Live progress display.  Using a
        # separate Console() instance causes Rich's Live re-draw to
        # garble or eat those messages because the two consoles share
        # the terminal but coordinate independently.
        tracker_console = getattr(progress_tracker, "console", None)
        self._console = tracker_console if tracker_console is not None else Console()

        # Track merge methods per repository
        self._pr_merge_methods: dict[str, str] = {}

        # Cache for organization-level settings to avoid repeated API calls
        # Key: org name, Value: org settings dict (or None on failure)
        self._org_settings_cache: dict[str, dict[str, Any] | None] = {}
        self._org_settings_locks: dict[str, asyncio.Lock] = {}
        self._org_settings_locks_lock = asyncio.Lock()

        # Cache for "does this branch mandate an approving review before
        # any merge" detection, keyed by "owner/repo@branch".  The answer
        # is fixed for the lifetime of a run (rulesets don't change
        # mid-merge), so the resolved verdict is reused for every PR
        # targeting the same repo+branch.
        self._branch_approval_cache: dict[str, bool] = {}
        self._branch_approval_locks: dict[str, asyncio.Lock] = {}
        self._branch_approval_locks_lock = asyncio.Lock()

        # Cache for the organization-level approval requirement, enumerated
        # once per org from its repository rulesets.  Value is the list of
        # approval-mandating rulesets (each ``{"name", "conditions"}``),
        # ``[]`` when the org mandates none, or ``None`` when enumeration
        # failed (e.g. the token cannot read org rulesets) so callers know
        # to consult the authoritative per-repo endpoint instead.
        self._org_approval_cache: dict[str, list[dict[str, Any]] | None] = {}
        self._org_approval_locks: dict[str, asyncio.Lock] = {}
        self._org_approval_locks_lock = asyncio.Lock()

        # Track last merge exception per PR for better error reporting
        self._last_merge_exception: dict[str, Exception] = {}

        # Track PRs that were just approved (for post-approval merge retry)
        self._recently_approved: set[str] = set()

        # Track repositories where the token has already failed a
        # permission check during this run.  Subsequent PRs in the
        # same repository are short-circuited with a clean skip
        # message rather than triggering another round-trip to
        # GitHub that will fail identically and emit another
        # screenful of guidance.
        self._permission_failed_repos: set[str] = set()

        # Track PRs where auto-merge has been enabled so that
        # post-timeout merge attempts can be skipped gracefully.
        self._auto_merge_enabled: set[str] = set()

        # Track PRs that have already gone through Step 5's
        # rebase + poll path so Step 5.5 can skip them and avoid
        # doubling the configured ``merge_timeout``. Set after
        # Step 5 completes its wait, regardless of whether the
        # final state is ``clean``, ``blocked``, or ``behind``.
        self._rebased_prs: set[str] = set()

        # Track PRs currently waiting for required checks to complete.
        # Maps ``pr_key`` -> deadline (monotonic seconds) so the
        # parallel merge ticker can render an aggregate countdown
        # without poking inside individual worker tasks.
        self._waiting_prs: dict[str, float] = {}
        self._waiting_lock = asyncio.Lock()

        # Per-repo locks that serialise the actual ``merge_pull_request``
        # API call.  Multiple workers can run in parallel through approve,
        # rebase polling, and Step 5.5's auto-merge wait loop — only the
        # final dispatch is serialised, and only between PRs that target
        # the same repository.  This avoids the head-of-line blocking we
        # used to get from forcing ``concurrency=1`` for repo-scoped
        # runs (where a single PR parked in the wait loop could block
        # every other PR in the batch for the full ``merge_timeout``)
        # while still preventing back-to-back merges on the same repo
        # from racing GitHub's branch-protection propagation.
        self._merge_dispatch_locks: dict[str, asyncio.Lock] = {}
        self._merge_dispatch_locks_lock = asyncio.Lock()

        # Delay (seconds) after submitting a new approval before attempting merge.
        # GitHub needs time to propagate the approval to branch-protection evaluation.
        default_post_approval_delay = 3.0
        env_post_approval_delay = os.getenv(
            "DEPENDAMERGE_POST_APPROVAL_DELAY",
            str(default_post_approval_delay),
        )
        try:
            parsed_delay = float(env_post_approval_delay)
            if not math.isfinite(parsed_delay) or parsed_delay < 0:
                raise ValueError(f"out of range: {parsed_delay}")
            self._post_approval_delay = parsed_delay
        except ValueError:
            self.log.warning(
                "Invalid DEPENDAMERGE_POST_APPROVAL_DELAY=%r; "
                "falling back to default of %.1f seconds",
                env_post_approval_delay,
                default_post_approval_delay,
            )
            self._post_approval_delay = default_post_approval_delay

    def __repr__(self) -> str:
        """Safe repr that never exposes the token value."""
        return "AsyncMergeManager(token=***)"

    def _get_mergeability_icon_and_style(
        self, mergeable_state: str | None
    ) -> tuple[str, str | None]:
        """Get appropriate icon and style for mergeable state."""
        if mergeable_state == "dirty":
            return "🛑", "red"
        elif mergeable_state == "behind":
            return "⚠️", "yellow"
        elif mergeable_state == "clean":
            return "✅", "green"
        elif mergeable_state == "draft":
            return "📝", "blue"
        else:
            return "🔍", None

    async def __aenter__(self):
        """Async context manager entry."""
        self._github_client = GitHubAsync(token=self.token)
        await self._github_client.__aenter__()

        self._github_service = GitHubService(token=self.token)

        # Initialize Copilot handler if dismissal is enabled
        if self.dismiss_copilot:
            self._copilot_handler = CopilotCommentHandler(
                self._github_client, preview_mode=self.preview_mode, debug=True
            )

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._github_service:
            await self._github_service.close()
        if self._github_client:
            await self._github_client.__aexit__(exc_type, exc_val, exc_tb)

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
        if not pr_list:
            return []

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

    def _collect_results(
        self,
        pr_list: list[tuple[PullRequestInfo, ComparisonResult | None]],
        results: list[Any],
    ) -> list[MergeResult]:
        """Map gathered task outcomes back to ``MergeResult`` objects.

        Converts any propagated exception (from a per-PR task run with
        ``return_exceptions=True``) into a FAILED result for the matching
        PR, preserving the input ordering.
        """
        final_results: list[MergeResult] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                pr_info = pr_list[i][0]
                error_result = MergeResult(
                    pr_info=pr_info, status=MergeStatus.FAILED, error=str(result)
                )
                final_results.append(error_result)
                # The exception escaped ``_merge_single_pr_with_semaphore``
                # before it could record a terminal outcome, so record
                # the failure here to keep the tracker counters exact.
                self._record_terminal_outcome(pr_info, MergeStatus.FAILED)
                self.log.error(
                    f"Unexpected error merging PR {pr_info.repository_full_name}#{pr_info.number}: {result}"
                )
            else:
                # result is guaranteed to be MergeResult here since it's not an Exception
                final_results.append(cast(MergeResult, result))
        return final_results

    def _record_terminal_outcome(
        self, pr_info: PullRequestInfo, status: MergeStatus
    ) -> None:
        """Record a PR's terminal outcome on the progress tracker.

        This is the **single** place terminal outcomes reach the
        tracker: every PR ends in exactly one counter (merged /
        failed / skipped / blocked / closed / pending), its
        transitory
        display state (rebasing, waiting, …) is cleared, and the
        PR-level completion percentage advances.  Centralising the
        accounting here closes the historical "result returned but
        tracker never told" and double-count bug classes.
        """
        tracker = self.progress_tracker
        if not tracker:
            return
        pr_key = f"{pr_info.repository_full_name}#{pr_info.number}"
        if status == MergeStatus.MERGED:
            tracker.merge_success(pr_key)
        elif status == MergeStatus.FAILED:
            tracker.merge_failure(pr_key)
        elif status == MergeStatus.SKIPPED:
            tracker.merge_skipped(pr_key)
        elif status == MergeStatus.BLOCKED:
            tracker.merge_blocked(pr_key)
        elif status == MergeStatus.CLOSED:
            tracker.increment_closed(pr_key)
        elif status == MergeStatus.AUTO_MERGE_PENDING:
            tracker.merge_pending(pr_key)
        else:
            # Defensive: an unexpected terminal status still counts
            # toward completion so the percentage reaches 100%.  Clear
            # any transitory display state first so the PR cannot be
            # left stuck in "rebasing"/"waiting" on the live display
            # if a new terminal status is added without a counter
            # mapping here.
            tracker.track_pr_state(pr_key, None)
            tracker.pr_completed()

    def _track_pr_state(self, pr_info: PullRequestInfo, state: str | None) -> None:
        """Move a PR between transitory tracker states (or clear)."""
        tracker = self.progress_tracker
        if not tracker:
            return
        pr_key = f"{pr_info.repository_full_name}#{pr_info.number}"
        tracker.track_pr_state(pr_key, state)

    def _pr_status(self, message: str, *, level: str = "info") -> None:
        """Emit a per-PR status line.

        Preview mode prints the line (the \"🔍 Dependamerge
        Evaluation\" section requires exactly one line per PR).  Real
        merges keep the console clean — progress is conveyed by the
        Rich tracker counters and the reasons are reported in the
        end-of-run summary — so the message goes to the log only.
        """
        if self.preview_mode:
            log_and_print(self.log, self._console, message, level=level)
        else:
            log_func = getattr(self.log, level.lower(), self.log.info)
            log_func(message)

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

    async def _wait_status_ticker(self) -> None:
        """Update the progress display while PRs wait for required checks.

        Runs as a background task for the lifetime of
        ``merge_prs_parallel``. Once per second it samples
        ``self._waiting_prs`` and pushes a single-line status
        message into the progress tracker (which uses Rich Live
        for in-place updates) so the user can see how much
        longer the tool will block before returning the shell
        prompt.

        The countdown uses the latest (worst-case) deadline across
        all waiting PRs so the displayed value reflects the longest
        remaining wait, not an arbitrary one. When no PRs are
        waiting, the message is cleared.

        When the progress tracker is in non-Rich (fallback) mode,
        the per-update line would print to stdout, which would spam
        logs every second. In that case we delegate to the plain
        ticker (15s cadence) instead.
        """
        if not self.progress_tracker:
            # Fallback: emit a periodic plain console line so the
            # user still gets feedback even without Rich progress.
            await self._wait_status_ticker_plain()
            return

        # If the tracker exists but Rich is unavailable (non-TTY,
        # no Rich library, etc.), it falls back to per-update
        # ``print()`` calls. Updating every second would spam the
        # user's terminal/logs, so use the slower plain ticker
        # cadence instead.
        rich_available = bool(getattr(self.progress_tracker, "rich_available", False))
        if not rich_available:
            await self._wait_status_ticker_plain()
            return

        last_message: str | None = None
        try:
            while True:
                async with self._waiting_lock:
                    snapshot = dict(self._waiting_prs)

                if snapshot:
                    now = asyncio.get_running_loop().time()
                    remaining = max(
                        0.0,
                        max(snapshot.values()) - now,
                    )
                    count = len(snapshot)
                    noun = "PR" if count == 1 else "PRs"
                    message = (
                        f"⏳ Waiting for {count} {noun} "
                        f"to complete checks [{int(remaining)}s]"
                    )
                else:
                    message = ""

                if message != last_message:
                    try:
                        self.progress_tracker.update_operation(message)
                    except Exception:
                        # Defensive: a failing tracker must never
                        # take down the whole merge run.
                        pass
                    last_message = message

                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            # Best-effort clear on shutdown so the final tracker
            # state isn't stuck on a stale countdown.
            if last_message:
                try:
                    self.progress_tracker.update_operation("")
                except Exception:
                    # Defensive: a failing tracker must never take down
                    # the shutdown path.
                    pass
            raise

    async def _wait_status_ticker_plain(self) -> None:
        """Plain-text countdown when no Rich progress tracker is present.

        Emits one console line every 15 seconds while PRs are
        waiting on required checks. Less granular than the Rich
        in-place update, but still gives the user visibility into
        why the tool is blocking.
        """
        last_emit: float = 0.0
        try:
            while True:
                async with self._waiting_lock:
                    snapshot = dict(self._waiting_prs)

                if snapshot:
                    now = asyncio.get_running_loop().time()
                    if now - last_emit >= 15.0:
                        remaining = max(0.0, max(snapshot.values()) - now)
                        count = len(snapshot)
                        noun = "PR" if count == 1 else "PRs"
                        try:
                            self._console.print(
                                f"⏳ Waiting for {count} {noun} "
                                f"to complete checks "
                                f"[{int(remaining)}s remaining]"
                            )
                        except Exception:
                            # Console output is best-effort; ignore
                            # write errors on unusual terminals.
                            pass
                        last_emit = now
                else:
                    last_emit = 0.0

                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            raise

    async def _detect_github2gerrit(
        self,
        repo_owner: str,
        repo_name: str,
        pr_number: int,
    ) -> GitHub2GerritDetectionResult:
        """
        Fetch issue comments for a PR and check for GitHub2Gerrit mapping.

        Args:
            repo_owner: Repository owner.
            repo_name: Repository name.
            pr_number: Pull request number.

        Returns:
            Detection result with mapping data if found.
        """
        try:
            if self._github_client is None:
                raise RuntimeError("GitHub client not initialized")

            # Fetch issue comments (not review comments) via REST API
            comments = await self._github_client.get(
                f"/repos/{repo_owner}/{repo_name}/issues/{pr_number}/comments"
            )

            if not isinstance(comments, list):
                return GitHub2GerritDetectionResult()

            return detect_github2gerrit_comments(comments)

        except Exception as exc:
            self.log.debug(
                "Failed to check GitHub2Gerrit comments for %s/%s#%d: %s",
                repo_owner,
                repo_name,
                pr_number,
                exc,
            )
            return GitHub2GerritDetectionResult()

    async def _submit_gerrit_change(
        self,
        mapping: GitHub2GerritMapping,
        pr_info: PullRequestInfo,
        repo_owner: str,
        repo_name: str,
    ) -> bool:
        """
        Submit the corresponding Gerrit change for a GitHub2Gerrit PR.

        Resolves Gerrit credentials, looks up the change by Change-ID,
        applies +2 Code-Review, and submits it.

        Args:
            mapping: The parsed GitHub2Gerrit mapping.
            pr_info: The GitHub pull request info.
            repo_owner: Repository owner (org or user).
            repo_name: Repository name.

        Returns:
            True if the Gerrit change was successfully submitted.
        """
        # We need to figure out the Gerrit host.  The mapping's topic name
        # follows the pattern "GH-<repo>-<number>" which doesn't embed the
        # host.  We look for a Gerrit change URL in the mapping comment body,
        # or fall back to well-known hosts.
        gerrit_host, gerrit_base_path = await self._resolve_gerrit_host(
            mapping, repo_owner, repo_name
        )

        if not gerrit_host:
            self.log.warning(
                "Cannot determine Gerrit host for GitHub2Gerrit mapping "
                "(topic: %s). Skipping Gerrit submission.",
                mapping.topic,
            )
            return False

        # Resolve credentials
        try:
            credentials = resolve_gerrit_credentials(
                host=gerrit_host,
                use_netrc=not self.no_netrc,
                netrc_file=self.netrc_file,
            )
        except NetrcParseError as exc:
            self.log.warning("Error parsing .netrc for Gerrit: %s", exc)
            credentials = None

        if credentials is None or not credentials.is_valid:
            self.log.warning(
                "No Gerrit credentials found for %s. Cannot submit "
                "GitHub2Gerrit change (topic: %s).",
                gerrit_host,
                mapping.topic,
            )
            return False

        try:
            service = create_gerrit_service(
                host=gerrit_host,
                base_path=gerrit_base_path,
                username=credentials.username,
                password=credentials.password,
            )

            # Query Gerrit for the change using the primary Change-ID
            change_id = mapping.primary_change_id
            changes = service._query_changes(
                query=f"change:{change_id} status:open",
                limit=5,
                offset=0,
                options=[
                    "CURRENT_REVISION",
                    "LABELS",
                    "DETAILED_LABELS",
                    "SUBMIT_REQUIREMENTS",
                ],
            )

            if not changes:
                self.log.warning(
                    "No open Gerrit change found for Change-Id %s on %s",
                    change_id,
                    gerrit_host,
                )
                return False

            # Use the first matching change
            gerrit_change = changes[0]
            self.log.info(
                "Found Gerrit change %s #%d for Change-Id %s",
                gerrit_change.project,
                gerrit_change.number,
                change_id,
            )

            submit_manager = create_submit_manager(
                host=gerrit_host,
                base_path=gerrit_base_path,
                username=credentials.username,
                password=credentials.password,
            )

            results = submit_manager.submit_changes(
                [(gerrit_change, None)],
                review_labels={"Code-Review": 2},
                dry_run=self.preview_mode,
            )

            if results and results[0].submitted:
                self.log.info(
                    "Successfully submitted Gerrit change %s #%d",
                    gerrit_change.project,
                    gerrit_change.number,
                )

                # Post a comment on the GitHub PR and close it
                gerrit_url = build_gerrit_change_url_from_mapping(
                    mapping, gerrit_host, gerrit_base_path
                )
                await self._close_github_pr_after_gerrit_submit(
                    pr_info, mapping, gerrit_url
                )

                return True

            if results and results[0].success and self.preview_mode:
                # Dry-run succeeded
                return True

            error_msg = results[0].error if results else "Unknown error"
            self.log.warning(
                "Failed to submit Gerrit change %s #%d: %s",
                gerrit_change.project,
                gerrit_change.number,
                error_msg,
            )
            return False

        except (GerritAuthError, GerritRestError) as exc:
            self.log.warning(
                "Gerrit error submitting change for topic %s: %s",
                mapping.topic,
                exc,
            )
            return False
        except Exception as exc:
            self.log.warning(
                "Unexpected error submitting Gerrit change for topic %s: %s",
                mapping.topic,
                exc,
            )
            return False

    async def _close_github_pr_after_gerrit_submit(
        self,
        pr_info: PullRequestInfo,
        mapping: GitHub2GerritMapping,
        gerrit_url: str,
    ) -> None:
        """
        Close the GitHub PR and post a comment after Gerrit submission.

        Args:
            pr_info: The GitHub pull request.
            mapping: The parsed mapping.
            gerrit_url: URL of the submitted Gerrit change.
        """
        if self.preview_mode:
            return

        repo_owner, repo_name = pr_info.repository_full_name.split("/", 1)

        try:
            if self._github_client is None:
                raise RuntimeError("GitHub client not initialized")

            # Post comment following GitHub2Gerrit conventions
            comment_body = build_gerrit_submission_comment(mapping, gerrit_url)
            await self._github_client.post_issue_comment(
                repo_owner, repo_name, pr_info.number, comment_body
            )

            # Close the PR
            await self._github_client.close_pull_request(
                repo_owner, repo_name, pr_info.number
            )

            self.log.info(
                "Closed GitHub PR %s#%d after Gerrit submission",
                pr_info.repository_full_name,
                pr_info.number,
            )
        except Exception as exc:
            self.log.warning(
                "Failed to close GitHub PR %s#%d after Gerrit submission: %s",
                pr_info.repository_full_name,
                pr_info.number,
                exc,
            )

    async def _resolve_gerrit_host(
        self,
        mapping: GitHub2GerritMapping,
        repo_owner: str,
        repo_name: str,
    ) -> tuple[str | None, str | None]:
        """
        Determine the Gerrit host and base path for a GitHub2Gerrit PR.

        Resolution priority (highest first):

        1. ``.gitreview`` file in the repository (canonical source of truth)
        2. ``GERRIT_HOST`` / ``GERRIT_BASE_PATH`` environment variables
        3. Gerrit URL embedded in the mapping comment body
        4. Well-known host conventions (e.g. ``lfit`` → LF Gerrit)
        5. ``GERRIT_URL`` environment variable

        The ``.gitreview`` file is treated as definitive because every
        repository that uses GitHub2Gerrit is required to have one, and it
        records the exact Gerrit host, port, and project path.

        Args:
            mapping: The parsed GitHub2Gerrit mapping from the PR comment.
            repo_owner: Repository owner (org or user).
            repo_name: Repository name.

        Returns:
            Tuple of (host, base_path) or (None, None) if unresolvable.
        """
        # 1. .gitreview file — highest priority / source of truth
        if self._github_client is not None:
            gitreview_info = await fetch_gitreview_from_github(
                self._github_client, repo_owner, repo_name
            )
            if gitreview_info and gitreview_info.is_valid:
                self.log.info(
                    "Resolved Gerrit host from .gitreview in %s/%s: %s (base_path=%s)",
                    repo_owner,
                    repo_name,
                    gitreview_info.host,
                    gitreview_info.base_path,
                )
                return gitreview_info.host, gitreview_info.base_path

        # 2. Explicit environment variables
        env_host = os.getenv("GERRIT_HOST", "").strip()
        env_base_path = os.getenv("GERRIT_BASE_PATH", "").strip() or None
        if env_host:
            return env_host, env_base_path

        # 3. Gerrit URL embedded in the mapping comment body
        if mapping.raw_comment_body:
            gerrit_url_match = re.search(
                r"https?://([^/\s]+)(?:/([\w-]+))?/c/",
                mapping.raw_comment_body,
            )
            if gerrit_url_match:
                host = gerrit_url_match.group(1)
                base_path = (
                    gerrit_url_match.group(2) if gerrit_url_match.group(2) else None
                )
                return host, base_path

        # 4. Well-known LF Gerrit host
        if (
            mapping.pr_url and "github.com/lfit/" in mapping.pr_url
        ) or repo_owner == "lfit":
            return "gerrit.linuxfoundation.org", "infra"

        # 5. GERRIT_URL environment variable (catch-all)
        gerrit_url = os.getenv("GERRIT_URL", "").strip()
        if gerrit_url:
            url_match = re.match(r"https?://([^/]+)(?:/([\w-]+))?/?$", gerrit_url)
            if url_match:
                return url_match.group(1), url_match.group(2) if url_match.group(
                    2
                ) else None

        return None, None

    async def _merge_single_pr(self, pr_info: PullRequestInfo) -> MergeResult:
        """
        Merge a single pull request with retry logic.

        Args:
            pr_info: Pull request information

        Returns:
            MergeResult with operation status and details
        """
        start_time = time.time()
        repo_owner, repo_name = pr_info.repository_full_name.split("/", 1)

        # Fast-fail when a previous PR in this batch has already
        # hit a permission error against the same repository.  In
        # that case the token genuinely lacks the rights to act on
        # any PR in this repo, so attempting the GitHub API calls
        # again would only produce another 403 and another copy of
        # the token-guidance block.  Report the failure cleanly
        # (single ❌ line, no traceback) and move on.
        if pr_info.repository_full_name in self._permission_failed_repos:
            result = MergeResult(pr_info=pr_info, status=MergeStatus.FAILED)
            result.error = (
                f"token lacks required permissions on {pr_info.repository_full_name}"
            )
            self._pr_status(
                f"❌ Failed: {pr_info.html_url} "
                "[token lacks permissions on this repository]",
                level="error",
            )
            result.duration = time.time() - start_time
            return result

        # Early determination of merge method based on repository settings
        merge_method = await self._get_merge_method_for_repo(repo_owner, repo_name)

        # Store the determined merge method for this PR
        self._pr_merge_methods[f"{repo_owner}/{repo_name}"] = merge_method

        result = MergeResult(pr_info=pr_info, status=MergeStatus.PENDING)

        try:
            if self.github2gerrit_mode != "ignore":
                g2g_result = await self._detect_github2gerrit(
                    repo_owner, repo_name, pr_info.number
                )

                if g2g_result.has_mapping and g2g_result.mapping:
                    mapping = g2g_result.mapping
                    skip_msg = build_gerrit_skip_message(mapping)

                    if self.github2gerrit_mode == "skip":
                        # Skip this PR entirely
                        result.status = MergeStatus.SKIPPED
                        result.error = f"Skipped: {skip_msg}"
                        self._pr_status(
                            f"⏩ Skipped: {pr_info.html_url} [{skip_msg}]",
                            level="info",
                        )
                        return result

                    # Default: "submit" mode - submit the Gerrit change
                    if self.preview_mode:
                        self._pr_status(
                            f"🔄 Gerrit submit: {pr_info.html_url} [{skip_msg}]",
                            level="info",
                        )
                        result.status = MergeStatus.MERGED
                        return result

                    # Attempt to submit the Gerrit change
                    self._pr_status(
                        f"🔄 Submitting Gerrit change for {pr_info.html_url} "
                        f"[{skip_msg}]",
                        level="info",
                    )
                    submitted = await self._submit_gerrit_change(
                        mapping, pr_info, repo_owner, repo_name
                    )

                    if submitted:
                        result.status = MergeStatus.MERGED
                        self._pr_status(
                            f"✅ Gerrit submitted: {pr_info.html_url}",
                            level="info",
                        )
                        return result

                    # Gerrit submission failed - report as failed
                    result.status = MergeStatus.FAILED
                    result.error = f"Failed to submit Gerrit change ({skip_msg})"
                    self._pr_status(
                        f"❌ Failed: {pr_info.html_url} "
                        f"[Gerrit submit failed for {skip_msg}]",
                        level="error",
                    )
                    return result

            # Check if PR is closed before processing.  If it has
            # been closed *and merged* by another process (a
            # concurrent dependamerge run, a human admin, an
            # auto-merge that landed mid-flight, etc.) we treat it
            # as a skip rather than a failure: there is no
            # remaining work or human follow-up to perform.
            if pr_info.state != "open":
                already_merged = await self._is_pr_already_merged(
                    pr_info, repo_owner, repo_name
                )
                if already_merged:
                    result.status = MergeStatus.SKIPPED
                    result.error = "already merged externally"
                    self._pr_status(
                        f"⏭️ Skipped: {pr_info.html_url} [already merged externally]",
                        level="info",
                    )
                    return result
                result.status = MergeStatus.CLOSED
                result.error = "PR was already closed without merging"
                self._pr_status(
                    f"🚪 Closed: {pr_info.html_url} [already closed]",
                    level="info",
                )
                return result

            # A merge conflict (``dirty``) has no merge path of its
            # own: route it to the conflict handler (dependabot →
            # ``@dependabot rebase`` + wait; other authors → report
            # and fail fast) rather than the generic not-mergeable
            # skip below.  Skipped in preview (no side effects) and
            # under ``force=all`` (which intentionally attempts the
            # merge regardless of state).
            if (
                pr_info.mergeable_state == "dirty"
                and not self.preview_mode
                and self.force_level != "all"
            ):
                return await self._handle_merge_conflict(
                    pr_info, repo_owner, repo_name, result
                )

            if not self._is_pr_mergeable(pr_info):
                return await self._handle_not_mergeable_pr(pr_info, result)

            # Check for blocking reviews (changes requested)
            if self._has_blocking_reviews(pr_info):
                # Only skip if not forcing with 'all' level
                if self.force_level != "all":
                    result.status = MergeStatus.SKIPPED
                    result.error = "PR has reviews requesting changes - will not override human feedback"
                    self._pr_status(
                        f"⏭️ Skipped: {pr_info.html_url} [has reviews requesting changes]",
                        level="debug",
                    )
                    return result
                else:
                    # Only log during preview evaluation to avoid duplicate messages
                    if self.preview_mode:
                        self.log.warning(
                            f"⚠️ Overriding blocking reviews for {pr_info.repository_full_name}#{pr_info.number} (--force=all)"
                        )

            # If the PR is blocked, check for stale pre-commit.ci
            # and trigger a re-run before evaluating merge requirements.
            # Avoid triggering side effects when running in preview mode.
            if (
                not self.preview_mode
                and pr_info.mergeable_state == "blocked"
                and self._github_client
            ):
                precommit_fixed = await self._trigger_stale_precommit_ci(pr_info)
                if precommit_fixed:
                    # Re-fetch PR state now that pre-commit.ci has passed
                    try:
                        updated = await self._github_client.get(
                            f"/repos/{repo_owner}/{repo_name}/pulls/{pr_info.number}"
                        )
                        if isinstance(updated, dict):
                            pr_info.mergeable = updated.get("mergeable")
                            pr_info.mergeable_state = updated.get("mergeable_state")
                    except Exception as e:
                        self.log.debug(
                            "Failed to refresh PR %s mergeable state after "
                            "pre-commit.ci rerun: %s",
                            f"{pr_info.repository_full_name}#{pr_info.number}",
                            e,
                        )

            can_merge, merge_check_reason = await self._check_merge_requirements(
                pr_info
            )

            if not can_merge:
                result.status = MergeStatus.SKIPPED
                result.error = f"Merge requirements not met: {merge_check_reason}"
                self._pr_status(
                    f"⏭️ Skipped: {pr_info.html_url} [{merge_check_reason.lower()}]",
                    level="debug",
                )
                return result

            copilot_processing_successful = True
            if self.dismiss_copilot and self._copilot_handler:
                # Analyze what types of reviews we have
                self._copilot_handler.analyze_copilot_review_dismissibility(pr_info)

                try:
                    (
                        processed_count,
                        total_count,
                    ) = await self._copilot_handler.dismiss_copilot_comments_for_pr(
                        pr_info
                    )
                    if total_count > 0:
                        # Silent processing in background
                        pass
                except Exception as e:
                    self.log.warning(
                        f"⚠️ Failed to process Copilot items for PR {pr_info.number}: {e}"
                    )
                    copilot_processing_successful = False

            # Gate on Copilot processing, but DO NOT approve
            # up-front. Approval is now performed on demand (approve-on-
            # demand): either just before arming auto-merge (see
            # _enable_auto_merge_with_approval) or after a direct merge is
            # rejected specifically for a missing review (see
            # _approve_and_retry_if_review_required). This avoids approving
            # PRs that did not actually need our review, while the Copilot
            # gate below still prevents acting on a PR with unresolved
            # Copilot feedback.
            if not copilot_processing_successful:
                result.status = MergeStatus.FAILED
                result.error = "Copilot review processing incomplete - not approving to avoid pollution"
                self._pr_status(
                    f"❌ Failed: {pr_info.html_url} [copilot processing incomplete]",
                    level="error",
                )
                return result

            # Handle rebase if needed before merge.
            #
            # Dispatched to the dedicated ``rebase`` module so the
            # local-vs-REST decision tree, the local-git workflow,
            # and the post-rebase polling loop all live in one
            # place where they can be tested in isolation.
            #
            # ``mergeable_state`` is a single value, so ``blocked``
            # (failing required check) masks ``behind`` (stale head).
            # A required check that *failed* on a branch that is
            # *behind base* was judged against pre-rebase content —
            # e.g. an org-required workflow audit that the base branch
            # has since fixed — and only a rebase re-runs it against
            # the current base.  Mirror the engine ladder's
            # "stale failing verdict" rung here: when a blocked PR's
            # block reason is check-related and the compare API shows
            # the head demonstrably behind, refresh the branch before
            # treating the failure as terminal.
            needs_rebase = pr_info.mergeable_state == "behind"
            if (
                not needs_rebase
                and self.fix_out_of_date
                and not self.preview_mode
                and pr_info.mergeable_state == "blocked"
                and self._github_client is not None
            ):
                needs_rebase = await self._blocked_pr_needs_rebase(
                    pr_info, repo_owner, repo_name
                )
            if needs_rebase and self.fix_out_of_date:
                rebase_ctx = rebase.RebaseContext(
                    github_client=self._github_client,
                    token=self.token,
                    rebase_local=self.rebase_local,
                    preview_mode=self.preview_mode,
                    merge_recheck_interval=self._merge_recheck_interval,
                    merge_poll_max_attempts=self._merge_poll_max_attempts,
                    log=self.log,
                    console=self._console,
                    rebased_prs=self._rebased_prs,
                    enable_auto_merge=self._enable_auto_merge_with_approval,
                    track_pr_state=self._track_pr_state,
                )
                outcome = await rebase.perform_step5_rebase(
                    ctx=rebase_ctx,
                    pr_info=pr_info,
                    owner=repo_owner,
                    repo=repo_name,
                )
                if outcome.failed:
                    result.status = MergeStatus.FAILED
                    result.error = outcome.error_message
                    return result

            # If the PR is still blocked (e.g. by a pending
            # required status check such as pre-commit.ci), behind
            # base branch, or unstable (a non-required check failed),
            # enable auto-merge and wait for required checks to
            # complete. Skipped when:
            #   * preview_mode (no side effects)
            #   * force_level == "all" (force semantics bypass wait)
            #   * Step 5 already ran a rebase + wait for this PR
            #     (avoid doubling the configured merge_timeout)
            #   * mergeable_state == "blocked" for a reason that
            #     cannot resolve on its own (e.g. "requires approval",
            #     missing code-owner reviews) — waiting would just
            #     delay the inevitable failure/merge by up to
            #     merge_timeout.
            #
            # Note that ``behind`` PRs go through Step 5.5 regardless
            # of ``fix_out_of_date``: even when we will not rebase the
            # PR ourselves, enabling auto-merge gives GitHub the chance
            # to finish merging once required checks land, and the
            # resulting AUTO_MERGE_PENDING outcome is friendlier than
            # a 405 manual-merge failure. This also covers the brief
            # window after Dependabot/pre-commit-ci rebase a PR where
            # GitHub still reports ``behind`` while it recomputes
            # mergeability.
            #
            # We accept any ``mergeable`` value (including ``False``)
            # when the state is one of these auto-merge-rescuable
            # states, because GitHub returns ``mergeable=False``
            # transiently while computing the value or when a
            # non-required check failed. The block-reason pre-check
            # below still weeds out genuinely-stuck cases (missing
            # approvals, etc.) so we don't burn ``merge_timeout`` on
            # them.
            pr_key_for_wait = f"{repo_owner}/{repo_name}#{pr_info.number}"
            already_rebased = pr_key_for_wait in self._rebased_prs
            # ``unstable`` means a non-required check is failing or
            # pending but the PR is otherwise mergeable.  When GitHub
            # also reports ``mergeable is True`` the green button is
            # live and a direct merge succeeds *now*, so entering the
            # auto-merge wait would be pure waste: the state never
            # reaches ``clean`` (the non-required check stays red, e.g.
            # an excluded Zizmor scan), so the loop burns the full
            # ``merge_timeout``; and ``enablePullRequestAutoMerge``
            # is rejected outright on an already-mergeable PR, so the
            # wait isn't even backed by auto-merge.  Route those
            # straight to the Step 6 direct merge.  We still wait on
            # ``unstable`` when ``mergeable`` is not literally True
            # (GitHub still computing the value, or a required check
            # transiently failing) so a genuinely not-yet-ready PR is
            # not merged prematurely.
            state_is_waitable = pr_info.mergeable_state in ("blocked", "behind") or (
                pr_info.mergeable_state == "unstable" and pr_info.mergeable is not True
            )
            base_should_wait = (
                not self.preview_mode
                and self._github_client is not None
                and state_is_waitable
                and self.force_level != "all"
                and not already_rebased
            )

            # For ``blocked`` PRs (but not ``behind``, which only
            # needs a rebase to clear), consult analyze_block_reason
            # before entering the wait loop so we don't burn the
            # full merge_timeout on PRs blocked for reasons that
            # cannot resolve on their own.
            should_wait = base_should_wait
            if (
                base_should_wait
                and pr_info.mergeable_state == "blocked"
                and self._github_client is not None
            ):
                pre_block_reason: str | None = None
                try:
                    pre_block_reason = await self._github_client.analyze_block_reason(
                        repo_owner,
                        repo_name,
                        pr_info.number,
                        pr_info.head_sha,
                        base_branch=pr_info.base_branch,
                    )
                except Exception as exc:
                    self.log.debug(
                        "analyze_block_reason failed for %s during "
                        "Step 5.5 pre-check: %s",
                        pr_key_for_wait,
                        exc,
                    )
                    # Treat analysis failures as 'do not wait' so we
                    # don't burn the full ``merge_timeout`` on a PR
                    # whose block reason we cannot classify. The PR
                    # will fall through to the Step 6 skip gate (which
                    # also calls ``analyze_block_reason``) and either
                    # defer to auto-merge or surface a manual-merge
                    # error promptly.
                    should_wait = False

                if should_wait and pre_block_reason is not None:
                    if not self._block_reason_indicates_pending_checks(
                        pre_block_reason
                    ):
                        self.log.debug(
                            "Skipping Step 5.5 wait for %s: block "
                            "reason '%s' will not resolve on its own",
                            pr_key_for_wait,
                            pre_block_reason,
                        )
                        should_wait = False

            if should_wait:
                if pr_key_for_wait not in self._auto_merge_enabled:
                    # Approve-on-demand: arming auto-merge implies we want
                    # the PR to merge once checks pass, so approve the
                    # current head first (idempotent) before enabling.
                    auto_ok_pre = await self._enable_auto_merge_with_approval(
                        pr_info, repo_owner, repo_name
                    )
                    if auto_ok_pre:
                        self._pr_status(
                            f"🤖 Auto-merge: {pr_info.html_url}",
                            level="debug",
                        )

                # Wait (bounded by ``merge_timeout``) for required
                # checks to complete and auto-merge to fire.  The
                # continue-states mirror the ``base_should_wait`` entry
                # condition above (blocked / behind / unstable).
                self._track_pr_state(pr_info, "waiting")
                (
                    closed_during_wait,
                    merged_during_wait,
                ) = await self._wait_for_auto_merge(
                    pr_info,
                    repo_owner,
                    repo_name,
                    continue_states=("blocked", "behind", "unstable"),
                )
                self._track_pr_state(pr_info, None)

                # If the wait revealed the PR is already closed,
                # short-circuit before attempting a manual merge.
                # Distinguish auto-merge success from
                # closed-without-merge using the ``merged`` boolean
                # captured from the refresh payload.
                if closed_during_wait:
                    if merged_during_wait:
                        result.status = MergeStatus.MERGED
                        self._pr_status(
                            f"✅ Merged (auto-merge): {pr_info.html_url}",
                            level="debug",
                        )
                    else:
                        result.status = MergeStatus.CLOSED
                        result.error = (
                            "PR closed without merging during auto-merge wait "
                            "(no operator follow-up needed)"
                        )
                        self._pr_status(
                            f"🚪 Closed without merging: {pr_info.html_url}",
                            level="warning",
                        )
                    return result

                # The wait can expire with the PR still blocked on a
                # pre-commit.ci run that will never finish.  Step 0.5
                # only retriggers a run that was already stale when
                # processing *started*; a run that went pending shortly
                # before this run began crosses the stuck threshold
                # *during* the wait above, so without this re-check the
                # merge below fails on the pending check without the
                # recovery macro ever being posted.  The helper re-gates
                # on required-check status, pending age, and duplicate
                # comments, so this is a no-op unless the run is
                # genuinely stuck.
                if (
                    pr_info.mergeable_state == "blocked"
                    and self._github_client is not None
                ):
                    late_precommit_fixed = await self._trigger_stale_precommit_ci(
                        pr_info
                    )
                    if late_precommit_fixed:
                        # pre-commit.ci now reports success.  Auto-merge
                        # was armed before the wait, so GitHub may merge
                        # the PR the moment the check lands — re-fetch
                        # and short-circuit on a closed PR before
                        # attempting a manual merge below.
                        late_updated: Any = None
                        try:
                            late_updated = await self._github_client.get(
                                f"/repos/{repo_owner}/{repo_name}"
                                f"/pulls/{pr_info.number}"
                            )
                        except Exception as e:
                            self.log.debug(
                                "Failed to refresh PR %s state after "
                                "post-wait pre-commit.ci rerun: %s",
                                pr_key_for_wait,
                                e,
                            )
                        if isinstance(late_updated, dict):
                            if late_updated.get("state") == "closed":
                                pr_info.state = "closed"
                                if late_updated.get("merged", False):
                                    result.status = MergeStatus.MERGED
                                    self._pr_status(
                                        f"✅ Merged (auto-merge): {pr_info.html_url}",
                                        level="debug",
                                    )
                                else:
                                    result.status = MergeStatus.CLOSED
                                    result.error = (
                                        "PR closed without merging during "
                                        "auto-merge wait "
                                        "(no operator follow-up needed)"
                                    )
                                    self._pr_status(
                                        f"🚪 Closed without merging: "
                                        f"{pr_info.html_url}",
                                        level="warning",
                                    )
                                return result
                            pr_info.mergeable = late_updated.get("mergeable")
                            pr_info.mergeable_state = late_updated.get(
                                "mergeable_state"
                            )
                            updated_head = (late_updated.get("head") or {}).get("sha")
                            if updated_head:
                                pr_info.head_sha = updated_head

            result.status = MergeStatus.MERGING
            if self.preview_mode:
                self._simulate_preview_merge(pr_info, result)
            else:
                if self.progress_tracker:
                    self.progress_tracker.update_operation(
                        f"Merging PR {pr_info.number} in {pr_info.repository_full_name}"
                    )

                # If auto-merge is enabled and the PR is in a state
                # that auto-merge can rescue (blocked, behind, or
                # unstable), skip the manual merge attempt — GitHub
                # will merge automatically once branch protection is
                # satisfied.
                #
                # We accept any ``mergeable`` value (including
                # ``False``) here for the same reason Step 5.5 does:
                # ``mergeable=False`` from the API can mean
                # "definitely failing", "still computing", or "a
                # non-required check failed". Letting auto-merge
                # decide whether the failing thing actually blocks
                # merge is more accurate than us treating
                # ``False`` as terminal here.
                #
                # For ``blocked`` PRs we still consult
                # ``analyze_block_reason()`` to weed out cases
                # auto-merge cannot resolve (missing approvals,
                # code-owner reviews, etc.). For ``behind`` and
                # ``unstable`` we accept directly: ``behind``
                # resolves once GitHub re-runs checks against the
                # rebased commit, and ``unstable`` means a
                # non-required check failed (which doesn't actually
                # block auto-merge).
                #
                # Do NOT skip when:
                #   * force_level == "all" — force semantics
                #     intentionally proceed despite the blocked
                #     state and must not be overridden by
                #     auto-merge.
                #   * the block reason (for ``blocked`` PRs) is
                #     something other than pending required
                #     checks (e.g. missing approvals).
                pr_key = f"{repo_owner}/{repo_name}#{pr_info.number}"
                auto_merge_pending_checks = False
                if (
                    pr_key in self._auto_merge_enabled
                    and pr_info.mergeable_state in ("blocked", "behind", "unstable")
                    and self.force_level != "all"
                ):
                    if pr_info.mergeable_state in ("behind", "unstable"):
                        # ``behind``: still behind after rebase polling
                        # timed out; auto-merge will pick the PR up
                        # once GitHub finishes rebase + required
                        # checks.
                        # ``unstable``: a non-required check failed but
                        # required checks may still be pending or
                        # passing; auto-merge will fire when branch
                        # protection allows.
                        auto_merge_pending_checks = True
                    else:
                        block_reason: str | None = None
                        if self._github_client is not None:
                            try:
                                block_reason = (
                                    await self._github_client.analyze_block_reason(
                                        repo_owner,
                                        repo_name,
                                        pr_info.number,
                                        pr_info.head_sha,
                                        base_branch=pr_info.base_branch,
                                    )
                                )
                            except Exception as exc:
                                self.log.debug(
                                    "analyze_block_reason failed for %s: %s",
                                    pr_key,
                                    exc,
                                )
                        # Treat any pending-checks-style block reason
                        # as auto-merge eligible. We previously matched
                        # only the literal substring "pending required
                        # check", but analyze_block_reason returns a
                        # range of phrasings (e.g. "required status
                        # check… pending") depending on which
                        # checks are outstanding. The same predicate is
                        # used by the Step 5.5 pre-check.
                        auto_merge_pending_checks = (
                            self._block_reason_indicates_pending_checks(block_reason)
                        )

                if auto_merge_pending_checks:
                    merged = None  # Sentinel: auto-merge pending
                else:
                    # Proactive approval: some organizations mandate an
                    # approving review via a repository ruleset before
                    # *any* merge is allowed.  When this PR's base branch
                    # is governed that way a merge-first attempt is
                    # guaranteed to be rejected, so approve the current
                    # head up-front and skip the doomed round-trip plus
                    # reactive recovery.  See the helper for details; on
                    # any lookup failure it no-ops and the reactive
                    # approve-on-demand path still covers us.
                    await self._approve_if_review_mandated(
                        pr_info, repo_owner, repo_name, pr_key
                    )
                    # Serialise the actual merge dispatch per repo so
                    # back-to-back merges don't race GitHub's branch
                    # protection propagation.  Workers on the same
                    # repo queue here; workers on different repos run
                    # in parallel.  See ``_get_merge_dispatch_lock``.
                    dispatch_lock = await self._get_merge_dispatch_lock(
                        repo_owner, repo_name
                    )
                    dirty_before_dispatch = False
                    async with dispatch_lock:
                        # Re-read live merge state *before* dispatch — a
                        # single GET, no recompute poll.  In a
                        # repo-scoped batch an earlier sibling merge can
                        # turn this PR ``dirty`` between the one-shot
                        # fetch and dispatch (the classic shared
                        # ``uv.lock`` conflict); routing it straight to
                        # conflict recovery avoids dispatching a doomed
                        # merge that 405s and then churns the retry loop
                        # against the stale ``clean`` snapshot.  We keep
                        # this to a single GET (not the polling
                        # ``_refresh_pr_mergeability``) because the
                        # dispatch lock is the one point serialised *and*
                        # ordered after a sibling merge, so polling
                        # GitHub's recompute window here would serialise
                        # the whole repo batch.
                        if self._repo_scoped:
                            dirty_before_dispatch = await self._is_pr_dirty_now(
                                pr_info, repo_owner, repo_name
                            )
                        if dirty_before_dispatch:
                            merged = False
                        else:
                            merged = await self._merge_pr_with_retry(
                                pr_info, repo_owner, repo_name
                            )
                    # Conflict recovery runs *outside* the dispatch lock
                    # so the rebase wait never blocks sibling merges.
                    if dirty_before_dispatch:
                        return await self._handle_merge_conflict(
                            pr_info, repo_owner, repo_name, result
                        )
                    # A PR can also turn ``dirty`` *during* our own merge
                    # window (a sibling merged between the pre-dispatch
                    # check and the merge call).  The post-failure
                    # refresh — off the lock, with its recompute poll —
                    # catches that so a freshly-dirty PR is never
                    # reported as a generic merge failure.
                    if not merged and self._repo_scoped:
                        await self._refresh_pr_mergeability(
                            pr_info, repo_owner, repo_name
                        )
                        if pr_info.mergeable_state == "dirty":
                            return await self._handle_merge_conflict(
                                pr_info, repo_owner, repo_name, result
                            )

                    # Approve-on-demand (merge-path trigger): if the
                    # direct merge was rejected solely because our review
                    # is missing, approve the current head and retry once.
                    # Returns True only if the retry merged the PR; any
                    # other failure is left for the classifier below.
                    if not merged:
                        if await self._approve_and_retry_if_review_required(
                            pr_info, repo_owner, repo_name
                        ):
                            merged = True

                    # A 405 "Required workflows … are not satisfied"
                    # rejection means ruleset-required workflows are
                    # still *executing* on the head commit — a pending
                    # condition, unlike the terminal "… failed"
                    # variant.  Wait for them to finish and retry
                    # rather than failing a PR whose checks are still
                    # green.  Skipped when Step 5/5.5 already spent
                    # this PR's wait budget (the workflows are then
                    # genuinely slower than merge_timeout) and under
                    # --force=all (force semantics bypass waits).
                    if (
                        not merged
                        and not should_wait
                        and not already_rebased
                        and self.force_level != "all"
                    ):
                        last_merge_exc = self._last_merge_exception.get(pr_key)
                        if (
                            last_merge_exc is not None
                            and self._merge_error_indicates_pending_workflows(
                                str(last_merge_exc)
                            )
                        ):
                            merged = (
                                await self._wait_for_required_workflows_and_retry(
                                    pr_info, repo_owner, repo_name
                                )
                            )

                if merged is None:
                    # Auto-merge is active — PR will merge asynchronously.
                    # Tailor the reason to the actual ``mergeable_state``
                    # so the end-of-run summary shows what auto-merge is
                    # waiting on, rather than always "pending checks".
                    result.status = MergeStatus.AUTO_MERGE_PENDING
                    if pr_info.mergeable_state == "behind":
                        wait_reason = "behind base branch"
                    elif pr_info.mergeable_state == "unstable":
                        wait_reason = "non-required check failure"
                    else:
                        # ``blocked`` (the only other state that
                        # reaches this branch) routed through
                        # ``analyze_block_reason()`` and was
                        # classified as pending required checks by
                        # ``_block_reason_indicates_pending_checks``.
                        wait_reason = "pending checks"
                    result.error = f"auto-merge pending: {wait_reason}"
                    self._pr_status(
                        f"⏳ Waiting: {pr_info.html_url} [{wait_reason}]",
                        level="debug",
                    )
                elif merged:
                    result.status = MergeStatus.MERGED
                    self._pr_status(
                        f"✅ Merged: {pr_info.html_url}",
                        level="debug",
                    )
                else:
                    # A failed merge attempt can mask two benign
                    # races: the PR merged externally (a concurrent
                    # dependamerge run at org scope, or a human
                    # admin), or the PR closed without merging
                    # (dependabot decided the update is no longer
                    # needed after sibling merges advanced the
                    # base).  Neither outcome needs human follow-up,
                    # so classify as SKIPPED / CLOSED rather than
                    # FAILED.
                    ext_state, ext_merged = await self._fetch_pr_state_now(
                        pr_info, repo_owner, repo_name
                    )
                    if ext_state == "closed" and ext_merged:
                        result.status = MergeStatus.SKIPPED
                        result.error = "already merged externally"
                        self._pr_status(
                            f"⏭️ Skipped: {pr_info.html_url} "
                            "[already merged externally]",
                            level="info",
                        )
                        return result
                    if ext_state == "closed":
                        result.status = MergeStatus.CLOSED
                        result.error = (
                            "PR closed without merging during the run "
                            "(no operator follow-up needed)"
                        )
                        self._pr_status(
                            f"🚪 Closed without merging: {pr_info.html_url}",
                            level="info",
                        )
                        return result

                    # Compute failure summary once — used for both the
                    # recreate decision and the final error reporting.
                    failure_reason = self._get_failure_summary(pr_info)

                    # Before giving up, check if this is a dependabot PR
                    # that failed due to unsigned commits.  If so, ask
                    # dependabot to recreate the PR and merge the new one.
                    #
                    # Two recreate triggers are considered:
                    #   1. Branch-protection failures (the original
                    #      unsigned-commit case).
                    #   2. A *required* verification check that has
                    #      been stuck (queued / in_progress / pending)
                    #      for longer than
                    #      ``STUCK_CHECK_THRESHOLD_SECONDS`` on a PR
                    #      that itself was created and last updated
                    #      that long ago. Required checks (DCO, lint,
                    #      build, etc.) normally start reporting in
                    #      seconds; when one stalls indefinitely, the
                    #      only reliable recovery for dependabot PRs
                    #      is to recreate the PR so the checks fire
                    #      again on a fresh head SHA. pre-commit.ci is
                    #      excluded here — it has its own dedicated
                    #      recovery via ``_trigger_stale_precommit_ci``
                    #      (which posts ``pre-commit.ci run``).
                    recreated_pr = None
                    if is_dependabot(pr_info.author) and not self.preview_mode:
                        reason_lower = failure_reason.lower()
                        # Branch protection *and* repository rulesets can
                        # both block a dependabot PR for reasons recreation
                        # resolves (most commonly an unsigned-commit /
                        # required-signature rule).  Treat them alike so the
                        # recreate path is not silently skipped on repos that
                        # have migrated from classic protection to rulesets.
                        should_recreate = (
                            "branch protection" in reason_lower
                            or "ruleset" in reason_lower
                        )
                        if not should_recreate:
                            try:
                                (
                                    is_stuck,
                                    stuck_check,
                                    stuck_age,
                                ) = await self._detect_stuck_required_check(pr_info)
                            except Exception as exc:
                                self.log.debug(
                                    "_detect_stuck_required_check failed for "
                                    "%s#%s: %s",
                                    pr_info.repository_full_name,
                                    pr_info.number,
                                    exc,
                                )
                                is_stuck = False
                                stuck_check = None
                                stuck_age = 0.0
                            if is_stuck:
                                self._pr_status(
                                    f"⏳ Stuck required check detected: "
                                    f"{pr_info.html_url} "
                                    f"[{stuck_check} pending for "
                                    f"{stuck_age:.0f}s, requesting recreate]",
                                    level="info",
                                )
                                should_recreate = True
                        if should_recreate:
                            self._track_pr_state(pr_info, "recreating")
                            try:
                                recreated_pr = await self._trigger_dependabot_recreate(
                                    pr_info
                                )
                            finally:
                                self._track_pr_state(pr_info, None)

                    if recreated_pr is not None:
                        # We have a fresh PR — approve and merge it
                        new_owner, new_repo = recreated_pr.repository_full_name.split(
                            "/", 1
                        )
                        await self._approve_pr(new_owner, new_repo, recreated_pr.number)

                        new_merge_method = self._pr_merge_methods.get(
                            f"{new_owner}/{new_repo}", self.default_merge_method
                        )
                        try:
                            if self._github_client is None:
                                raise RuntimeError("GitHub client not initialized")
                            # Same per-repo dispatch serialisation as
                            # the main merge path — see
                            # ``_get_merge_dispatch_lock``.
                            new_dispatch_lock = await self._get_merge_dispatch_lock(
                                new_owner, new_repo
                            )
                            async with new_dispatch_lock:
                                new_merged = (
                                    await self._github_client.merge_pull_request(
                                        new_owner,
                                        new_repo,
                                        recreated_pr.number,
                                        new_merge_method,
                                    )
                                )
                        except Exception as merge_err:
                            self.log.error(
                                "Failed to merge recreated PR %s#%s: %s",
                                recreated_pr.repository_full_name,
                                recreated_pr.number,
                                merge_err,
                            )
                            new_merged = False

                        if new_merged:
                            result.status = MergeStatus.MERGED
                            result.pr_info = recreated_pr
                            self._pr_status(
                                f"✅ Merged (recreated): {recreated_pr.html_url}",
                                level="debug",
                            )
                        else:
                            result.status = MergeStatus.FAILED
                            result.error = (
                                f"Dependabot recreated PR #{recreated_pr.number} "
                                "but merge still failed"
                            )
                            self.log.error(
                                "Failed to merge recreated PR %s#%s",
                                recreated_pr.repository_full_name,
                                recreated_pr.number,
                            )
                            self._pr_status(
                                f"❌ Failed: {recreated_pr.html_url} "
                                "[recreated PR merge failed]",
                                level="error",
                            )
                    else:
                        await self._report_merge_failure(
                            pr_info,
                            repo_owner,
                            repo_name,
                            result,
                            failure_reason,
                        )

        except GitHubPermissionError as e:
            # Handle permission errors with detailed guidance.
            #
            # When the token lacks rights on a repository the same
            # error fires for every PR processed.  Record the repo
            # so subsequent PRs in the batch short-circuit via the
            # fast-fail check at the top of this method, and emit
            # the verbose guidance block only the first time we
            # see the failure for a given repository.
            result.status = MergeStatus.FAILED
            result.error = str(e)

            first_failure_for_repo = (
                pr_info.repository_full_name not in self._permission_failed_repos
            )
            self._permission_failed_repos.add(pr_info.repository_full_name)

            operation_desc = e.operation.replace("_", " ")
            self._pr_status(
                f"❌ Failed: {pr_info.html_url} [permission denied: {operation_desc}]",
                level="error",
            )

            if not first_failure_for_repo:
                # Already printed the full guidance for this repo;
                # do not repeat it for every remaining PR.
                return result

            # Provide token-specific guidance (printed once per repo)
            self._console.print("\n💡 Token Permission Issue:")
            self._console.print(f"   Problem: {e}")

            if e.token_type_guidance:
                self._console.print("\n   For Classic Tokens:")
                self._console.print(
                    f"   • {e.token_type_guidance.get('classic', 'Check token scopes')}"
                )
                self._console.print("\n   For Fine-Grained Tokens:")
                self._console.print(
                    f"   • {e.token_type_guidance.get('fine_grained', 'Check token permissions')}"
                )
                if "fix" in e.token_type_guidance:
                    self._console.print("\n   Quick Fix:")
                    self._console.print(f"   • {e.token_type_guidance['fix']}")

            self._console.print()

        except Exception as e:
            result.status = MergeStatus.FAILED
            result.error = str(e)

            # Provide clean single-line error messages for other errors.
            # The stack trace is attached only when the logger is in
            # DEBUG mode (i.e. the user passed ``--verbose``).  In the
            # default WARNING setup the trace would otherwise be
            # printed to stderr for every failure, swamping a
            # repo-scoped batch run with several hundred lines of
            # noise per PR when the underlying cause is something
            # uniform (e.g. token without the required scope) that a
            # single clean line already conveys.
            self.log.error(
                "Failed to process PR %s: %s",
                pr_info.html_url,
                e,
                exc_info=self.log.isEnabledFor(logging.DEBUG),
            )
            self._pr_status(
                f"❌ Failed: {pr_info.html_url} [processing error: {e}]",
                level="error",
            )

        finally:
            result.duration = time.time() - start_time
            # Clean up recently-approved tracking to avoid unbounded growth
            pr_key = f"{repo_owner}/{repo_name}#{pr_info.number}"
            self._recently_approved.discard(pr_key)

        return result

    async def _handle_not_mergeable_pr(
        self, pr_info: PullRequestInfo, result: MergeResult
    ) -> MergeResult:
        """Classify and report a PR that failed the mergeability gate.

        Extracted from ``_merge_single_pr`` to keep that method's
        branch count manageable. Produces a detailed skip/block
        reason, sets ``result`` accordingly, and returns it.
        """
        # Get detailed status for a more informative skip message
        # Use async method to avoid event loop conflicts
        repo_owner, repo_name = pr_info.repository_full_name.split("/")

        # Check if blocked to get more detailed status
        if pr_info.mergeable_state == "blocked" and self._github_client:
            try:
                detailed_status = await self._github_client.analyze_block_reason(
                    repo_owner,
                    repo_name,
                    pr_info.number,
                    pr_info.head_sha,
                    base_branch=pr_info.base_branch,
                )
            except Exception:
                detailed_status = f"Blocked (state: {pr_info.mergeable_state})"
        else:
            # For non-blocked states, provide basic status
            if pr_info.mergeable_state == "dirty":
                detailed_status = "Merge conflicts"
            elif pr_info.mergeable_state == "behind":
                detailed_status = "Rebase required (out of date)"
            elif pr_info.mergeable_state == "draft":
                detailed_status = "Draft PR"
            else:
                detailed_status = f"Not mergeable (state: {pr_info.mergeable_state})"

        # Use the detailed status as the skip reason, with fallback
        if detailed_status and detailed_status != "Status unclear":
            skip_reason = detailed_status.lower()
        else:
            # Fallback to basic mapping if detailed status is unclear
            if pr_info.mergeable_state == "dirty":
                skip_reason = "merge conflicts"
            elif pr_info.mergeable_state == "behind":
                skip_reason = "behind"
            elif pr_info.mergeable_state == "blocked":
                if pr_info.mergeable is True:
                    skip_reason = "blocked, requires review"
                else:
                    skip_reason = "blocked by failing checks"
            elif pr_info.mergeable_state == "unstable":
                skip_reason = "unstable"
            elif pr_info.mergeable is False:
                skip_reason = "not mergeable"
            else:
                skip_reason = "unknown"

        # Determine if this is truly blocked (unmergeable) or just skipped
        if pr_info.mergeable_state == "dirty" or (
            pr_info.mergeable_state == "behind" and pr_info.mergeable is False
        ):
            result.status = MergeStatus.BLOCKED
            icon = "🛑"
            status = "Blocked"
        else:
            result.status = MergeStatus.SKIPPED
            icon = "⏭️"
            status = "Skipped"

        self._pr_status(
            f"{icon} {status}: {pr_info.html_url} [{skip_reason}]",
            level="info",
        )

        result.error = f"PR is not mergeable (state: {pr_info.mergeable_state}, mergeable: {pr_info.mergeable})"

        # For the result error (used in CLI output), use the detailed status if it's more informative
        if detailed_status and detailed_status != "Status unclear":
            result.error = detailed_status

        return result

    def _simulate_preview_merge(
        self, pr_info: PullRequestInfo, result: MergeResult
    ) -> None:
        """Simulate the Step 6 merge outcome for preview mode.

        Preview output must be SINGLE LINE per PR for clean evaluation
        display: each PR should produce exactly one line under the
        "🔍 Dependamerge Evaluation" heading. Mutates ``result`` in place.
        """
        if pr_info.mergeable_state == "behind" and not self.fix_out_of_date:
            result.status = MergeStatus.SKIPPED
            result.error = "PR is behind base branch and --no-fix option is set"
            self._console.print(
                f"⏭️ Skipped: {pr_info.html_url} [behind, rebase disabled]",
                markup=False,
            )
        elif pr_info.mergeable_state == "behind" and self.fix_out_of_date:
            # For behind PRs with fix enabled, show warning with rebase info
            result.status = MergeStatus.MERGED  # Would succeed after rebase
            # Use ``warning`` (not ``error``) so the MERGED result
            # does not carry a contradictory error message.
            result.warning = "behind base branch"
            self._console.print(
                f"⚠️ Rebase/merge: {pr_info.html_url} [behind base branch]",
                markup=False,
            )
        elif pr_info.mergeable_state == "dirty":
            result.status = MergeStatus.BLOCKED
            result.error = "PR has merge conflicts"
            self._console.print(
                f"🛑 Blocked: {pr_info.html_url} [merge conflicts]",
                markup=False,
            )
        elif pr_info.mergeable is False and pr_info.mergeable_state == "blocked":
            result.status = MergeStatus.BLOCKED
            result.error = "PR blocked by failing checks"
            self._console.print(
                f"🛑 Blocked: {pr_info.html_url} [blocked by failing checks]",
                markup=False,
            )
        else:
            # Simulate successful merge in preview mode
            result.status = MergeStatus.MERGED
            # Single line summary for successful preview
            log_and_print(
                self.log,
                self._console,
                f"☑️ Approve/merge: {pr_info.html_url}",
                level="debug",
            )

    @staticmethod
    def _block_reason_indicates_pending_checks(
        block_reason: str | None,
    ) -> bool:
        """Return True if a block reason indicates pending required checks.

        Both Step 5.5 (whether to enter the wait loop) and Step 6
        (whether to defer to auto-merge instead of attempting a
        manual merge) need to recognise the same set of phrasings
        returned by ``GitHubAsync.analyze_block_reason()``.
        Centralising the predicate here keeps the two call sites
        consistent so a new phrasing only has to be added once.

        The predicate matches **only** wording that explicitly
        signals a check is still in progress / waiting to start.
        It deliberately excludes:

        - ``Blocked by failing check: …`` — the check has run and
          reported a non-pending failure; auto-merge will not
          rescue this.
        - ``Blocked by missing required status: …`` — the check
          has not been registered against the commit at all;
          auto-merge will not retry it on its own.
        - any reason where ``failing`` or ``missing`` appears
          before a service name (defensive: covers future GitHub
          phrasing changes that include both keywords).

        Args:
            block_reason: The string returned by
                ``analyze_block_reason()``, or ``None`` if the
                analysis failed or returned nothing.

        Returns:
            True when the reason mentions pending required checks
            in any of the recognised phrasings; False otherwise
            (including when ``block_reason`` is ``None``).
        """
        if block_reason is None:
            return False
        reason_lower = block_reason.lower()

        # Defensive negative gate: never classify a reason as
        # 'pending' if it explicitly says the check has failed or
        # is missing. This guards the bare-substring matches
        # below against future phrasings that combine both terms
        # (e.g. "failing check (pending retry): pre-commit.ci").
        if "failing check" in reason_lower:
            return False
        if "missing required status" in reason_lower:
            return False
        if "missing required check" in reason_lower:
            return False

        return (
            "pending required check" in reason_lower
            or "pending check" in reason_lower
            or ("required" in reason_lower and "pending" in reason_lower)
            or "waiting for status" in reason_lower
            or "queued" in reason_lower
        )

    @staticmethod
    def _block_reason_indicates_check_blockage(
        block_reason: str | None,
    ) -> bool:
        """Return True if a block reason concerns status checks at all.

        Broader sibling of
        :meth:`_block_reason_indicates_pending_checks`: matches any
        ``analyze_block_reason()`` phrasing about checks — failing,
        missing, or pending — while rejecting reasons a rebase cannot
        influence (missing approvals, requested changes, unresolved
        Copilot feedback, opaque ruleset blocks).

        Step 5 uses this to decide whether a ``blocked`` PR is worth
        probing for staleness: refreshing the branch re-runs checks
        against the current base, so only check-related blockage can
        possibly be cured by a rebase.

        Args:
            block_reason: The string returned by
                ``analyze_block_reason()``, or ``None`` if the
                analysis failed or returned nothing.

        Returns:
            True when the reason mentions failing, missing, or
            pending checks; False otherwise (including ``None``).
        """
        if block_reason is None:
            return False
        reason_lower = block_reason.lower()
        return (
            "failing check" in reason_lower
            or "missing required status" in reason_lower
            or "missing required check" in reason_lower
            or "pending required check" in reason_lower
            or ("required" in reason_lower and "pending" in reason_lower)
            or "waiting for status" in reason_lower
            or "queued" in reason_lower
        )

    async def _blocked_pr_needs_rebase(
        self,
        pr_info: PullRequestInfo,
        repo_owner: str,
        repo_name: str,
    ) -> bool:
        """Decide whether a ``blocked`` PR is really stale-and-fixable.

        Implements the staleness probe behind Step 5's
        blocked-masks-behind handling: a ``blocked`` PR is treated
        like ``behind`` when **both** hold:

        1. Its block reason is check-related (failing, missing, or
           pending checks) — the only class of blockage a branch
           refresh can cure, because the refresh re-runs checks
           against the current base.
        2. The compare API confirms the head is at least one commit
           behind the base branch.  ``None`` (comparison failed)
           counts as "not behind": a rebase is a write action and a
           CI-time expense, so it must rest on positive evidence.

        The two probes run in this order so the cheaper classification
        gates the extra compare call.

        Args:
            pr_info: The pull request under evaluation.
            repo_owner: Base repository owner.
            repo_name: Base repository name.

        Returns:
            True when the PR should take the Step 5 rebase path.
        """
        if self._github_client is None:
            return False
        pr_key = f"{repo_owner}/{repo_name}#{pr_info.number}"
        block_reason: str | None = None
        try:
            block_reason = await self._github_client.analyze_block_reason(
                repo_owner,
                repo_name,
                pr_info.number,
                pr_info.head_sha,
                base_branch=pr_info.base_branch,
            )
        except Exception as exc:
            self.log.debug(
                "analyze_block_reason failed for %s during Step 5 "
                "staleness probe: %s",
                pr_key,
                exc,
            )
            return False
        if not self._block_reason_indicates_check_blockage(block_reason):
            return False

        behind_by = await self._github_client.get_behind_by(
            repo_owner,
            repo_name,
            pr_info.base_branch or "main",
            pr_info.head_sha,
        )
        if behind_by is None or behind_by <= 0:
            return False

        pr_info.behind_by = behind_by
        self._pr_status(
            f"\U0001f504 Stale head: {pr_info.html_url} "
            f"[blocked ({block_reason}); {behind_by} commit(s) behind "
            f"base — rebasing to re-run checks]",
            level="debug",
        )
        return True

    def _is_pr_mergeable(self, pr_info: PullRequestInfo) -> bool:
        """Check whether a PR is worth attempting to merge.

        This returns ``True`` for any state where dependamerge can
        plausibly make progress — either by approving + merging,
        rebasing, or enabling auto-merge and waiting (Step 5.5).
        We deliberately err on the side of letting Step 5.5 see the
        PR: it has finer-grained logic (block-reason analysis,
        merge-timeout-bounded waits) than this gate, so a False here
        denies a PR the chance to be auto-merge-rescued.

        Returns False only for states where no amount of waiting,
        approving, or auto-merging can help:

        - ``dirty``: real merge conflict; the branch must be
          rebased by a human (or by ``--fix``).
        - ``draft``: GitHub blocks merging draft PRs by design.

        For all other states (``blocked``, ``behind``, ``unstable``,
        empty/``"unknown"``) we return True regardless of the
        ``mergeable`` boolean. ``mergeable=False`` from the API can
        mean "definitely failing", but it can also mean "GitHub is
        still computing" or "a non-required check failed" — the
        downstream Step 5.5 + Step 6 gates have the context to make
        the right call.
        """
        # Hard skips: states where merging is impossible regardless
        # of mergeable value or downstream rescue logic.
        if pr_info.mergeable_state == "dirty":
            self.log.debug(
                "🛑 Skipping PR %s/%s#%s: merge conflict (dirty)",
                pr_info.repository_full_name.split("/", 1)[0]
                if "/" in pr_info.repository_full_name
                else pr_info.repository_full_name,
                pr_info.repository_full_name.split("/", 1)[-1],
                pr_info.number,
            )
            return False
        if pr_info.mergeable_state == "draft":
            self.log.debug(
                "⏭️ Skipping draft PR %s#%s",
                pr_info.repository_full_name,
                pr_info.number,
            )
            return False

        # Everything else — ``blocked``, ``behind``, ``unstable``,
        # ``clean``, empty/None state, plus any ``mergeable`` value
        # — reaches the merge flow. Step 5.5 will route
        # not-yet-merge-ready cases to AUTO_MERGE_PENDING after
        # consulting block-reason analysis and bounded by
        # ``merge_timeout``, which is a much friendlier outcome than
        # a hard skip from here.
        self.log.debug(
            "✅ PR %s#%s eligible for merge flow (mergeable=%s, state=%s)",
            pr_info.repository_full_name,
            pr_info.number,
            pr_info.mergeable,
            pr_info.mergeable_state,
        )
        return True

    def _has_blocking_reviews(self, pr_info: PullRequestInfo) -> bool:
        """
        Check if a PR has reviews that would block automatic approval.

        Args:
            pr_info: Pull request information

        Returns:
            True if there are blocking reviews (changes requested), False otherwise
        """
        for review in pr_info.reviews:
            if review.state == "CHANGES_REQUESTED":
                self.log.info(
                    f"⚠️ PR {pr_info.number} has changes requested by {review.user} - will not override human feedback"
                )
                return True
        return False

    async def _post_pr_comment_with_retry(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        html_url: str,
        body: str,
    ) -> bool:
        """Post a PR comment with one retry after a 5s pause.

        Used for the auto-merge audit-trail comment so the PR
        conversation reflects that dependamerge enabled auto-merge.
        Approval comments take a different path: the approval body
        is passed directly to ``approve_pull_request()``, which
        creates a review (not an issue comment), so this helper is
        not used there. If both attempts fail, emit a single
        user-visible warning to the console rather than silently
        failing.

        Args:
            owner: Repository owner.
            repo: Repository name.
            pr_number: Pull request number.
            html_url: Full PR URL, used for the warning message.
            body: Markdown body of the comment.

        Returns:
            True if the comment posted successfully (first or
            second attempt), False otherwise.
        """
        if not self._github_client:
            return False

        for attempt in (1, 2):
            try:
                await self._github_client.post_issue_comment(
                    owner, repo, pr_number, body
                )
                return True
            except GitHubPermissionError as exc:
                # Permission errors (typically HTTP 403) are not
                # transient — the token lacks the required scope or
                # the repo's branch protection forbids comments.
                # Skip the retry to avoid a pointless 5s delay per
                # PR and surface the failure right away.
                self.log.debug(
                    "Audit comment post denied (permission) for %s: %s",
                    html_url,
                    exc,
                )
                break
            except Exception as exc:
                # Heuristic: treat 4xx (other than 408/429) as
                # permanent and skip the retry. 5xx, 429 (rate
                # limit), 408 (timeout), and network/transport
                # errors get one retry after a short pause.
                #
                # We check several attribute paths so the
                # heuristic works across the various exception
                # shapes we may see:
                #   * ``exc.status_code`` — some custom wrappers
                #   * ``exc.status`` — ``aiohttp.ClientResponseError``
                #   * ``exc.response.status_code`` — ``httpx`` raises
                #     ``HTTPStatusError`` whose status lives on the
                #     attached ``Response`` object (this is what
                #     ``httpx.Response.raise_for_status()`` produces).
                response = getattr(exc, "response", None)
                status_code = (
                    getattr(exc, "status_code", None)
                    or getattr(exc, "status", None)
                    or getattr(response, "status_code", None)
                    or getattr(response, "status", None)
                )
                permanent = (
                    isinstance(status_code, int)
                    and 400 <= status_code < 500
                    and status_code not in (408, 429)
                )
                self.log.debug(
                    "Audit comment post attempt %d failed for %s: %s"
                    " (status=%r, permanent=%s)",
                    attempt,
                    html_url,
                    exc,
                    status_code,
                    permanent,
                )
                if permanent:
                    break
                if attempt == 1:
                    await asyncio.sleep(5.0)

        # Both attempts failed — surface a single line so the
        # user knows the PR-side audit trail is incomplete.
        self._pr_status(
            f"⚠️ Unable to add pull request comment: {html_url}",
            level="warning",
        )
        return False

    async def _enable_auto_merge_for_pr(
        self, pr_info: PullRequestInfo, owner: str, repo: str
    ) -> bool:
        """Enable auto-merge on a PR so it merges when checks pass.

        Idempotent and safe to call when auto-merge may already be
        active. Outcomes:

        - GraphQL ``enablePullRequestAutoMerge`` mutation succeeds:
          add the PR to ``_auto_merge_enabled``, post the audit
          comment, and return ``True``.
        - GraphQL mutation reports failure (commonly because
          auto-merge is *already* active on the PR — the response
          omits ``autoMergeRequest`` or the request raises): fall
          back to a REST GET on the PR and inspect ``auto_merge``.
          If non-null, treat as already-enabled — add the PR to
          ``_auto_merge_enabled`` (so the Step 6 skip gate still
          routes it to ``AUTO_MERGE_PENDING`` rather than
          attempting a 405-prone manual merge) and return ``True``,
          but skip the audit comment so re-runs against an
          already-configured PR don't post duplicates.
        - GraphQL mutation reports failure AND the PR has no
          ``auto_merge`` set: auto-merge is genuinely unavailable
          (e.g. the repository setting is off, the PR has
          conflicts, no required-checks are configured). Return
          ``False`` and let the caller fall through to manual
          polling/merge.
        - The PR has no ``node_id`` or there is no GitHub client:
          return ``False`` without making any API calls.

        Args:
            pr_info: Pull request information (must have ``node_id``).
            owner: Repository owner.
            repo: Repository name.

        Returns:
            True if auto-merge is active on the PR after this call
            (whether enabled by this call or already-active before
            it). False if auto-merge is unavailable.
        """
        if not self._github_client:
            return False

        if not pr_info.node_id:
            self.log.debug(
                "Cannot enable auto-merge for %s/%s#%s: missing node_id",
                owner,
                repo,
                pr_info.number,
            )
            return False

        pr_key = f"{owner}/{repo}#{pr_info.number}"

        # Already enabled in this run — skip duplicate API call
        if pr_key in self._auto_merge_enabled:
            return True

        cache_key = f"{owner}/{repo}"
        merge_method = self._pr_merge_methods.get(cache_key, self.default_merge_method)

        enabled = await self._github_client.enable_auto_merge(
            pr_info.node_id, merge_method
        )
        if not enabled:
            # The GraphQL mutation reports failure when auto-merge
            # is already active on the PR (the response omits
            # ``autoMergeRequest`` or the request raises). Check
            # the PR's current auto-merge state via REST so the
            # Step 6 skip gate still routes the PR to
            # ``AUTO_MERGE_PENDING`` instead of falling through to
            # a manual merge attempt that would 405 on pending
            # required checks.
            try:
                pr_payload = await self._github_client.get(
                    f"/repos/{owner}/{repo}/pulls/{pr_info.number}"
                )
            except Exception as exc:
                self.log.debug(
                    "Could not refresh PR %s to check existing auto-merge state: %s",
                    pr_key,
                    exc,
                )
                pr_payload = None

            if (
                isinstance(pr_payload, dict)
                and pr_payload.get("auto_merge") is not None
            ):
                self._auto_merge_enabled.add(pr_key)
                self.log.debug(
                    "Auto-merge already active for %s; treating "
                    "as enabled (no audit comment posted)",
                    pr_key,
                )
                # Skip the audit comment in this branch —
                # someone (a previous run, the author, or the
                # repo's auto-merge bot) already enabled it; we
                # don't want to post a duplicate comment every
                # time dependamerge runs.
                return True
            return False

        self._auto_merge_enabled.add(pr_key)
        self.log.debug(
            "Auto-merge enabled for %s (method=%s)",
            pr_key,
            merge_method,
        )
        # Post a visible audit-trail comment so reviewers can
        # see at a glance that dependamerge enabled auto-merge
        # on the PR.
        audit_comment = (
            "🤖 Dependamerge\nEnabled auto-merge due to pending updates/checks ⏳"
        )
        await self._post_pr_comment_with_retry(
            owner, repo, pr_info.number, pr_info.html_url, audit_comment
        )
        return True

    async def _ensure_pr_approved(
        self,
        pr_info: PullRequestInfo,
        owner: str,
        repo: str,
        *,
        propagation_delay: bool = True,
    ) -> bool:
        """Approve the current PR head on demand and track the approval.

        Thin wrapper around :meth:`_approve_pr` that also records the PR
        in ``_recently_approved`` and applies the post-approval
        propagation delay, exactly as the (now removed) up-front Step 3
        approval used to.  :meth:`_approve_pr` is idempotent — it no-ops
        when the current user already has an active ``APPROVED`` review on
        the current head — so this is safe to call unconditionally at any
        approve-on-demand trigger.

        ``propagation_delay=False`` skips the post-approval sleep.  The
        delay exists to let GitHub propagate the approval into branch-
        protection evaluation before an *immediate* merge dispatch; when
        the caller is arming auto-merge instead (GitHub re-evaluates
        protection when checks complete, typically minutes later) the
        sleep is pure dead time on the critical path.

        Returns:
            True if a *new* approval was submitted, False if the PR was
            already approved (or approval was declined).
        """
        approved = await self._approve_pr(owner, repo, pr_info.number)
        if approved:
            pr_key = f"{owner}/{repo}#{pr_info.number}"
            self._recently_approved.add(pr_key)
            # Give GitHub time to propagate the approval to the branch
            # protection evaluation before a merge is attempted.
            if propagation_delay and self._post_approval_delay > 0:
                self.log.debug(
                    f"Waiting {self._post_approval_delay}s for approval "
                    f"propagation on {pr_key}"
                )
                await asyncio.sleep(self._post_approval_delay)
        return approved

    async def _enable_auto_merge_with_approval(
        self, pr_info: PullRequestInfo, owner: str, repo: str
    ) -> bool:
        """Approve the current head (if needed) then enable auto-merge.

        Enabling auto-merge is a commitment to let GitHub complete the
        merge as soon as branch protection is satisfied, so the current
        head must already carry our approval — otherwise auto-merge would
        wait forever on a missing review.  This is the *de-facto*
        approve-on-demand trigger for the auto-merge path: when we enable
        auto-merge on a PR whose current version we have not approved, we
        approve it as part of arming auto-merge.

        Approval failures other than typed permission errors are logged
        and swallowed so a transient approval hiccup does not prevent us
        from at least arming auto-merge; the permission error is
        propagated so the caller's dedicated handler can report it.

        Used at the Step 5.5 auto-merge enable point and as the rebase
        module's auto-merge callback (which fires *after* the rebase, so
        we approve the rebased head rather than a soon-to-be-dismissed
        pre-rebase commit).
        """
        if not self.preview_mode:
            try:
                # No propagation delay here: we are arming auto-merge,
                # not dispatching an immediate merge.  GitHub re-checks
                # branch protection when the required checks complete
                # (≫ the propagation window), so sleeping now would
                # only stall the pipeline.
                await self._ensure_pr_approved(
                    pr_info, owner, repo, propagation_delay=False
                )
            except GitHubPermissionError:
                # Surface token permission problems to the caller's
                # dedicated handler rather than masking them here.
                raise
            except Exception as exc:
                self.log.warning(
                    "Could not approve %s/%s#%s before enabling auto-merge: %s",
                    owner,
                    repo,
                    pr_info.number,
                    exc,
                )
        return await self._enable_auto_merge_for_pr(pr_info, owner, repo)

    async def _approve_and_retry_if_review_required(
        self, pr_info: PullRequestInfo, owner: str, repo: str
    ) -> bool:
        """Approve-on-demand recovery after a failed direct merge.

        This is the merge-path approve-on-demand trigger: rather than
        approving every PR up-front, we attempt the merge first and only
        approve when GitHub rejects it *specifically* because our review
        is missing.  This avoids approving PRs that did not need it (e.g.
        a PR that fails for an unrelated reason).

        Called only after a direct merge attempt returned ``False``.  It
        consults :meth:`analyze_block_reason`; if (and only if) the PR is
        blocked pending approval and we have not already approved it this
        run, it approves the current head and retries the merge once.

        Returns:
            True if the approve-then-retry merged the PR, False otherwise
            (including when the failure was not approval-related, so the
            caller can proceed to its normal failure handling).
        """
        if self.preview_mode or self._github_client is None:
            return False

        pr_key = f"{owner}/{repo}#{pr_info.number}"
        if pr_key in self._recently_approved:
            # We already approved this PR this run; the retry machinery in
            # _merge_pr_with_retry has already had its post-approval
            # propagation retry, so a missing approval is not the cause.
            return False

        # Prefer GitHub's own merge-rejection body over the heuristic
        # block-reason classifier.  When the merge endpoint refuses the
        # merge it states the *authoritative* reason in the response body
        # (captured in the stored exception), e.g. "Repository rule
        # violations found Waiting on required approvals from <team>".
        # The heuristic ``analyze_block_reason`` only reports a single,
        # highest-priority reason and ranks the missing-approval condition
        # below required-status checks, so an unrelated or false-positive
        # "missing required status" (e.g. a DCO check GitHub does not
        # actually gate the merge on) masks the real cause and the
        # approve-on-demand recovery never fires.  Trust GitHub first.
        #
        # This authoritative check runs *regardless* of mergeable_state:
        # that field lags and is blind to repository rulesets, so GitHub
        # can reject a merge for a missing required approval even while the
        # cached state is not ``blocked``.  Gating it behind a ``blocked``
        # state would strand exactly the PRs this recovery exists to save.
        last_exception = self._last_merge_exception.get(pr_key)
        if last_exception is not None and self._merge_error_indicates_missing_approval(
            str(last_exception)
        ):
            self.log.debug(
                "Merge for %s was rejected by GitHub pending required "
                "approval; approving on demand and retrying",
                pr_key,
            )
            approved = await self._ensure_pr_approved(pr_info, owner, repo)
            if not approved:
                return False
            return await self._merge_pr_with_retry(pr_info, owner, repo)

        # Fall back to the heuristic block-reason classifier only when the
        # cached state actually shows the PR as ``blocked``.  A missing
        # review manifests as that state; a merge that fails from any other
        # state (e.g. a transient 405 on a ``clean`` PR) without an
        # authoritative approval signal above is not an approval problem,
        # so don't probe or approve it — let the caller's classifier
        # handle it.
        if pr_info.mergeable_state != "blocked":
            return False

        try:
            block_reason = await self._github_client.analyze_block_reason(
                owner,
                repo,
                pr_info.number,
                pr_info.head_sha,
                base_branch=pr_info.base_branch,
            )
        except Exception as exc:
            self.log.debug(
                "approve-on-demand block-reason check failed for %s: %s",
                pr_key,
                exc,
            )
            return False

        if not block_reason or "requires approval" not in block_reason.lower():
            # The merge failed for some reason other than a missing
            # review — do not approve; let the caller classify and report.
            return False

        self.log.debug(
            "Merge for %s was blocked pending approval; approving on "
            "demand and retrying",
            pr_key,
        )
        approved = await self._ensure_pr_approved(pr_info, owner, repo)
        if not approved:
            return False
        return await self._merge_pr_with_retry(pr_info, owner, repo)

    @staticmethod
    def _merge_error_indicates_missing_approval(error_text: str) -> bool:
        """Detect a missing-required-approval signal in a merge error body.

        GitHub's merge endpoint reports the authoritative rejection reason
        in its response body, which is preserved in the exception text
        raised by ``merge_pull_request``.  A merge blocked solely because
        our approving review is missing is recoverable: we can approve the
        head and retry.  This recognises the phrasings GitHub uses for
        both repository rulesets and classic branch protection, e.g.:

        - "Waiting on required approvals from <team>" (ruleset)
        - "At least 1 approving review is required by reviewers with
          write access." (branch protection)
        - "Required review ... review required"

        It deliberately does *not* match "changes requested" wording,
        which an approval cannot clear.
        """
        if not error_text:
            return False
        text = error_text.lower()
        return (
            "required approval" in text
            or "approving review" in text
            or "review required" in text
        )

    @staticmethod
    def _merge_error_indicates_pending_workflows(error_text: str) -> bool:
        """Detect still-executing required workflows in a merge error body.

        GitHub's merge endpoint rejects with 405 "Repository rule
        violations found … Required workflows 'X' are not satisfied"
        while ruleset-required workflows are still *executing* on the
        head commit — a pending condition that clears by itself once
        they finish.  That is distinct from the failure variant
        ("Required workflows 'X' … fail…"), where a workflow ran and
        reported failure: terminal, retrying cannot help.

        Only the clause starting at the ``required workflow`` wording
        is inspected, because the enhanced exception text always begins
        with "Failed to merge PR …" — matching ``fail`` against the
        whole message would classify every rejection as terminal.  A
        workflow *name* containing "fail" also suppresses the match;
        that conservative false negative merely preserves the previous
        fail-fast behaviour.
        """
        if not error_text:
            return False
        text = error_text.lower()
        idx = text.find("required workflow")
        if idx == -1:
            return False
        clause = text[idx:]
        # Trim the PR-state context ``_validate_merge_result`` appends
        # after GitHub's detail so it cannot influence classification.
        clause = clause.split(" (pr state:", 1)[0]
        return "not satisfied" in clause and "fail" not in clause

    async def _check_merge_requirements(
        self, pr_info: PullRequestInfo
    ) -> tuple[bool, str]:
        """
        Check if a PR meets all requirements for merging, including branch protection rules.

        Args:
            pr_info: Pull request information

        Returns:
            Tuple of (can_merge: bool, reason: str)
        """
        if not self._github_client:
            return False, "GitHub client not initialized"

        repo_owner, repo_name = pr_info.repository_full_name.split("/")

        try:
            base_branch = pr_info.base_branch or "main"
            protection_rules = await self._github_client.get_branch_protection(
                repo_owner, repo_name, base_branch
            )

            if protection_rules:
                required_reviews = protection_rules.get(
                    "required_pull_request_reviews", {}
                )
                if required_reviews:
                    require_code_owner = required_reviews.get(
                        "require_code_owner_reviews", False
                    )

                    # If code owner reviews are required, our automated approval might not be sufficient
                    if require_code_owner:
                        # Check if user wants to bypass code owner checks
                        if self.force_level in [
                            "code-owners",
                            "protection-rules",
                            "all",
                        ]:
                            # Only log during preview evaluation to avoid duplicate messages
                            if self.preview_mode:
                                self.log.warning(
                                    f"⚠️ Bypassing code owner review requirement for {repo_owner}/{repo_name}#{pr_info.number} (--force={self.force_level})"
                                )
                            return (
                                True,
                                "code owner review requirement bypassed by force level",
                            )
                        else:
                            return (
                                False,
                                "code owner reviews are required - cannot auto-approve",
                            )

        except Exception as exc:
            # Don't fail the merge attempt if we can't check protection rules.
            self.log.debug(
                "Branch protection check failed for %s/%s#%s: %s",
                repo_owner,
                repo_name,
                pr_info.number,
                exc,
            )

        # Predictive merge probe. This is a *best-effort* dry-run verdict
        # only: GitHub's mergeable_state can lag, and repository rulesets
        # are invisible to it, so it must never gate the real merge. Run it
        # only in preview mode to render the evaluation; the execution path
        # is attempt-first and lets the actual merge response be
        # authoritative (Step 6 + _merge_pr_with_retry).
        if self.preview_mode:
            try:
                # Use pre-determined merge method for this repository
                cache_key = f"{repo_owner}/{repo_name}"
                merge_method = self._pr_merge_methods.get(
                    cache_key, self.default_merge_method
                )

                # Predict the outcome to detect hidden branch protection rules
                test_result = await self._predict_merge_outcome(
                    repo_owner, repo_name, pr_info.number, merge_method
                )
                if not test_result[0]:
                    # Check if we should bypass protection rules
                    if self.force_level in [
                        "code-owners",
                        "protection-rules",
                        "all",
                    ]:
                        if self._github_client:
                            self.log.debug(
                                f"Checking bypass permissions for {repo_owner}/{repo_name} with force_level={self.force_level}"
                            )
                            (
                                can_bypass,
                                bypass_reason,
                            ) = await self._github_client.check_user_can_bypass_protection(
                                repo_owner, repo_name, self.force_level
                            )
                            self.log.debug(
                                f"Bypass check result: can_bypass={can_bypass}, reason={bypass_reason}"
                            )
                            if not can_bypass:
                                self.log.warning(
                                    f"Cannot bypass branch protection for {repo_owner}/{repo_name}#{pr_info.number}: {bypass_reason}"
                                )
                                return (
                                    False,
                                    f"cannot bypass branch protection: {bypass_reason}",
                                )

                        self.log.warning(
                            f"⚠️ Bypassing branch protection check for {repo_owner}/{repo_name}#{pr_info.number}: {test_result[1]} (--force={self.force_level})"
                        )
                        # When bypassing, return early to allow merge
                        return (
                            True,
                            f"branch protection check bypassed (--force={self.force_level})",
                        )
                    else:
                        return False, test_result[1]

            except Exception as e:
                # If we can't predict the outcome, continue with other checks
                self.log.debug(
                    f"Could not predict merge outcome for {repo_owner}/{repo_name}#{pr_info.number}: {e}"
                )

        # Additional checks based on PR state
        if pr_info.mergeable_state == "blocked":
            # Check if Copilot comments might be the blocker
            if self.dismiss_copilot and self._copilot_handler:
                has_copilot_comments = (
                    self._copilot_handler.has_blocking_copilot_comments(pr_info)
                )
                if has_copilot_comments:
                    return (
                        True,
                        "PR blocked but has Copilot comments that will be dismissed",
                    )

            # For blocked PRs, if mergeable is True, it just needs approval - we can handle that
            if pr_info.mergeable is True:
                return True, "PR ready for approval and merge"
            else:
                # If mergeable is False and state is blocked, it's blocked by failing checks
                if self.force_level == "all":
                    # Only log during preview evaluation to avoid duplicate messages
                    if self.preview_mode:
                        self.log.warning(
                            f"⚠️ Bypassing failing status checks for {repo_owner}/{repo_name}#{pr_info.number} (--force=all)"
                        )
                    return True, "PR blocked but forcing merge attempt (--force=all)"
                else:
                    # Don't hard-fail here: let Step 5.5 enable
                    # auto-merge and route to AUTO_MERGE_PENDING.
                    # The block reason might be "failing required
                    # check" right now but the check could still
                    # complete successfully — GitHub returns
                    # ``mergeable=False`` transiently for several
                    # reasons (still computing, non-required check
                    # failed). Step 5.5's analyze_block_reason
                    # pre-check still weeds out genuinely-stuck
                    # cases (missing approvals, etc.).
                    return (
                        True,
                        "PR blocked — Step 5.5 will enable auto-merge",
                    )
        elif pr_info.mergeable_state == "behind":
            if not self.fix_out_of_date:
                if self.force_level == "all":
                    self.log.warning(
                        f"⚠️ Attempting merge despite being behind for {repo_owner}/{repo_name}#{pr_info.number} (--force=all)"
                    )
                    return True, "PR behind but forcing merge attempt (--force=all)"
                else:
                    # Don't hard-fail when behind + --no-fix: the
                    # user opted out of *us* rebasing the branch,
                    # but enabling auto-merge in Step 5.5 is a
                    # separate, non-rewriting operation. If a third
                    # party (Dependabot, pre-commit-ci) rebases the
                    # PR while we wait, auto-merge will fire.
                    return (
                        True,
                        "PR behind — Step 5.5 will enable auto-merge",
                    )
            else:
                return True, "PR is behind - will rebase before merge"
        elif pr_info.mergeable_state == "unstable":
            # ``unstable`` means a non-required check failed but
            # the PR is otherwise mergeable. Auto-merge can still
            # fire because non-required checks don't block branch
            # protection. Let Step 5.5 handle it.
            return (
                True,
                "PR unstable — Step 5.5 will enable auto-merge",
            )
        elif pr_info.mergeable_state == "dirty":
            if self.force_level == "all":
                self.log.warning(
                    f"⚠️ Attempting merge despite conflicts for {repo_owner}/{repo_name}#{pr_info.number} (--force=all)"
                )
                return True, "PR has conflicts but forcing merge attempt (--force=all)"
            else:
                return (False, "merge conflicts")

        return True, "All merge requirements appear to be met"

    async def _trigger_stale_precommit_ci(self, pr_info: PullRequestInfo) -> bool:
        """Detect and retrigger a stuck pre-commit.ci run by posting a comment.

        pre-commit.ci uses the commit status API and sometimes gets
        stuck — either never reporting a status at all, or leaving the
        ``pre-commit.ci - pr`` context in ``pending`` indefinitely.
        Either way the PR stays blocked when that context is a required
        status check.  Posting ``pre-commit.ci run`` triggers a fresh
        run.

        A run is treated as stuck when the status is missing entirely,
        or when it has been ``pending`` for longer than
        :data:`PRECOMMIT_CI_STUCK_PENDING_SECONDS` (a slow-but-normal
        run within that window is left alone).

        Args:
            pr_info: Pull request information

        Returns:
            True if a retrigger comment was posted and the status check
            subsequently completed, False otherwise.
        """
        if not self._github_client:
            return False

        repo_owner, repo_name = pr_info.repository_full_name.split("/", 1)
        precommit_context = "pre-commit.ci - pr"

        # 1. Check whether pre-commit.ci is a required status check
        try:
            required_checks = await self._github_client.get_required_status_checks(
                repo_owner, repo_name, pr_info.base_branch or "main"
            )
            required_contexts = [
                c.get("context", "") for c in required_checks if isinstance(c, dict)
            ]
            if precommit_context not in required_contexts:
                return False
        except Exception:
            return False

        # 2. Inspect the existing pre-commit.ci status.  Retrigger when
        #    it is missing entirely or has been ``pending`` past the
        #    stuck threshold; leave any other state (success / failure
        #    / error, or a pending run still within its normal window).
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        try:
            status_data = await self._github_client.get(
                f"/repos/{repo_owner}/{repo_name}/commits/{pr_info.head_sha}/status"
            )
        except Exception as e:
            self.log.debug(
                "Failed to fetch commit status for pre-commit.ci check on %s#%s "
                "(sha=%s); skipping retrigger: %s",
                pr_info.repository_full_name,
                pr_info.number,
                pr_info.head_sha,
                e,
            )
            return False

        precommit_status: dict[str, Any] | None = None
        if isinstance(status_data, dict):
            for s in status_data.get("statuses", []):
                if isinstance(s, dict) and s.get("context") == precommit_context:
                    precommit_status = s
                    break

        if precommit_status is not None:
            state = precommit_status.get("state")
            if state != "pending":
                # A reported, non-pending result (success / failure /
                # error) is not stale — nothing to retrigger.
                return False
            # Pending: only stuck once it has been pending longer than
            # the threshold.  Use ``updated_at`` (when pre-commit.ci
            # set the pending status), falling back to ``created_at``.
            raw_ts = precommit_status.get("updated_at") or precommit_status.get(
                "created_at"
            )
            pending_age: float | None = None
            if isinstance(raw_ts, str) and raw_ts:
                try:
                    ts = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
                    pending_age = (now - ts).total_seconds()
                except (ValueError, TypeError):
                    # ``ValueError``: unparsable timestamp.
                    # ``TypeError``: a timestamp lacking tz info parses
                    # to a naive datetime, which cannot be subtracted
                    # from the tz-aware ``now``.  Either way, degrade to
                    # ``None`` (fail closed) rather than abort the run.
                    pending_age = None
            if pending_age is None or pending_age < PRECOMMIT_CI_STUCK_PENDING_SECONDS:
                # Still within the normal window (or no timestamp to
                # judge by) — leave the run to finish.
                return False
            self.log.info(
                "pre-commit.ci on %s#%s pending for %.0fs; treating as stuck.",
                pr_info.repository_full_name,
                pr_info.number,
                pending_age,
            )

        # 3. The run is stale (missing, or stuck pending) — check for
        # an existing trigger comment before posting a duplicate
        # (avoids spam if dependamerge runs repeatedly while the
        # status is still not progressing).
        try:
            comments = await self._github_client.get(
                f"/repos/{repo_owner}/{repo_name}/issues/{pr_info.number}/comments?per_page=100"
            )
            if isinstance(comments, list):
                for c in comments:
                    if not isinstance(c, dict):
                        continue
                    body = c.get("body")
                    if isinstance(body, str) and body.strip() == "pre-commit.ci run":
                        self.log.info(
                            "Found existing pre-commit.ci trigger comment on "
                            f"{pr_info.repository_full_name}#{pr_info.number}; "
                            "skipping duplicate comment."
                        )
                        return False
        except Exception:
            # If we fail to list comments, continue and attempt to post the
            # trigger anyway.
            pass

        self._pr_status(
            f"🔄 Re-triggering pre-commit.ci: {pr_info.html_url}",
            level="info",
        )

        try:
            await self._github_client.post_issue_comment(
                repo_owner, repo_name, pr_info.number, "pre-commit.ci run"
            )
        except Exception as e:
            self.log.warning(
                f"Failed to post pre-commit.ci trigger comment on "
                f"{pr_info.repository_full_name}#{pr_info.number}: {e}"
            )
            return False

        # 4. Poll for the status to appear (up to ~5 minutes)
        # pre-commit.ci can take up to five minutes to run and report back,
        # so we need a generous timeout to avoid prematurely marking PRs as
        # unmergeable when the check simply hasn't finished yet.  The
        # whole poll is a wait on an external service, so the worker's
        # concurrency slot is released for its duration (``parked()``).
        max_polls = self._merge_poll_max_attempts
        async with parked():
            for attempt in range(max_polls):
                await asyncio.sleep(self._merge_recheck_interval)
                try:
                    status_data = await self._github_client.get(
                        f"/repos/{repo_owner}/{repo_name}/commits/{pr_info.head_sha}/status"
                    )
                    if isinstance(status_data, dict):
                        for s in status_data.get("statuses", []):
                            if not isinstance(s, dict):
                                continue
                            if s.get("context") != precommit_context:
                                continue
                            state = s.get("state")
                            if state == "success":
                                self._pr_status(
                                    f"✅ pre-commit.ci passed: {pr_info.html_url}",
                                    level="info",
                                )
                                return True
                            elif state in ("failure", "error"):
                                self._pr_status(
                                    f"❌ pre-commit.ci failed: {pr_info.html_url}",
                                    level="warning",
                                )
                                return False
                            # state == "pending" — keep polling
                except Exception as e:
                    self.log.debug(
                        "Failed to poll pre-commit.ci status for %s: %s",
                        f"{pr_info.repository_full_name}#{pr_info.number}",
                        e,
                    )

                if attempt == max_polls - 1:
                    self.log.debug(
                        f"Still waiting for pre-commit.ci on "
                        f"{pr_info.repository_full_name}#{pr_info.number} "
                        f"({(attempt + 1) * self._merge_recheck_interval:.0f}s elapsed)"
                    )

        self.log.warning(
            f"Timed out waiting for pre-commit.ci on "
            f"{pr_info.repository_full_name}#{pr_info.number}"
        )
        return False

    async def _detect_stuck_required_check(
        self,
        pr_info: PullRequestInfo,
    ) -> tuple[bool, str | None, float]:
        """Detect whether a *required* verification check is stuck.

        Required checks (DCO, lint, build, license scans, etc.)
        normally start reporting within a handful of seconds.  When
        one has been queued / in-progress / pending for longer than
        :data:`STUCK_CHECK_THRESHOLD_SECONDS` on a PR that itself was
        created and last updated more than that long ago, treat it
        as stuck so the caller can decide whether to ask dependabot
        to recreate the PR (the only reliable recovery for a
        dependabot PR with no ``recreate``/``rebase`` macro of its
        own once a required check has stalled indefinitely).

        Only checks that are *required* on the PR's base branch are
        considered, since a non-required check cannot block the
        merge.  DCO-shaped checks are additionally always treated as
        eligible as a safety net: the GitHub DCO App check is the
        canonical stuck-check case and is effectively always blocking
        where it is enabled, even when the required-checks lookup
        cannot enumerate it.

        ``pre-commit.ci`` checks are explicitly excluded here even
        when required — they have their own dedicated recovery via
        :meth:`_trigger_stale_precommit_ci`, which posts the
        ``pre-commit.ci run`` comment (dependabot's ``recreate`` macro
        does not retrigger pre-commit.ci).

        The age floor on PR ``created_at`` / ``updated_at`` avoids
        false positives on PRs that were touched seconds before we
        observed them — in those cases the check is simply running
        normally and should be allowed to finish.

        Args:
            pr_info: The pull request being evaluated.

        Returns:
            A 3-tuple ``(is_stuck, check_name, age_seconds)``.
            ``check_name`` is the GitHub check / status name of the
            stuck check (or ``None`` when no stuck check was found).
            ``age_seconds`` is the time the check has been pending
            (or ``0.0`` when no candidate check was found).
        """
        if not self._github_client:
            return False, None, 0.0

        repo_owner, repo_name = pr_info.repository_full_name.split("/", 1)
        threshold = STUCK_CHECK_THRESHOLD_SECONDS

        from datetime import datetime, timezone

        def _parse_ts(value: Any) -> datetime | None:
            if not isinstance(value, str) or not value:
                return None
            try:
                # GitHub returns RFC 3339 with a trailing ``Z``.
                ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
            # A timestamp without tz info parses to a naive datetime,
            # which raises ``TypeError`` when subtracted from the
            # tz-aware ``now`` below.  Treat it as unparsable (fail
            # closed) so the detector degrades gracefully instead of
            # aborting the merge run.
            if ts.tzinfo is None:
                return None
            return ts

        def _is_dco_name(name: str) -> bool:
            """Return True when ``name`` looks like a DCO check.

            Matches the common variants emitted by the GitHub DCO
            App and similar bots: ``DCO``, ``dco/dco``, ``dcobot``,
            and any name containing ``signoff`` / ``sign-off`` /
            ``signed-off`` (case-insensitive).
            """
            n = (name or "").strip().lower()
            if not n:
                return False
            if n in {"dco", "dco/dco", "dcobot"} or n.startswith("dco/"):
                return True
            return "signoff" in n or "sign-off" in n or "signed-off" in n

        def _is_precommit_name(name: str) -> bool:
            """Return True when ``name`` is a pre-commit.ci check.

            pre-commit.ci reports as ``pre-commit.ci - pr`` (and the
            ``- ci`` variant).  It is excluded from this detector
            because dependabot's ``recreate`` macro does not
            retrigger it; ``_trigger_stale_precommit_ci`` handles it
            via the ``pre-commit.ci run`` comment instead.
            """
            n = (name or "").strip().lower()
            return "pre-commit.ci" in n or "pre-commit-ci" in n

        # 1. PR-level age floor — don't fire on PRs we caught right
        #    after they were opened or force-pushed; checks on those
        #    are simply running normally.
        now = datetime.now(timezone.utc)
        try:
            pr_data = await self._github_client.get(
                f"/repos/{repo_owner}/{repo_name}/pulls/{pr_info.number}"
            )
        except Exception as exc:
            self.log.debug(
                "_detect_stuck_required_check: pr fetch failed for %s#%s: %s",
                pr_info.repository_full_name,
                pr_info.number,
                exc,
            )
            return False, None, 0.0

        if not isinstance(pr_data, dict):
            return False, None, 0.0

        pr_created = _parse_ts(pr_data.get("created_at"))
        pr_updated = _parse_ts(pr_data.get("updated_at"))
        if pr_created is None or pr_updated is None:
            # Without timing data we cannot safely judge stuckness;
            # fail closed.
            return False, None, 0.0

        pr_age = (now - pr_created).total_seconds()
        pr_idle = (now - pr_updated).total_seconds()
        if pr_age < threshold or pr_idle < threshold:
            return False, None, 0.0

        # 2. Determine which checks are *required* on the base branch
        #    so a non-blocking check is never treated as stuck.  On
        #    any failure we fall back to an empty set, leaving the
        #    DCO safety net (below) as the only eligible matcher.
        required_contexts: set[str] = set()
        try:
            required = await self._github_client.get_required_status_checks(
                repo_owner, repo_name, pr_info.base_branch or "main"
            )
            if isinstance(required, list):
                required_contexts = {
                    str(c.get("context", "")).strip().lower()
                    for c in required
                    if isinstance(c, dict) and c.get("context")
                }
        except Exception as exc:
            self.log.debug(
                "_detect_stuck_required_check: required-checks fetch failed "
                "for %s#%s: %s",
                pr_info.repository_full_name,
                pr_info.number,
                exc,
            )
            required_contexts = set()

        def _is_eligible(name: str) -> bool:
            """Return True when a stuck ``name`` should drive recreate.

            Eligible when the check is required on the base branch or
            is a DCO-shaped check (safety net), and is *not* a
            pre-commit.ci check (handled separately).
            """
            if _is_precommit_name(name):
                return False
            return (name or "").strip().lower() in required_contexts or _is_dco_name(
                name
            )

        # 3. Examine check-runs and status contexts on the head SHA.
        candidate_name: str | None = None
        candidate_age: float = 0.0

        try:
            runs = await self._github_client.get(
                f"/repos/{repo_owner}/{repo_name}/commits/{pr_info.head_sha}/check-runs"
            )
        except Exception as exc:
            self.log.debug(
                "_detect_stuck_required_check: check-runs fetch failed for "
                "%s#%s: %s",
                pr_info.repository_full_name,
                pr_info.number,
                exc,
            )
            runs = None

        if isinstance(runs, dict):
            for run in runs.get("check_runs") or []:
                if not isinstance(run, dict):
                    continue
                name = run.get("name", "")
                if not _is_eligible(name):
                    continue
                status = run.get("status")
                if status not in ("queued", "in_progress"):
                    continue
                started = _parse_ts(run.get("started_at"))
                # Use the *latest* of started_at and PR updated_at
                # as the reference so a stale started_at left over
                # from a prior head SHA does not inflate the age.
                ref = max(started, pr_updated) if started else pr_updated
                age = (now - ref).total_seconds()
                if age >= threshold and age > candidate_age:
                    candidate_name = name
                    candidate_age = age

        try:
            statuses = await self._github_client.get(
                f"/repos/{repo_owner}/{repo_name}/commits/{pr_info.head_sha}/status"
            )
        except Exception as exc:
            self.log.debug(
                "_detect_stuck_required_check: status fetch failed for " "%s#%s: %s",
                pr_info.repository_full_name,
                pr_info.number,
                exc,
            )
            statuses = None

        if isinstance(statuses, dict):
            for s in statuses.get("statuses") or []:
                if not isinstance(s, dict):
                    continue
                ctx = s.get("context", "")
                if not _is_eligible(ctx):
                    continue
                if s.get("state") != "pending":
                    continue
                updated = _parse_ts(s.get("updated_at")) or pr_updated
                ref = max(updated, pr_updated)
                age = (now - ref).total_seconds()
                if age >= threshold and age > candidate_age:
                    candidate_name = ctx
                    candidate_age = age

        if candidate_name is None:
            return False, None, 0.0
        return True, candidate_name, candidate_age

    async def _trigger_dependabot_recreate(
        self, pr_info: PullRequestInfo
    ) -> PullRequestInfo | None:
        """
        Detect an unsigned dependabot commit and ask dependabot to recreate
        the pull request so that the new commit is properly signed.

        When a repository's branch protection requires commit signatures,
        dependabot PRs can end up with unverified commits (e.g. after a
        rebase or force-push by GitHub).  Posting ``@dependabot recreate``
        causes dependabot to close the current PR and open a fresh one
        whose commit is signed by GitHub.

        Args:
            pr_info: Pull request information for the failing PR.

        Returns:
            A new ``PullRequestInfo`` for the recreated PR if the recreate
            was triggered, the old PR was closed, and a replacement was
            found.  Returns ``None`` if the recreate was not applicable or
            did not succeed within the polling window.
        """
        if not self._github_client:
            return None

        repo_owner, repo_name = pr_info.repository_full_name.split("/", 1)

        # 1. Only applies to dependabot PRs
        if not is_dependabot(pr_info.author):
            return None

        # 2. Check whether the branch requires signed commits
        try:
            requires_signatures = await self._github_client.requires_commit_signatures(
                repo_owner, repo_name, pr_info.base_branch or "main"
            )
            if not requires_signatures:
                self.log.debug(
                    "Branch %s/%s:%s does not require commit signatures; "
                    "skipping dependabot recreate.",
                    repo_owner,
                    repo_name,
                    pr_info.base_branch or "main",
                )
                return None
        except Exception as e:
            self.log.debug(
                "Could not determine signature requirement for %s: %s",
                pr_info.repository_full_name,
                e,
            )
            return None

        # 3. Check whether any commits are unverified
        try:
            (
                all_verified,
                unverified_shas,
            ) = await self._github_client.check_pr_commit_signatures(
                repo_owner, repo_name, pr_info.number
            )
            if all_verified:
                self.log.debug(
                    "All commits on %s#%s are verified; recreate not needed.",
                    pr_info.repository_full_name,
                    pr_info.number,
                )
                return None
        except Exception as e:
            self.log.debug(
                "Could not check commit signatures for %s#%s: %s",
                pr_info.repository_full_name,
                pr_info.number,
                e,
            )
            return None

        # 4. Guard against duplicate recreate comments
        try:
            comments = await self._github_client.get(
                f"/repos/{repo_owner}/{repo_name}/issues/{pr_info.number}/comments"
                f"?per_page=100&direction=desc"
            )
            if isinstance(comments, list):
                for c in comments:
                    if not isinstance(c, dict):
                        continue
                    body = c.get("body")
                    if isinstance(body, str) and "@dependabot recreate" in body:
                        self.log.info(
                            "Found existing @dependabot recreate comment on "
                            "%s#%s; skipping duplicate.",
                            pr_info.repository_full_name,
                            pr_info.number,
                        )
                        return None
        except Exception as e:
            self.log.warning(
                "Could not list comments for %s#%s to check for existing "
                "@dependabot recreate comment: %s",
                pr_info.repository_full_name,
                pr_info.number,
                e,
            )
            return None

        # 5. Post the recreate comment
        self._pr_status(
            f"🔄 Requesting dependabot recreate: {pr_info.html_url} "
            f"[unverified commits: {', '.join(unverified_shas)}]",
            level="info",
        )

        try:
            await self._github_client.post_issue_comment(
                repo_owner, repo_name, pr_info.number, "@dependabot recreate"
            )
        except Exception as e:
            self.log.warning(
                "Failed to post @dependabot recreate comment on %s#%s: %s",
                pr_info.repository_full_name,
                pr_info.number,
                e,
            )
            return None

        # 6. Poll for the old PR to close and a replacement to appear.
        #    Dependabot typically responds within 30-90 seconds.
        #    We poll using the centralised merge timeout.  The whole
        #    poll (including the nested recreated-PR checks wait) is a
        #    wait on dependabot + CI, so the worker's concurrency slot
        #    is released for its duration (``parked()``).
        max_polls = self._merge_poll_max_attempts
        old_pr_closed = False

        async with parked():
            for attempt in range(max_polls):
                await asyncio.sleep(self._merge_recheck_interval)

                # 6a. Check if the old PR has been closed
                if not old_pr_closed:
                    try:
                        old_pr_data = await self._github_client.get(
                            f"/repos/{repo_owner}/{repo_name}/pulls/{pr_info.number}"
                        )
                        if isinstance(old_pr_data, dict):
                            if old_pr_data.get("state") == "closed":
                                old_pr_closed = True
                                self._pr_status(
                                    f"✅ Old PR closed by dependabot: "
                                    f"{pr_info.html_url} "
                                    f"({(attempt + 1) * self._merge_recheck_interval:.0f}s elapsed)",
                                    level="info",
                                )
                    except Exception as e:
                        self.log.debug(
                            "Error polling old PR state for %s#%s: %s",
                            pr_info.repository_full_name,
                            pr_info.number,
                            e,
                        )

                # 6b. Once the old PR is closed, look for the replacement
                if old_pr_closed:
                    try:
                        # Search for open PRs from dependabot on the same head branch
                        prs = await self._github_client.get(
                            f"/repos/{repo_owner}/{repo_name}/pulls"
                            f"?state=open&head={repo_owner}:{pr_info.head_branch}&per_page=5"
                        )
                        if isinstance(prs, list):
                            for pr_data in prs:
                                if not isinstance(pr_data, dict):
                                    continue
                                pr_author = pr_data.get("user", {}).get("login", "")
                                if not is_dependabot(pr_author):
                                    continue

                                new_number = pr_data.get("number")
                                if new_number is None or new_number == pr_info.number:
                                    continue

                                # Verify the replacement targets the same base branch
                                new_base = pr_data.get("base", {}).get("ref", "")
                                if new_base != (pr_info.base_branch or "main"):
                                    self.log.debug(
                                        "Skipping candidate PR #%s: targets %s, "
                                        "expected %s",
                                        new_number,
                                        new_base,
                                        pr_info.base_branch or "main",
                                    )
                                    continue

                                # Found a replacement — now wait for checks to pass
                                new_pr_info = await self._wait_for_recreated_pr_checks(
                                    repo_owner, repo_name, new_number, pr_data
                                )
                                # Always return after the first wait attempt to avoid
                                # performing multiple long waits for the same PR.
                                return new_pr_info
                    except Exception as e:
                        self.log.debug(
                            "Error searching for replacement PR for %s#%s: %s",
                            pr_info.repository_full_name,
                            pr_info.number,
                            e,
                        )

                if attempt % 3 == 2:
                    self.log.debug(
                        "Still waiting for dependabot recreate on %s#%s (%.0fs elapsed, old_pr_closed=%s)",
                        pr_info.repository_full_name,
                        pr_info.number,
                        (attempt + 1) * self._merge_recheck_interval,
                        old_pr_closed,
                    )

        self.log.warning(
            "Timed out waiting for dependabot to recreate %s#%s",
            pr_info.repository_full_name,
            pr_info.number,
        )
        return None

    async def _wait_for_recreated_pr_checks(
        self,
        repo_owner: str,
        repo_name: str,
        new_number: int,
        pr_data: dict[str, Any],
    ) -> PullRequestInfo | None:
        """
        Wait for the recreated PR's status checks to complete.

        Polls the new PR using the shared merge timeout settings. The
        total wait here is controlled by ``self._merge_poll_max_attempts
        * self._merge_recheck_interval`` (default: ~5 minutes), so
        ``--merge-timeout`` also affects this loop.

        Args:
            repo_owner: Repository owner.
            repo_name: Repository name.
            new_number: The PR number of the recreated pull request.
            pr_data: The initial PR data dict from the GitHub API.

        Returns:
            A ``PullRequestInfo`` if the PR became mergeable, None on timeout.
        """
        if not self._github_client:
            return None

        full_name = f"{repo_owner}/{repo_name}"
        html_url = pr_data.get(
            "html_url", f"https://github.com/{full_name}/pull/{new_number}"
        )

        self._pr_status(
            f"🔍 Found recreated PR, waiting for checks: {html_url}",
            level="info",
        )

        # Enable auto-merge on the recreated PR so it merges
        # even if we time out waiting for status checks.
        if pr_data.get("node_id"):
            # We don't have a full PullRequestInfo yet, but we can
            # construct a minimal one for the auto-merge helper.
            _tmp_pr = PullRequestInfo(
                number=new_number,
                node_id=pr_data.get("node_id"),
                title=pr_data.get("title", ""),
                body=pr_data.get("body"),
                author=((pr_data.get("user") or {}).get("login", "")),
                head_sha=((pr_data.get("head") or {}).get("sha", "")),
                base_branch=((pr_data.get("base") or {}).get("ref", "")),
                head_branch=((pr_data.get("head") or {}).get("ref", "")),
                state="open",
                mergeable=None,
                mergeable_state=None,
                behind_by=None,
                files_changed=[],
                repository_full_name=full_name,
                html_url=html_url,
            )
            await self._enable_auto_merge_for_pr(_tmp_pr, repo_owner, repo_name)

        # Poll for the new PR to become mergeable
        max_check_polls = self._merge_poll_max_attempts
        for check_attempt in range(max_check_polls):
            await asyncio.sleep(self._merge_recheck_interval)
            try:
                refreshed = await self._github_client.get(
                    f"/repos/{repo_owner}/{repo_name}/pulls/{new_number}"
                )
                if not isinstance(refreshed, dict):
                    continue

                mergeable = refreshed.get("mergeable")
                mergeable_state = refreshed.get("mergeable_state")

                if mergeable_state == "clean" or (
                    mergeable is True and mergeable_state in ("clean", "unstable")
                ):
                    self._pr_status(
                        f"✅ Recreated PR is ready to merge: {html_url}",
                        level="info",
                    )
                    from .models import FileChange

                    files_changed: list[FileChange] = []
                    try:
                        async for files_data in self._github_client.get_paginated(
                            f"/repos/{repo_owner}/{repo_name}/pulls/{new_number}/files",
                            per_page=100,
                        ):
                            if isinstance(files_data, list):
                                for f in files_data:
                                    if isinstance(f, dict):
                                        files_changed.append(
                                            FileChange(
                                                filename=f.get("filename", ""),
                                                additions=int(f.get("additions", 0)),
                                                deletions=int(f.get("deletions", 0)),
                                                changes=int(f.get("changes", 0)),
                                                status=f.get("status", "modified"),
                                            )
                                        )
                    except Exception as e:
                        self.log.debug(
                            "Failed to fetch files for recreated PR %s#%s: %s",
                            f"{repo_owner}/{repo_name}",
                            new_number,
                            e,
                        )

                    return PullRequestInfo(
                        number=new_number,
                        node_id=refreshed.get("node_id"),
                        title=refreshed.get("title", ""),
                        body=refreshed.get("body"),
                        author=((refreshed.get("user") or {}).get("login", "")),
                        head_sha=((refreshed.get("head") or {}).get("sha", "")),
                        base_branch=((refreshed.get("base") or {}).get("ref", "")),
                        head_branch=((refreshed.get("head") or {}).get("ref", "")),
                        state=refreshed.get("state", "open"),
                        mergeable=mergeable,
                        mergeable_state=mergeable_state,
                        behind_by=None,
                        files_changed=files_changed,
                        repository_full_name=full_name,
                        html_url=html_url,
                    )

                if mergeable_state == "dirty":
                    self.log.warning(
                        "Recreated PR %s#%s has merge conflicts; aborting wait.",
                        full_name,
                        new_number,
                    )
                    return None

                # blocked / behind / unknown — keep polling
                if check_attempt % 3 == 2:
                    self.log.debug(
                        "Waiting for checks on recreated PR %s#%s "
                        "(state=%s, %.0fs elapsed)",
                        full_name,
                        new_number,
                        mergeable_state,
                        (check_attempt + 1) * self._merge_recheck_interval,
                    )

            except Exception as e:
                self.log.debug(
                    "Error polling recreated PR %s#%s: %s",
                    full_name,
                    new_number,
                    e,
                )

        self.log.warning(
            "Timed out waiting for checks on recreated PR %s#%s",
            full_name,
            new_number,
        )
        return None

    async def _approve_pr(self, owner: str, repo: str, pr_number: int) -> bool:
        """
        Approve a pull request if not already approved by the current user or sufficiently approved.

        Args:
            owner: Repository owner
            repo: Repository name
            pr_number: Pull request number

        Returns:
            True if approval was added, False if already approved/sufficient

        Raises:
            Exception: If approval fails
        """
        if not self._github_client:
            raise RuntimeError("GitHub client not initialized")

        try:
            # Check if current user has already approved this PR
            pr_data = await self._github_client.get(
                f"/repos/{owner}/{repo}/pulls/{pr_number}"
            )

            if isinstance(pr_data, dict):
                # Get current user login (cached on the client after the
                # first call — the login is session-constant, so this
                # costs one round-trip per run instead of one per PR).
                current_user = await self._github_client.get_authenticated_user_login()

                if current_user:
                    reviews_data = await self._github_client.get(
                        f"/repos/{owner}/{repo}/pulls/{pr_number}/reviews"
                    )

                    if isinstance(reviews_data, list):
                        # Look for existing approval by current user
                        for review in reviews_data:
                            if (
                                review.get("user", {}).get("login") == current_user
                                and review.get("state") == "APPROVED"
                            ):
                                self.log.debug(
                                    f"⏩ Already approved: {owner}/{repo}#{pr_number} [{current_user}]"
                                )
                                return False

                        # Check if PR already has sufficient approvals from others
                        approved_reviews = [
                            review
                            for review in reviews_data
                            if review.get("state") == "APPROVED"
                            and review.get("user", {}).get("login") != current_user
                        ]

                        if (
                            approved_reviews
                            and pr_data.get("mergeable_state") == "clean"
                        ):
                            approvers = [
                                review.get("user", {}).get("login", "unknown")
                                for review in approved_reviews
                            ]
                            approvers_str = ", ".join(approvers)
                            self.log.debug(
                                f"⏩ Already approved: {owner}/{repo}#{pr_number} [{approvers_str}]"
                            )
                            return False

            await self._github_client.approve_pull_request(
                owner,
                repo,
                pr_number,
                "🤖 Dependamerge\nApproved this pull request ✅",
            )
            return True
        except GitHubPermissionError:
            # Let typed permission errors propagate to the caller's
            # dedicated handler in ``_merge_single_pr``.  Wrapping
            # them in a generic ``RuntimeError`` (as the old broad
            # ``except Exception`` below did) hid them from that
            # handler and routed the failure through the catch-all
            # path, which dumps a full stack trace to stderr on
            # every PR in the batch.
            raise
        except Exception as e:
            error_str = str(e)

            # Check for 403 Forbidden - missing pull request review permissions
            if "403" in error_str and "Forbidden" in error_str:
                raise RuntimeError(
                    f"Failed to approve PR {owner}/{repo}#{pr_number}: Missing 'Pull requests: Read and write' permission. "
                    f"For fine-grained tokens, enable 'Pull requests: Read and write' access. "
                    f"For classic tokens, ensure 'repo' scope is enabled."
                ) from e
            elif "422" in error_str and "Unprocessable Entity" in error_str:
                # This usually means the PR can't be approved (e.g., already approved by user, or other restrictions)
                self.log.debug(
                    f"⏩ Already approved: {owner}/{repo}#{pr_number} [cannot approve - already approved or restricted]"
                )
                return False
            else:
                raise RuntimeError(
                    f"Failed to approve PR {owner}/{repo}#{pr_number}: {e}"
                ) from e

    async def _merge_pr_with_retry(
        self, pr_info: PullRequestInfo, owner: str, repo: str
    ) -> bool:
        """
        Merge a PR with retry logic for transient failures.

        Args:
            pr_info: Pull request information
            owner: Repository owner
            repo: Repository name

        Returns:
            True if merged successfully, False otherwise
        """
        if not self._github_client:
            raise RuntimeError("GitHub client not initialized")

        for attempt in range(self.max_retries + 1):
            try:
                # Check if PR has already been closed/merged before attempting
                if attempt > 0:
                    # Re-fetch PR state to check if it was merged by a previous attempt
                    # or by external processes
                    try:
                        current_pr_data = await self._github_client.get(
                            f"/repos/{owner}/{repo}/pulls/{pr_info.number}"
                        )
                        if isinstance(current_pr_data, dict):
                            current_state = current_pr_data.get("state")
                            current_merged = current_pr_data.get("merged", False)

                            if current_state == "closed" and current_merged:
                                self.log.info(
                                    f"✅ PR {owner}/{repo}#{pr_info.number} was already merged, skipping retry"
                                )
                                return True
                            elif current_state == "closed" and not current_merged:
                                self.log.info(
                                    f"⚠️ PR {owner}/{repo}#{pr_info.number} was closed without merging, aborting retry"
                                )
                                # This will be caught by the outer merge logic and formatted consistently
                                return False
                    except Exception as state_check_error:
                        self.log.debug(
                            f"Failed to check PR state before retry {attempt + 1}: {state_check_error}"
                        )

                # Use pre-determined merge method for this repository
                cache_key = f"{owner}/{repo}"
                merge_method = self._pr_merge_methods.get(
                    cache_key, self.default_merge_method
                )

                # Attempt the merge
                self.log.debug(
                    f"Attempting merge for {owner}/{repo}#{pr_info.number} with method={merge_method}"
                )
                merged = await self._github_client.merge_pull_request(
                    owner, repo, pr_info.number, merge_method
                )
                self.log.debug(
                    f"Merge API returned {merged} for {owner}/{repo}#{pr_info.number}"
                )

                if merged:
                    return True

                # Merge failed, check if we can fix it
                self.log.warning(
                    f"⚠️ Merge API returned false for PR {owner}/{repo}#{pr_info.number} (attempt {attempt + 1})"
                )
                if attempt < self.max_retries:
                    should_retry = await self._handle_merge_failure(
                        pr_info, owner, repo
                    )
                    if should_retry:
                        self.log.info(
                            f"Retrying merge for PR {owner}/{repo}#{pr_info.number} (attempt {attempt + 2})"
                        )
                        continue
                    else:
                        self.log.info(
                            f"Not retrying PR {owner}/{repo}#{pr_info.number} - no fixable issues found"
                        )
                        break

            except GitHubPermissionError:
                # Token cannot merge on this repo — propagate to the
                # caller so the PermissionError handler in
                # ``_merge_single_pr`` reports it cleanly and records
                # the repo for fast-fail of remaining PRs in the
                # batch.  Retrying or breaking silently here would
                # downgrade the failure into a generic
                # "merge failed: clean" reason that misleads users.
                raise
            except Exception as e:
                error_msg = str(e)

                # Store exception for better error reporting
                pr_key = f"{owner}/{repo}#{pr_info.number}"
                self._last_merge_exception[pr_key] = e
                self.log.debug(
                    f"Stored exception for {pr_key}: {type(e).__name__}: {str(e)[:200]}"
                )

                # Enhanced error handling with specific status code checks
                if "405" in error_msg and "Method Not Allowed" in error_msg:
                    # Don't log here - will be handled in failure summary
                    if "base branch was modified" in error_msg.lower():
                        # Pure concurrency race: in a same-repo batch a
                        # sibling PR merged and advanced the base branch
                        # between GitHub computing this PR's merge commit
                        # and applying it, so GitHub returns 405 "Base
                        # branch was modified. Review and try the merge
                        # again."  It is always transient and unrelated to
                        # the PR's own mergeability (no rebase or approval
                        # is needed), so a short delay lets GitHub recompute
                        # against the new base head, then we retry.
                        if attempt < self.max_retries:
                            retry_delay = 2.0 * (attempt + 1)
                            self.log.info(
                                f"Base branch moved under {pr_key} (concurrent "
                                f"merge); waiting {retry_delay}s before retry "
                                f"(attempt {attempt + 1}/{self.max_retries + 1})…"
                            )
                            await asyncio.sleep(retry_delay)
                            continue
                        else:
                            break
                    if "behind" in error_msg.lower() and self.fix_out_of_date:
                        # Allow retry for behind PRs
                        pass
                    elif pr_info.mergeable_state in ("clean", "unstable"):
                        # The PR should be mergeable but GitHub returned 405 —
                        # this is a transient API error (often follows a 502
                        # during GitHub degradation).  Re-fetch state and retry.
                        if attempt < self.max_retries:
                            retry_delay = 3.0 * (attempt + 1)
                            self.log.info(
                                f"Transient 405 on mergeable PR {pr_key} "
                                f"(state={pr_info.mergeable_state}), "
                                f"waiting {retry_delay}s before retry "
                                f"(attempt {attempt + 1}/{self.max_retries + 1})…"
                            )
                            await asyncio.sleep(retry_delay)
                            # Refresh PR state in case something changed
                            try:
                                if self._github_client:
                                    refreshed = await self._github_client.get(
                                        f"/repos/{owner}/{repo}/pulls/{pr_info.number}"
                                    )
                                    if isinstance(refreshed, dict):
                                        pr_info.mergeable = refreshed.get("mergeable")
                                        pr_info.mergeable_state = refreshed.get(
                                            "mergeable_state"
                                        )
                                        self.log.debug(
                                            f"Refreshed {pr_key}: mergeable={pr_info.mergeable}, "
                                            f"mergeable_state={pr_info.mergeable_state}"
                                        )
                            except Exception as refresh_err:
                                self.log.debug(
                                    f"Failed to refresh PR state for {pr_key}: {refresh_err}"
                                )
                            continue
                        else:
                            break
                    elif pr_info.mergeable_state == "blocked":
                        # If we just approved this PR, the branch protection
                        # evaluator may not have caught up yet.  Re-fetch the
                        # PR state and, if it has become "clean", allow a retry
                        # instead of giving up immediately.
                        if (
                            pr_key in self._recently_approved
                            and attempt < self.max_retries
                        ):
                            try:
                                if self._github_client:
                                    if self._post_approval_delay <= 0:
                                        retry_delay = 0.0
                                    else:
                                        retry_delay = self._post_approval_delay + 2.0
                                    self.log.info(
                                        f"Post-approval propagation retry for {pr_key}, "
                                        f"waiting {retry_delay}s before re-checking…"
                                    )
                                    if retry_delay > 0:
                                        await asyncio.sleep(retry_delay)
                                    refreshed = await self._github_client.get(
                                        f"/repos/{owner}/{repo}/pulls/{pr_info.number}"
                                    )
                                    if isinstance(refreshed, dict):
                                        new_state = refreshed.get("mergeable_state")
                                        new_mergeable = refreshed.get("mergeable")
                                        self.log.info(
                                            f"Refreshed {pr_key}: mergeable={new_mergeable}, "
                                            f"mergeable_state={new_state}"
                                        )
                                        pr_info.mergeable = new_mergeable
                                        pr_info.mergeable_state = new_state
                                        if new_state == "clean":
                                            # Approval has propagated — retry the merge
                                            continue
                            except Exception as refresh_err:
                                self.log.debug(
                                    f"Failed to refresh PR state for {pr_key}: {refresh_err}"
                                )
                            # Remove from recently-approved so we don't loop forever
                            self._recently_approved.discard(pr_key)
                        # Still blocked after re-check (or not recently approved)
                        break
                    else:
                        # Don't retry 405 errors unless they're "behind" issues
                        break
                elif "403" in error_msg and "Forbidden" in error_msg:
                    break
                elif "422" in error_msg:
                    break
                else:
                    # Only log for debugging purposes
                    self.log.debug(
                        f"Merge attempt {attempt + 1} failed for PR {owner}/{repo}#{pr_info.number}: {e}"
                    )

                if attempt >= self.max_retries:
                    break

                # Don't retry certain error types that are unlikely to be transient
                # Exception: Allow retry for 405 errors on "behind" PRs if fix_out_of_date is enabled
                if ("405" in error_msg and "behind" not in error_msg.lower()) or (
                    "422" in error_msg and "not mergeable" in error_msg.lower()
                ):
                    self.log.info(
                        f"Not retrying PR {owner}/{repo}#{pr_info.number} due to permanent error condition"
                    )
                    break
                elif (
                    "405" in error_msg
                    and "behind" in error_msg.lower()
                    and not self.fix_out_of_date
                ):
                    self.log.info(
                        f"Not retrying PR {owner}/{repo}#{pr_info.number} - behind base branch but --no-fix is set"
                    )
                    break

                # Wait a bit before retrying
                await asyncio.sleep(1.0)

        self.log.debug(
            f"_merge_pr_with_retry returning False for {owner}/{repo}#{pr_info.number} after all retries"
        )
        return False

    async def _wait_for_required_workflows_and_retry(
        self, pr_info: PullRequestInfo, owner: str, repo: str
    ) -> bool:
        """Wait out still-executing required workflows, then retry the merge.

        Recovery for a merge rejected with 405 "Repository rule
        violations found … Required workflows '…' are not satisfied"
        (see :meth:`_merge_error_indicates_pending_workflows`): the
        ruleset-required workflows are still running, so the rejection
        is pending, not terminal.  Alternates between waiting for the
        PR to leave a blocked/unstable state and re-attempting the
        merge, all under one shared deadline of ``merge_timeout`` (the
        same shared-budget pattern conflict recovery uses for its
        rebase/checks phases) so the recovery can never double-spend
        the wait budget.  Each cycle dispatches a *single* merge call
        rather than :meth:`_merge_pr_with_retry`, whose internal
        retry ladder would multiply the number of merge attempts.

        The loop also covers the stale-``clean`` snapshot case (the
        REST ``mergeable_state`` can lag behind or be blind to ruleset
        workflows): when the wait exits immediately because the cached
        state already looks mergeable, the failed re-merge plus the
        interval sleep below degrade gracefully into a bounded
        merge-poll loop.

        Returns True when a retry merged the PR (or it merged on its
        own during the wait); False otherwise, leaving the last stored
        merge exception in place for the caller's failure reporting.
        """
        if self._github_client is None or self.preview_mode or self._no_wait:
            return False

        pr_key = f"{owner}/{repo}#{pr_info.number}"
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._merge_timeout
        if self._run_deadline is not None:
            deadline = min(deadline, self._run_deadline)

        merge_method = self._pr_merge_methods.get(
            f"{owner}/{repo}", self.default_merge_method
        )

        self._pr_status(
            f"⏳ Waiting: {pr_info.html_url} [required workflows still running]",
            level="info",
        )

        # Refresh so the wait loop starts from live state rather than
        # the (possibly stale-``clean``) snapshot that let the doomed
        # merge dispatch in the first place.
        try:
            refreshed = await self._github_client.get(
                f"/repos/{owner}/{repo}/pulls/{pr_info.number}"
            )
            if isinstance(refreshed, dict):
                if refreshed.get("mergeable") is not None:
                    pr_info.mergeable = refreshed.get("mergeable")
                refreshed_state = refreshed.get("mergeable_state")
                if refreshed_state not in (None, "", "unknown"):
                    pr_info.mergeable_state = refreshed_state
                refreshed_head = (refreshed.get("head") or {}).get("sha")
                if refreshed_head:
                    pr_info.head_sha = refreshed_head
        except Exception as exc:
            self.log.debug(
                "Failed to refresh %s before required-workflows wait: %s",
                pr_key,
                exc,
            )

        self._track_pr_state(pr_info, "waiting")
        try:
            while True:
                closed, merged_during = await self._wait_for_auto_merge(
                    pr_info,
                    owner,
                    repo,
                    continue_states=("blocked", "unstable"),
                    deadline=deadline,
                )
                if closed:
                    return merged_during
                try:
                    merged = await self._github_client.merge_pull_request(
                        owner, repo, pr_info.number, merge_method
                    )
                except GitHubPermissionError:
                    raise
                except Exception as exc:
                    self._last_merge_exception[pr_key] = exc
                    if not self._merge_error_indicates_pending_workflows(str(exc)):
                        # The rejection reason changed (e.g. a workflow
                        # finished and *failed*) — terminal; let the
                        # caller classify and report it.
                        return False
                    merged = False
                if merged:
                    return True
                remaining = deadline - loop.time()
                if remaining <= 0:
                    return False
                await asyncio.sleep(min(self._merge_recheck_interval, remaining))
        finally:
            self._track_pr_state(pr_info, None)

    async def _is_pr_already_merged(
        self, pr_info: PullRequestInfo, owner: str, repo: str
    ) -> bool:
        """Return ``True`` if the PR has been merged externally.

        Called when the PR was already closed at fetch time to
        distinguish two outcomes:

        * The PR was merged while we were processing it (a
          concurrent ``dependamerge`` run at org scope, a human
          admin, or auto-merge landing mid-flight) — classify as
          ``SKIPPED`` because there is no remaining work.
        * The PR was closed without merging (superseded, no longer
          needed, or closed by a human) — callers classify as
          ``CLOSED``, which also needs no operator follow-up.

        Any API error during the recheck (network, rate limit,
        permission, unexpected payload) degrades to ``False`` so
        the caller falls back to its non-merged path.  The intent
        here is to upgrade the user experience for known benign
        races, not to mask genuine errors.
        """
        state, merged = await self._fetch_pr_state_now(pr_info, owner, repo)
        return state == "closed" and merged is True

    async def _fetch_pr_state_now(
        self, pr_info: PullRequestInfo, owner: str, repo: str
    ) -> tuple[str | None, bool | None]:
        """Best-effort fetch of a PR's current ``(state, merged)``.

        Returns ``(None, None)`` on any API error or unexpected
        payload so callers can fall back to their existing paths.
        Used to distinguish merged-externally (SKIPPED) from
        closed-without-merge (CLOSED — e.g. dependabot decided the
        update is no longer needed after sibling merges advanced the
        base branch) after a merge attempt fails.
        """
        if not self._github_client:
            return None, None
        try:
            pr_data = await self._github_client.get(
                f"/repos/{owner}/{repo}/pulls/{pr_info.number}"
            )
        except Exception as e:
            self.log.debug(
                "Failed to recheck %s/%s#%s state: %s",
                owner,
                repo,
                pr_info.number,
                e,
            )
            return None, None
        if not isinstance(pr_data, dict):
            return None, None
        state = pr_data.get("state")
        if not isinstance(state, str):
            return None, None
        merged = pr_data.get("merged")
        if not isinstance(merged, bool):
            # The full PR object always carries ``merged`` and ``merged_at``,
            # but a proxy or trimmed payload may omit the boolean. Derive it
            # from ``merged_at`` (an ISO timestamp when merged, ``null``
            # otherwise) when present rather than degrading a recoverable
            # payload to unknown; only a genuinely absent ``merged_at`` or an
            # unexpected type falls through to the conservative default.
            if "merged_at" not in pr_data:
                return None, None
            merged_at = pr_data.get("merged_at")
            if merged_at is not None and not isinstance(merged_at, str):
                return None, None
            merged = merged_at is not None
        return state, merged

    async def _is_pr_dirty_now(
        self, pr_info: PullRequestInfo, owner: str, repo: str
    ) -> bool:
        """Best-effort single GET: ``True`` only if the PR is *concretely* dirty.

        Called inside the per-repo dispatch lock, immediately before the
        merge is dispatched, to catch a PR that an earlier sibling merge
        in the same repo-scoped batch has turned ``dirty`` (the classic
        shared-``uv.lock`` conflict) since the one-shot fetch snapshot.
        Routing such a PR straight to conflict recovery avoids
        dispatching a doomed merge that 405s and then churns
        :meth:`_merge_pr_with_retry`'s retry loop against the stale
        ``clean`` snapshot (which misreads the 405 as a transient error
        on a mergeable PR and sleeps under the dispatch lock before
        re-fetching).

        Deliberately a *single* GET with no recompute poll — unlike
        :meth:`_refresh_pr_mergeability`, which polls GitHub's
        "still computing" window and therefore runs only *after* a
        failed merge, off the lock.  The dispatch lock is the one point
        serialised *and* ordered after a sibling merge, so polling here
        would serialise the whole repo batch.  We therefore act only on
        a *concrete* ``dirty``; a still-computing, closed, non-dirty, or
        errored result returns ``False`` so the merge attempt proceeds
        and the off-lock post-failure refresh settles anything that
        turns out to be a fresh conflict.

        Mutates ``pr_info`` (``mergeable``, ``mergeable_state``,
        ``head_sha``) only when it confirms a concrete ``dirty``, so the
        conflict handler and failure summary report accurate state.  The
        snapshot is otherwise left untouched — in particular a transient
        ``unknown`` never overwrites a concrete ``clean``, preserving the
        transient-405-on-``clean`` retry path in
        :meth:`_merge_pr_with_retry`.
        """
        if not self._github_client:
            return False
        try:
            data = await self._github_client.get(
                f"/repos/{owner}/{repo}/pulls/{pr_info.number}"
            )
        except Exception as exc:
            self.log.debug(
                "Pre-dispatch dirty check failed for %s/%s#%s: %s",
                owner,
                repo,
                pr_info.number,
                exc,
            )
            return False
        if not isinstance(data, dict):
            return False
        # A closed PR is handled by the caller's closed-PR path, not as
        # a conflict.
        if data.get("state") == "closed":
            return False
        if data.get("mergeable_state") != "dirty":
            return False
        # Concrete conflict: record it so ``_handle_merge_conflict`` and
        # the failure summary act on accurate, current state.
        pr_info.mergeable = data.get("mergeable")
        pr_info.mergeable_state = "dirty"
        head_sha = (data.get("head") or {}).get("sha")
        if head_sha:
            pr_info.head_sha = head_sha
        return True

    async def _refresh_pr_mergeability(
        self, pr_info: PullRequestInfo, owner: str, repo: str
    ) -> None:
        """Refresh ``pr_info`` with the PR's current live merge state.

        The batch of PRs is fetched once up front, so a worker may act
        on a snapshot taken seconds-to-minutes earlier.  In a
        repo-scoped run this is routinely wrong: merging one PR can
        immediately make a sibling PR ``dirty`` (the classic
        ``uv.lock`` / workflow-pin conflict) or ``behind``.  A
        concurrent ``dependamerge`` run elsewhere in the org can do the
        same.

        This method is the **post-failure** half of the conflict-
        detection pair, called from ``_merge_single_pr`` *after* a
        repo-scoped merge attempt returns falsy and always **outside**
        the per-repo dispatch lock.  It catches the case where a PR
        turned ``dirty`` *during* our own merge window (a sibling
        merged between the pre-dispatch check and the merge call) so a
        freshly-conflicted PR is routed to conflict recovery rather
        than reported as a generic merge failure.  The complementary
        **pre-dispatch** check is the single-GET ``_is_pr_dirty_now``,
        which runs *inside* the dispatch lock; the polling done here
        deliberately stays off that lock so GitHub's recompute window
        never serialises the whole repo batch.

        GitHub recomputes ``mergeable`` / ``mergeable_state``
        asynchronously after the base branch moves, reporting
        ``mergeable=None`` and ``mergeable_state="unknown"`` (or an
        empty string) in the gap — usually for a few seconds.  When we
        catch the PR in that window we poll up to
        :data:`MERGEABILITY_REFRESH_TIMEOUT_SECONDS` for a concrete
        value so the merge decision is made against real data.

        Mutates ``pr_info`` in place (``state``, ``mergeable``,
        ``mergeable_state``, ``head_sha``).  Best-effort: any API error
        leaves the existing snapshot untouched so the caller's
        downstream logic still runs.
        """
        if not self._github_client:
            return

        loop = asyncio.get_running_loop()
        deadline = loop.time() + MERGEABILITY_REFRESH_TIMEOUT_SECONDS
        # Poll cadence for the "still computing" window.  Kept short
        # (GitHub usually settles in ~5s) but never longer than the
        # configured recheck interval.
        poll_interval = min(2.0, self._merge_recheck_interval)

        while True:
            try:
                data = await self._github_client.get(
                    f"/repos/{owner}/{repo}/pulls/{pr_info.number}"
                )
            except Exception as exc:
                self.log.debug(
                    "Mergeability refresh failed for %s/%s#%s: %s",
                    owner,
                    repo,
                    pr_info.number,
                    exc,
                )
                return

            if not isinstance(data, dict):
                return

            state = data.get("state")
            if isinstance(state, str) and state:
                pr_info.state = state
            head_sha = (data.get("head") or {}).get("sha")
            if head_sha:
                pr_info.head_sha = head_sha

            mergeable = data.get("mergeable")
            mergeable_state = data.get("mergeable_state")

            # A closed PR will never resolve to a concrete mergeable
            # value; record what we have and let the caller's
            # closed-PR handling take over.
            if state == "closed":
                pr_info.mergeable = mergeable
                pr_info.mergeable_state = mergeable_state
                return

            # GitHub signals "still computing" with a null ``mergeable``
            # and an ``unknown``/empty ``mergeable_state``.  Keep
            # polling until it settles or the deadline passes.
            still_computing = mergeable is None or mergeable_state in (
                None,
                "",
                "unknown",
            )
            if not still_computing:
                pr_info.mergeable = mergeable
                pr_info.mergeable_state = mergeable_state
                return

            now = loop.time()
            if now >= deadline:
                # Timed out waiting for GitHub to settle.  Record the
                # latest values we did get (even if still computing) so
                # downstream logic sees GitHub's current best answer
                # rather than the older snapshot.  Reaching here means we
                # are still in the recompute window, where GitHub signals
                # "still computing" with ``mergeable=None`` and a
                # ``mergeable_state`` of ``None``, ``""`` or
                # ``"unknown"``.  Normalise any of those to a concrete
                # ``"unknown"`` and always record it, so a stale concrete
                # state (e.g. ``clean``) is never left in place —
                # consistent with the ``still_computing`` check above,
                # which treats ``None``/``""``/``"unknown"`` alike.
                if mergeable is not None:
                    pr_info.mergeable = mergeable
                pr_info.mergeable_state = mergeable_state or "unknown"
                self.log.debug(
                    "Mergeability for %s/%s#%s still computing after %.0fs; "
                    "proceeding with mergeable=%s state=%s",
                    owner,
                    repo,
                    pr_info.number,
                    MERGEABILITY_REFRESH_TIMEOUT_SECONDS,
                    pr_info.mergeable,
                    pr_info.mergeable_state,
                )
                return

            await asyncio.sleep(min(poll_interval, deadline - now))

    async def _wait_for_auto_merge(
        self,
        pr_info: PullRequestInfo,
        owner: str,
        repo: str,
        *,
        continue_states: tuple[str, ...],
        deadline: float | None = None,
        stop_on_clean: bool = True,
    ) -> tuple[bool, bool]:
        """Poll a PR until it merges, closes, settles, or times out.

        Registers ``owner/repo#N`` in ``_waiting_prs`` so the parallel
        progress ticker can render an aggregate countdown, then polls
        every ``_merge_recheck_interval`` seconds until one of:

          * ``mergeable_state`` becomes ``clean`` (only when
            ``stop_on_clean`` is True — the default; callers that have
            enabled auto-merge and want to observe the PR actually
            close set it False and include ``"clean"`` in
            ``continue_states`` instead),
          * the PR closes (capturing the ``merged`` flag so the caller
            can tell auto-merge success from closed-without-merge),
          * ``mergeable_state`` leaves ``continue_states`` (the caller
            decides what the new state means), or
          * the deadline passes.

        The total wait is bounded by ``merge_timeout`` unless an
        explicit ``deadline`` is supplied — used to share a single
        budget across sequential waits (e.g. the rebase-then-checks
        phases of conflict recovery).

        Mutates ``pr_info`` in place (``mergeable``,
        ``mergeable_state``, ``head_sha``, ``state``).  Returns
        ``(closed_during_wait, merged_during_wait)``.
        """
        if self._github_client is None:
            return False, False

        # Fire-and-forget (``max_wait == 0``): never block.  Returning
        # "not closed" lets callers fall through to the auto-merge-pending
        # path (Step 6 / the conflict handler arms auto-merge and reports
        # AUTO_MERGE_PENDING).  Auto-merge is armed by the caller before
        # this point, so GitHub still completes the merge later.
        if self._no_wait:
            return False, False

        pr_key = f"{owner}/{repo}#{pr_info.number}"
        loop = asyncio.get_running_loop()
        # Drive the wait off a monotonic deadline so the total is
        # bounded even if a single iteration over-sleeps slightly.
        if deadline is None:
            deadline = loop.time() + self._merge_timeout
        # Clamp to the owner-wide global ceiling (when set) so no single
        # PR's wait can push the whole run past ``max_wait``.
        if self._run_deadline is not None:
            deadline = min(deadline, self._run_deadline)
        # Local alias so the type checker can narrow
        # ``self._github_client`` across the await boundary.
        wait_client = self._github_client

        async with self._waiting_lock:
            self._waiting_prs[pr_key] = deadline

        # Track whether the PR was closed during the wait and, if so,
        # whether it was actually merged.  The REST payload's
        # ``merged`` boolean distinguishes auto-merge success from
        # closed-without-merge (a human closed it, dependabot
        # superseded it, etc.).
        closed_during_wait = False
        merged_during_wait = False
        first_poll = True
        try:
            # The whole poll loop is a wait on an external event
            # (auto-merge / CI / a rebase), so release this worker's
            # concurrency slot for its duration — a parked PR must
            # never starve runnable PRs (see ``slot_lease.py``).  The
            # polling GETs are paced by the HTTP client's own limits.
            async with parked():
                while loop.time() < deadline:
                    if stop_on_clean and pr_info.mergeable_state == "clean":
                        break
                    # Sleep no longer than the time remaining so we don't
                    # overshoot the deadline.  Clamp to non-negative: the
                    # ``while`` check and this ``time()`` call are not
                    # atomic, so a near-deadline crossing could otherwise
                    # pass ``asyncio.sleep`` a tiny negative value.  The
                    # first poll uses a much shorter delay (see
                    # ``MERGE_WAIT_FIRST_POLL_SECONDS``) so a PR that
                    # resolved the moment the wait started is detected
                    # promptly instead of a full interval late.
                    interval = (
                        min(
                            MERGE_WAIT_FIRST_POLL_SECONDS,
                            self._merge_recheck_interval,
                        )
                        if first_poll
                        else self._merge_recheck_interval
                    )
                    first_poll = False
                    remaining = max(0.0, deadline - loop.time())
                    await asyncio.sleep(min(interval, remaining))
                    try:
                        refreshed_wait = await wait_client.get(
                            f"/repos/{owner}/{repo}/pulls/{pr_info.number}"
                        )
                    except Exception as wait_exc:
                        self.log.debug(
                            "Failed to refresh PR state during auto-merge "
                            "wait for %s: %s",
                            pr_key,
                            wait_exc,
                        )
                        continue
                    if isinstance(refreshed_wait, dict):
                        # Only overwrite when present, and preserve the
                        # previous non-None value when GitHub returns null
                        # (it does so while recomputing) so the
                        # ``continue_states`` check below does not break the
                        # loop early on a transient null.
                        if "mergeable" in refreshed_wait:
                            refreshed_mergeable = refreshed_wait.get("mergeable")
                            if refreshed_mergeable is not None:
                                pr_info.mergeable = refreshed_mergeable
                        if "mergeable_state" in refreshed_wait:
                            refreshed_state = refreshed_wait.get("mergeable_state")
                            # Preserve the previous concrete state while
                            # GitHub is still recomputing mergeability: it
                            # returns ``null`` / ``""`` / ``"unknown"``
                            # transiently, and overwriting with those would
                            # push the state out of ``continue_states`` and
                            # break the wait loop early (e.g. a ``blocked``
                            # PR briefly going ``unknown`` would exit and
                            # trigger a premature manual merge).
                            if refreshed_state not in (None, "", "unknown"):
                                pr_info.mergeable_state = refreshed_state
                        # The head can change while we wait (rebase,
                        # force-push); keep it current so any later
                        # block-reason analysis queries the right commit.
                        refreshed_head = (refreshed_wait.get("head") or {}).get("sha")
                        if refreshed_head:
                            pr_info.head_sha = refreshed_head
                        if refreshed_wait.get("state") == "closed":
                            closed_during_wait = True
                            merged_during_wait = bool(
                                refreshed_wait.get("merged", False)
                            )
                            pr_info.state = "closed"
                            break
                    # A PR that becomes immediately mergeable while still
                    # ``unstable`` (only a non-required check is red) will
                    # never reach ``clean`` and never leave ``unstable``,
                    # so keeping it in ``continue_states`` would spin the
                    # wait to the deadline — the exact slow hang the
                    # Step 5.5 routing fix removes for PRs that *start*
                    # mergeable.  Break out as soon as GitHub reports
                    # ``unstable`` + ``mergeable is True`` so the caller can
                    # dispatch (auto-merge, already armed before this wait,
                    # will land it; failing that the caller merges directly).
                    if (
                        pr_info.mergeable_state == "unstable"
                        and pr_info.mergeable is True
                    ):
                        break
                    # Continue waiting only while the PR is in a state the
                    # caller still considers rescuable; any other value
                    # means it became mergeable, closed, or hit a terminal
                    # state, so exit and let the caller decide.
                    if pr_info.mergeable_state not in continue_states:
                        break
        finally:
            async with self._waiting_lock:
                self._waiting_prs.pop(pr_key, None)

        return closed_during_wait, merged_during_wait

    async def _request_dependabot_rebase(
        self, pr_info: PullRequestInfo, owner: str, repo: str
    ) -> bool:
        """Post ``@dependabot rebase`` on a conflicted dependabot PR.

        Dependabot responds by rebasing the PR branch onto the latest
        base, regenerating any lockfiles and re-signing the commit —
        the reliable way to clear a ``uv.lock`` / dependency conflict
        that a plain ``git rebase`` cannot resolve.

        Guards against duplicate comments: when a ``@dependabot
        rebase`` is already present the request is treated as
        in-flight and ``True`` is returned (the caller proceeds to
        wait).  Returns ``False`` only when the comment could not be
        posted.
        """
        if self._github_client is None:
            return False

        # Duplicate guard — don't stack rebase requests if one is
        # already pending from a previous run / trigger.
        try:
            comments = await self._github_client.get(
                f"/repos/{owner}/{repo}/issues/{pr_info.number}/comments"
                f"?per_page=100&direction=desc"
            )
            if isinstance(comments, list):
                for c in comments:
                    if not isinstance(c, dict):
                        continue
                    body = c.get("body")
                    if isinstance(body, str) and "@dependabot rebase" in body:
                        self.log.info(
                            "Existing @dependabot rebase comment on %s#%s; "
                            "treating rebase as already requested.",
                            pr_info.repository_full_name,
                            pr_info.number,
                        )
                        return True
        except Exception as exc:
            # If we can't list comments, fall through and post anyway:
            # a duplicate rebase request is harmless (dependabot just
            # rebases again) and is better than skipping recovery.
            self.log.debug(
                "Could not list comments for %s#%s before rebase request: %s",
                pr_info.repository_full_name,
                pr_info.number,
                exc,
            )

        try:
            await self._github_client.post_issue_comment(
                owner, repo, pr_info.number, "@dependabot rebase"
            )
            return True
        except Exception as exc:
            self.log.warning(
                "Failed to post @dependabot rebase on %s#%s: %s",
                pr_info.repository_full_name,
                pr_info.number,
                exc,
            )
            return False

    def _finish_conflict_close(
        self, pr_info: PullRequestInfo, result: MergeResult, merged: bool
    ) -> MergeResult:
        """Finalise a conflict-recovery result when the PR closed mid-wait.

        ``merged`` distinguishes auto-merge success (the rebase landed
        and GitHub merged the PR) from closed-without-merge (a human
        closed it, dependabot superseded it, etc.).
        """
        if merged:
            result.status = MergeStatus.MERGED
            self._pr_status(
                f"✅ Merged (auto-merge): {pr_info.html_url}",
                level="debug",
            )
        else:
            result.status = MergeStatus.CLOSED
            result.error = (
                "PR closed without merging during conflict rebase "
                "(no operator follow-up needed)"
            )
            self._pr_status(
                f"🚪 Closed without merging: {pr_info.html_url}",
                level="warning",
            )
        return result

    def _dependabot_is_rebasing(self, body: str | None) -> bool:
        """Return True when a PR body shows dependabot mid-self-rebase.

        Dependabot writes a notice into the PR body while it rebases the
        branch on its own (after the base moved).  Detecting it lets the
        conflict handler wait for the in-progress rebase instead of
        sending a redundant ``@dependabot rebase`` macro.
        """
        if not body:
            return False
        lowered = body.lower()
        return "dependabot is rebasing" in lowered or "is rebasing this pr" in lowered

    async def _handle_merge_conflict(
        self,
        pr_info: PullRequestInfo,
        owner: str,
        repo: str,
        result: MergeResult,
    ) -> MergeResult:
        """Recover from (or report) a PR with a real merge conflict.

        A ``dirty`` PR has no merge path of its own.  For a dependabot
        PR we ask dependabot to rebase — which regenerates lockfiles
        and re-signs the commit — then wait (bounded by
        ``merge_timeout``) for the rebase and required checks to land,
        approving the *rebased* commit and enabling auto-merge so
        GitHub completes the merge.  For any other author there is no
        automated way to resolve a content conflict, so we report it
        and fail fast (no wait).

        Must be called *outside* the per-repo dispatch lock: the wait
        can run for the full ``merge_timeout`` and must not block
        sibling merges.  Sets ``result`` and returns it.
        """
        # Non-dependabot authors: no comment macro regenerates a
        # conflicted lockfile, and a blind force-push would only break
        # the approval chain (this org forbids self-merge of pushed
        # commits).  Report the conflict and fail fast.
        if not is_dependabot(pr_info.author):
            self._pr_status(
                f"🔀 Merge conflict: {pr_info.html_url}",
                level="info",
            )
            result.status = MergeStatus.FAILED
            result.error = "merge conflicts"
            return result

        # Detect whether dependabot is already self-rebasing this PR.
        # When its base branch moves (e.g. a sibling PR merged), it
        # rebases the branch on its own and writes a marker into the PR
        # body while it does.  In that window we must not send a
        # duplicate ``@dependabot rebase`` macro: the in-progress rebase
        # will clear the conflict, so we wait for it rather than poke it.
        already_rebasing = self._dependabot_is_rebasing(pr_info.body)

        # Fire-and-forget (``max_wait == 0``): ask dependabot to rebase
        # (unless it already is), arm auto-merge, and report pending
        # without blocking this repository's serial worker.  Approval is
        # best-effort here — a subsequent dependabot force-push dismisses
        # it when the branch enables "dismiss stale reviews", which is the
        # documented trade-off of not waiting to approve the rebased head.
        if self._no_wait:
            if not already_rebasing:
                await self._request_dependabot_rebase(pr_info, owner, repo)
            try:
                await self._approve_pr(owner, repo, pr_info.number)
            except Exception as exc:
                self.log.debug(
                    "no-wait approve failed for %s/%s#%s: %s",
                    owner,
                    repo,
                    pr_info.number,
                    exc,
                )
            auto_ok = await self._enable_auto_merge_for_pr(pr_info, owner, repo)
            if auto_ok:
                result.status = MergeStatus.AUTO_MERGE_PENDING
                result.error = "auto-merge pending: conflict rebase requested (no-wait)"
                self._pr_status(
                    f"⏳ Auto-merge armed (no-wait): {pr_info.html_url}",
                    level="info",
                )
            else:
                # Auto-merge could not be armed (e.g. the repository has
                # the feature disabled), so nothing will merge this PR
                # later.  Report BLOCKED rather than a misleading
                # AUTO_MERGE_PENDING: the PR is left approved and rebased
                # but will not merge on its own.
                result.status = MergeStatus.BLOCKED
                result.error = (
                    "auto-merge unavailable (no-wait); PR approved and "
                    "rebase requested but not merged"
                )
                self._pr_status(
                    f"🛑 Auto-merge unavailable (no-wait): {pr_info.html_url}",
                    level="warning",
                )
            return result

        # Dependabot: request a rebase (regenerates the lockfile +
        # signs the commit) unless it is already rebasing on its own.
        if already_rebasing:
            self._pr_status(
                f"🔄 Dependabot already rebasing: {pr_info.html_url}",
                level="info",
            )
        else:
            self._pr_status(
                f"🔄 Requesting dependabot rebase: {pr_info.html_url}",
                level="info",
            )
            if not await self._request_dependabot_rebase(pr_info, owner, repo):
                self._pr_status(
                    f"🔀 Merge conflict: {pr_info.html_url}",
                    level="info",
                )
                result.status = MergeStatus.FAILED
                result.error = "merge conflicts"
                return result

        # Share a single ``merge_timeout`` budget across both wait
        # phases (waiting for the rebase, then for checks).
        deadline = asyncio.get_running_loop().time() + self._merge_timeout

        # Wait for dependabot's rebase to clear the conflict.
        # Keep waiting while still ``dirty`` or while GitHub recomputes
        # mergeability (a transient null is preserved as the prior
        # ``dirty`` by ``_wait_for_auto_merge``).
        self._track_pr_state(pr_info, "rebasing")
        try:
            closed, merged = await self._wait_for_auto_merge(
                pr_info,
                owner,
                repo,
                continue_states=("dirty", "unknown", ""),
                deadline=deadline,
            )
        finally:
            self._track_pr_state(pr_info, None)
        if closed:
            return self._finish_conflict_close(pr_info, result, merged)
        if pr_info.mergeable_state == "dirty":
            # Timed out still conflicting — the rebase did not happen
            # or could not resolve the conflict.
            self._pr_status(
                f"🔀 Merge conflict: {pr_info.html_url}",
                level="info",
            )
            result.status = MergeStatus.FAILED
            result.error = "merge conflicts"
            return result

        # Conflict cleared.  Approve the rebased commit *now* (not
        # before — approving the pre-rebase head would just be
        # dismissed by dependabot's force-push, producing the
        # duplicate approvals we want to avoid) and try to enable
        # auto-merge.  Handle an approval failure (permission / API
        # error) here rather than letting it bubble to the generic
        # catch-all, which would lose the conflict-recovery context.
        try:
            await self._approve_pr(owner, repo, pr_info.number)
        except Exception as exc:
            self.log.warning(
                "Failed to approve %s/%s#%s after rebase: %s",
                owner,
                repo,
                pr_info.number,
                exc,
            )
            result.status = MergeStatus.FAILED
            result.error = f"rebase cleared the conflict but approval failed: {exc}"
            self._pr_status(f"❌ Failed: {pr_info.html_url}", level="error")
            return result
        auto_ok = await self._enable_auto_merge_for_pr(pr_info, owner, repo)
        if auto_ok:
            self._pr_status(
                f"🤖 Auto-merge: {pr_info.html_url}",
                level="debug",
            )

        # Wait (sharing the deadline) for required checks to
        # land.  When auto-merge is armed we wait *through* ``clean``
        # (``stop_on_clean=False``) so we can observe GitHub actually
        # close the PR and report MERGED.  When auto-merge could NOT be
        # enabled, waiting through ``clean`` would just spin until the
        # deadline (nothing would merge the PR), so we stop on ``clean``
        # and merge it ourselves below.
        if auto_ok:
            continue_states: tuple[str, ...] = (
                "clean",
                "blocked",
                "behind",
                "unstable",
                "unknown",
                "",
            )
        else:
            continue_states = ("blocked", "behind", "unstable", "unknown", "")
        self._track_pr_state(pr_info, "rebased")
        try:
            closed, merged = await self._wait_for_auto_merge(
                pr_info,
                owner,
                repo,
                continue_states=continue_states,
                deadline=deadline,
                stop_on_clean=not auto_ok,
            )
        finally:
            self._track_pr_state(pr_info, None)
        if closed:
            return self._finish_conflict_close(pr_info, result, merged)

        if auto_ok:
            # Auto-merge is armed: GitHub will complete the merge once
            # the required checks pass (often after our run ends).
            result.status = MergeStatus.AUTO_MERGE_PENDING
            result.error = "auto-merge pending: checks after conflict rebase"
            self._pr_status(
                f"⏳ Waiting: {pr_info.html_url} [auto-merge after rebase]",
                level="debug",
            )
            return result

        # Auto-merge could not be enabled.  If the rebase left the PR
        # mergeable, merge it directly now; otherwise it will not land
        # on its own — report the failure rather than a misleading
        # ``AUTO_MERGE_PENDING`` that would never resolve.
        if pr_info.mergeable_state == "clean":
            dispatch_lock = await self._get_merge_dispatch_lock(owner, repo)
            async with dispatch_lock:
                merged = await self._merge_pr_with_retry(pr_info, owner, repo)
            if merged:
                result.status = MergeStatus.MERGED
                self._pr_status(
                    f"✅ Merged: {pr_info.html_url}",
                    level="debug",
                )
                return result

        result.status = MergeStatus.FAILED
        result.error = (
            "rebase cleared the conflict but the PR could not be merged "
            "(auto-merge unavailable)"
        )
        self._pr_status(f"❌ Failed: {pr_info.html_url}", level="error")
        return result

    async def _report_merge_failure(
        self,
        pr_info: PullRequestInfo,
        owner: str,
        repo: str,
        result: MergeResult,
        failure_reason: str,
    ) -> MergeResult:
        """Report a failed merge, upgrading to a stuck-check cause if found.

        Called when ``_merge_pr_with_retry`` failed and no dependabot
        recreate produced a replacement PR.  For a non-dependabot PR
        we check whether a required check is stuck (Option A): if so,
        print ``⚠️ Stuck check`` and arm auto-merge (when the PR is
        otherwise mergeable) so it lands once the check is
        re-triggered, without a force-push that would break this org's
        self-merge rule.  Otherwise emit the generic failure line.

        Sets ``result`` to ``FAILED`` and returns it.
        """
        stuck_reported = False
        if not is_dependabot(pr_info.author) and not self.preview_mode:
            try:
                detection = await self._detect_stuck_required_check(pr_info)
            except Exception as exc:
                self.log.debug(
                    "_detect_stuck_required_check failed for %s#%s: %s",
                    pr_info.repository_full_name,
                    pr_info.number,
                    exc,
                )
                detection = None
            if detection is not None and detection[0]:
                stuck_check = detection[1]
                self._pr_status(
                    f"⚠️ Stuck check: {pr_info.html_url} [{stuck_check}]",
                    level="warning",
                )
                # Arm auto-merge when the PR is otherwise mergeable
                # (not dirty) so it lands automatically once the stuck
                # check is re-triggered, without a second review round.
                # Approve the current head first (approve-on-demand): the
                # PR is no longer approved up-front, so auto-merge would
                # otherwise wait forever on a missing review.
                if pr_info.mergeable_state != "dirty":
                    await self._enable_auto_merge_with_approval(pr_info, owner, repo)
                result.error = f"stuck check: {stuck_check}"
                stuck_reported = True

        result.status = MergeStatus.FAILED
        if not stuck_reported:
            # Use the (now informative) failure reason as the result
            # error too, so the end-of-run summary surfaces the real
            # cause rather than a generic "all retry attempts" line.
            result.error = failure_reason or "Failed to merge after all retry attempts"
        # Keep the live output terse: the full (often long) reason is
        # shown in the end-of-run summary via ``result.error``, so
        # repeating it inline only duplicates it.  The ``⚠️ Stuck
        # check`` line above already carries the cause for stuck PRs.
        if not stuck_reported:
            self._pr_status(f"❌ Failed: {pr_info.html_url}", level="error")
        return result

    def _get_failure_summary(self, pr_info: PullRequestInfo) -> str:
        """
        Generate a detailed failure summary based on PR state.

        Args:
            pr_info: Pull request information

        Returns:
            Detailed description of why the merge failed
        """
        # Check if we have a stored exception for this PR
        pr_key = f"{pr_info.repository_full_name}#{pr_info.number}"
        last_exception = self._last_merge_exception.get(pr_key)
        self.log.debug(
            f"_get_failure_summary called for {pr_key}, mergeable_state={pr_info.mergeable_state}, mergeable={pr_info.mergeable}, has_exception={last_exception is not None}"
        )
        if last_exception:
            error_msg = str(last_exception)
            self.log.debug(f"Last exception for {pr_key}: {error_msg[:200]}")
            # The merge layer (github_async.merge_pull_request) embeds
            # GitHub's own explanation after a "GitHub: " marker — the
            # ruleset violation, "Required workflows ... are not
            # satisfied", required-check names, etc.  This is the
            # actionable cause, so surface it ahead of any generic
            # state-based inference.  We trim the PR-state context we
            # appended after it so the reason stays concise.
            marker = "GitHub: "
            if marker in error_msg:
                detail = error_msg.split(marker, 1)[1]
                detail = detail.split(" (PR state:", 1)[0].strip()
                if detail:
                    return detail[:300]
            # Workflow-scope failures surface in several phrasings: the
            # PermissionError messages we raise ("Missing 'workflow' scope",
            # "Missing workflow permissions") and GitHub's own response body
            # ("refusing to allow ... without `workflow` scope").  Match all
            # of them, but require the word "workflow" so unrelated 403s do
            # not get mislabelled as a scope problem.
            error_lower = error_msg.lower()
            if "workflow" in error_lower and (
                "missing 'workflow' scope" in error_lower
                or "missing workflow permissions" in error_lower
                or "refusing to allow" in error_lower
            ):
                return "missing 'workflow' token scope"
            # The token already had the 'workflow' scope but GitHub still
            # refused the workflow-file update — a ruleset or SSO problem,
            # not a scope problem.  Report it as such rather than telling the
            # user to add a scope they already hold.
            elif "blocked by something other than token scope" in error_lower:
                return (
                    "workflow update blocked by repository ruleset or SSO "
                    "(token already has 'workflow' scope)"
                )
            elif "403" in error_msg and "forbidden" in error_lower:
                return "insufficient permissions"
            # Surface transient HTTP errors (502, 405 etc.) accurately instead
            # of falling through to infer a reason from mergeable_state, which
            # may be stale or misleading (e.g. "clean" → "branch protection").
            elif "405" in error_msg and "Method Not Allowed" in error_msg:
                if pr_info.mergeable_state in ("clean", "unstable"):
                    return (
                        "GitHub API returned transient 405 error "
                        "(PR appears mergeable — GitHub may be experiencing issues, "
                        "see https://www.githubstatus.com)"
                    )
                # For non-clean states, fall through to state-based analysis below
            elif "502" in error_msg:
                return (
                    "GitHub API returned 502 Bad Gateway "
                    "(GitHub may be experiencing issues, "
                    "see https://www.githubstatus.com)"
                )

        if pr_info.mergeable_state == "behind":
            return "behind base branch"
        elif pr_info.mergeable_state == "blocked":
            # Use detailed block analysis for blocked PRs
            try:
                from .github_client import GitHubClient

                client = GitHubClient(token=self.token)
                detailed_reason = client._analyze_block_reason(pr_info)
                # Convert the detailed reason to a more concise format for console output
                if detailed_reason.startswith("Blocked by failing check:"):
                    check_name = detailed_reason.replace(
                        "Blocked by failing check: ", ""
                    )
                    return f"failing check: {check_name}"
                elif (
                    detailed_reason.startswith("Blocked by")
                    and "failing checks" in detailed_reason
                ):
                    return detailed_reason.replace("Blocked by ", "").lower()
                elif "Human reviewer requested changes" in detailed_reason:
                    return "human reviewer requested changes"
                elif "Copilot" in detailed_reason:
                    return detailed_reason.replace("Blocked by ", "").lower()
                elif "ruleset" in detailed_reason.lower():
                    return "repository ruleset prevents merge"
                elif "undetermined reason" in detailed_reason.lower():
                    return "blocked for an undetermined reason"
                elif "branch protection" in detailed_reason.lower():
                    return "branch protection rules prevent merge"
                else:
                    return detailed_reason.replace("Blocked by ", "").lower()
            except Exception as e:
                self.log.debug(f"Failed to get detailed block reason: {e}")
                # Fallback to generic message

            # Fallback logic when detailed analysis fails
            if pr_info.mergeable is True:
                return "branch protection rules prevent merge"
            else:
                return "blocked by failing status checks"
        elif pr_info.mergeable_state == "dirty":
            return "merge conflicts"
        elif pr_info.mergeable_state == "draft":
            return "draft PR"
        elif pr_info.mergeable is False:
            return "cannot update protected ref - organization or branch protection rules prevent merge"
        elif pr_info.mergeable_state == "unknown":
            # For unknown state, try to get more details using the GitHub client
            try:
                from .github_client import GitHubClient

                client = GitHubClient(token=self.token)
                detailed_reason = client._analyze_block_reason(pr_info)
                if "failing check" in detailed_reason.lower():
                    if detailed_reason.startswith("Blocked by failing check:"):
                        check_name = detailed_reason.replace(
                            "Blocked by failing check: ", ""
                        )
                        return f"failing check: {check_name}"
                    else:
                        return detailed_reason.replace("Blocked by ", "").lower()
                else:
                    return detailed_reason.replace("Blocked by ", "").lower()
            except Exception as e:
                self.log.debug(f"Failed to analyze unknown state: {e}")
                return "status checks pending or failed"
        else:
            return f"merge failed: {pr_info.mergeable_state}"

    async def _get_merge_method_for_repo(self, owner: str, repo: str) -> str:
        """
        Get the appropriate merge method for a specific repository based on branch protection settings.

        Args:
            owner: Repository owner
            repo: Repository name

        Returns:
            Merge method to use: "merge", "squash", or "rebase"
        """
        if not self._github_service:
            self.log.warning("GitHubService not available, using default merge method")
            return self.default_merge_method

        try:
            protection_settings = (
                await self._github_service.get_branch_protection_settings(
                    owner, repo, "main"
                )
            )

            # Determine appropriate merge method
            merge_method = self._github_service.determine_merge_method(
                protection_settings, self.default_merge_method
            )

            if merge_method != self.default_merge_method:
                self.log.debug(
                    f"Repository {owner}/{repo} requires '{merge_method}' merge method "
                    f"(protection: requiresLinearHistory={protection_settings and protection_settings.get('requiresLinearHistory', False)})"
                )

            return merge_method

        except Exception as e:
            self.log.warning(
                f"Failed to determine merge method for {owner}/{repo}, using default '{self.default_merge_method}': {e}"
            )
            return self.default_merge_method

    async def _handle_merge_failure(
        self, pr_info: PullRequestInfo, owner: str, repo: str
    ) -> bool:
        """
        Handle a merge failure and determine if we should retry.

        Args:
            pr_info: Pull request information
            owner: Repository owner
            repo: Repository name

        Returns:
            True if we should retry, False otherwise
        """
        if not self._github_client:
            return False

        # Check if the branch is out of date and we can fix it
        if self.fix_out_of_date and pr_info.mergeable_state == "behind":
            try:
                self.log.info(
                    f"PR {owner}/{repo}#{pr_info.number} is behind - updating branch"
                )
                await self._github_client.update_branch(owner, repo, pr_info.number)
                # Wait a moment for GitHub to process the update
                await asyncio.sleep(min(2.0, self._merge_recheck_interval))
                return True
            except Exception as e:
                self.log.error(
                    f"Failed to update branch for PR {owner}/{repo}#{pr_info.number}: {e}"
                )

        # For other failure types, don't retry
        return False

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

    async def _get_org_settings(self, owner: str) -> dict[str, Any] | None:
        """
        Get organization-level settings, with caching.

        Organization settings (e.g. web_commit_signoff_required) don't change
        between PRs in the same org, so we cache the result for the lifetime
        of the merge session.

        Args:
            owner: Organization/owner name

        Returns:
            Organization settings dict, or None if the lookup failed
        """
        # Fast path: no lock needed if already cached
        if owner in self._org_settings_cache:
            return self._org_settings_cache[owner]

        # Acquire a per-owner lock so concurrent lookups for the same
        # org are serialised, but lookups for *different* orgs proceed
        # in parallel without blocking each other.
        async with self._org_settings_locks_lock:
            if owner not in self._org_settings_locks:
                self._org_settings_locks[owner] = asyncio.Lock()
            owner_lock = self._org_settings_locks[owner]

        async with owner_lock:
            # Re-check after acquiring the per-owner lock (another
            # task may have populated the cache while we waited).
            if owner in self._org_settings_cache:
                return self._org_settings_cache[owner]

            if not self._github_client:
                return None

            try:
                org_data = await self._github_client.get(f"/orgs/{owner}")
                if isinstance(org_data, dict):
                    self._org_settings_cache[owner] = org_data
                    # Log org-level details once, not per-PR
                    web_commit_signoff = org_data.get(
                        "web_commit_signoff_required", False
                    )
                    if web_commit_signoff:
                        self.log.debug(f"Organization {owner} requires commit signoff")
                    return org_data
                else:
                    self._org_settings_cache[owner] = None
                    return None
            except Exception as e:
                self.log.debug(
                    f"Could not check organization settings for {owner}: {e}"
                )
                self._org_settings_cache[owner] = None
                return None

    @staticmethod
    def _rules_require_approval(rules: Any) -> bool:
        """Return True if any effective branch rule mandates an approval.

        ``rules`` is the JSON body returned by
        ``GET /repos/{owner}/{repo}/rules/branches/{branch}`` — a flat
        list of the rules that *actually apply* to the branch, with all
        ruleset conditions (repository include/exclude, ref matching)
        already evaluated server-side and org- and repo-level rulesets
        already merged.  We treat a branch as requiring an approval when a
        ``pull_request`` rule asks for at least one approving review.

        This is intentionally org-agnostic: it keys off the rule *type*
        and its ``required_approving_review_count`` parameter, never the
        ruleset's name, so it works for any organization's naming.
        """
        if not isinstance(rules, list):
            return False
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            if rule.get("type") != "pull_request":
                continue
            params = rule.get("parameters")
            if not isinstance(params, dict):
                # A pull_request rule with no readable parameters still
                # signals that reviews are governed here; treat the
                # presence of the rule as requiring approval rather than
                # risk a doomed merge-first attempt.
                return True
            count = params.get("required_approving_review_count")
            if isinstance(count, int) and count >= 1:
                return True
        return False

    async def _approve_if_review_mandated(
        self, pr_info: PullRequestInfo, owner: str, repo: str, pr_key: str
    ) -> None:
        """Approve up-front when the base branch mandates a review to merge.

        No-ops in preview mode, when the PR was already approved this run,
        or when the base branch carries no required-approval rule.  This
        is the proactive counterpart to
        :meth:`_approve_and_retry_if_review_required`: detecting the
        requirement (org-agnostically, from the branch's effective rules)
        before dispatch avoids a guaranteed-to-fail merge attempt for
        organizations that gate every merge on an approving review.
        """
        if self.preview_mode or pr_key in self._recently_approved:
            return
        if await self._branch_requires_approval(owner, repo, pr_info.base_branch):
            await self._ensure_pr_approved(pr_info, owner, repo)

    async def _org_approval_rulesets(self, org: str) -> list[dict[str, Any]] | None:
        """Enumerate active org rulesets that mandate an approving review.

        Queried **once per org** and cached for the run — the approval
        requirement originates from a single organization ruleset, so it
        is wasteful to rediscover it per repository.  Returns one entry
        per approval-mandating ruleset (``{"name", "conditions"}``), ``[]``
        when the org mandates none, or ``None`` when enumeration failed
        (e.g. the token cannot read org rulesets) so the caller can fall
        back to the authoritative per-repo endpoint.

        The first time an org is found to gate merges on a review a single
        user-facing line is emitted, so the requirement is visible at the
        point of detection rather than buried in debug logs.
        """
        if org in self._org_approval_cache:
            return self._org_approval_cache[org]

        async with self._org_approval_locks_lock:
            if org not in self._org_approval_locks:
                self._org_approval_locks[org] = asyncio.Lock()
            org_lock = self._org_approval_locks[org]

        async with org_lock:
            # Re-check after acquiring the per-org lock (another task may
            # have populated the cache while we waited).
            if org in self._org_approval_cache:
                return self._org_approval_cache[org]

            if not self._github_client:
                return None

            result: list[dict[str, Any]] = []
            try:
                # The list endpoint is paginated (default page size 30),
                # so an org with many rulesets could otherwise silently
                # drop an approval-mandating one.  Walk every page.
                page = 1
                per_page = 100
                while True:
                    rulesets = await self._github_client.get(
                        f"/orgs/{org}/rulesets?per_page={per_page}&page={page}"
                    )
                    if not isinstance(rulesets, list) or not rulesets:
                        break
                    for rs in rulesets:
                        if not isinstance(rs, dict):
                            continue
                        # Only active branch rulesets gate merges;
                        # "evaluate" and "disabled" rulesets do not block,
                        # and tag rulesets are irrelevant to PR merges.
                        if rs.get("enforcement") != "active":
                            continue
                        if rs.get("target", "branch") != "branch":
                            continue
                        rid = rs.get("id")
                        if rid is None:
                            continue
                        detail = await self._github_client.get(
                            f"/orgs/{org}/rulesets/{rid}"
                        )
                        if not isinstance(detail, dict):
                            continue
                        if self._rules_require_approval(detail.get("rules")):
                            result.append(
                                {
                                    "name": rs.get("name", ""),
                                    "conditions": detail.get("conditions") or {},
                                }
                            )
                    if len(rulesets) < per_page:
                        break
                    page += 1
            except Exception as e:
                # Enumeration failed (often a permission/SSO problem).
                # Cache ``None`` so callers consult the per-repo endpoint
                # rather than silently skipping proactive approval.
                self.log.debug(f"Could not enumerate org rulesets for {org}: {e}")
                self._org_approval_cache[org] = None
                return None

            self._org_approval_cache[org] = result
            if result:
                names = ", ".join(r["name"] for r in result if r.get("name")) or (
                    "unnamed ruleset"
                )
                log_and_print(
                    self.log,
                    self._console,
                    "🔐 Organization requires approving reviews before merging\n"
                    f"Ruleset: {names}",
                    level="info",
                )
            return result

    @staticmethod
    def _ruleset_name_matches(
        name: str, include: list[Any], exclude: list[Any]
    ) -> bool:
        """Evaluate a ruleset ``repository_name`` condition against a repo.

        ``include``/``exclude`` are fnmatch-style globs; the sentinel
        ``~ALL`` matches every repository.  A repo is in scope when it
        matches an include pattern and no exclude pattern.
        """

        def match_any(patterns: list[Any]) -> bool:
            for pat in patterns:
                if pat == "~ALL":
                    return True
                if isinstance(pat, str) and fnmatch.fnmatch(name, pat):
                    return True
            return False

        if exclude and match_any(exclude):
            return False
        if not include:
            return False
        return match_any(include)

    @staticmethod
    def _ruleset_ref_matches(
        branch: str, include: list[Any], exclude: list[Any]
    ) -> bool | None:
        """Evaluate a ruleset ``ref_name`` condition against a branch.

        Returns ``True``/``False`` when it can be decided locally, or
        ``None`` when it cannot (so the caller consults the authoritative
        per-repo endpoint).  ``~ALL`` matches any branch.  ``~DEFAULT_BRANCH``
        is treated as in scope: confirming it would need an extra per-repo
        default-branch lookup, and the automation PRs this gates target
        the default branch — a spurious approval on a non-default-base PR
        (which we were about to merge anyway) is harmless.
        """
        ref = f"refs/heads/{branch}"

        def match_any(patterns: list[Any]) -> bool:
            for pat in patterns:
                if pat in ("~ALL", "~DEFAULT_BRANCH"):
                    return True
                if isinstance(pat, str) and (
                    fnmatch.fnmatch(ref, pat) or fnmatch.fnmatch(branch, pat)
                ):
                    return True
            return False

        if exclude and match_any(exclude):
            return False
        if not include:
            # An empty include is unusual; defer to the authoritative
            # endpoint rather than guess.
            return None
        return match_any(include)

    def _ruleset_condition_applies(
        self, conditions: Any, repo: str, branch: str
    ) -> bool | None:
        """Whether a ruleset's ``conditions`` select ``repo@branch``.

        Returns ``True``/``False`` when the verdict is decidable from the
        ``repository_name`` and ``ref_name`` conditions, or ``None`` when
        the ruleset uses a condition type we do not evaluate locally
        (e.g. ``repository_id`` or ``repository_property``) so the caller
        falls back to GitHub's authoritative per-repo evaluation.
        """
        if not isinstance(conditions, dict):
            return None
        # Any condition type beyond the two we evaluate means we cannot be
        # sure locally — signal the caller to ask GitHub directly.
        if any(key not in ("repository_name", "ref_name") for key in conditions):
            return None

        repo_cond = conditions.get("repository_name")
        if isinstance(repo_cond, dict):
            if not self._ruleset_name_matches(
                repo,
                repo_cond.get("include") or [],
                repo_cond.get("exclude") or [],
            ):
                return False

        ref_cond = conditions.get("ref_name")
        if isinstance(ref_cond, dict):
            ref_applies = self._ruleset_ref_matches(
                branch,
                ref_cond.get("include") or [],
                ref_cond.get("exclude") or [],
            )
            if ref_applies is not True:
                # False or None (undecidable) — propagate so an undecidable
                # ref falls back to the authoritative endpoint.
                return ref_applies

        return True

    async def _branch_requires_approval(
        self, owner: str, repo: str, branch: str
    ) -> bool:
        """Whether ``owner/repo@branch`` mandates an approving review to merge.

        Some organizations enforce a repository ruleset that requires at
        least one approving review before *any* merge is permitted (the
        ``lfreleng-actions`` "Base Protections" ruleset is one example).
        Under "merge first, approve on demand" every such PR would incur a
        guaranteed-to-fail merge attempt before we approve and retry.
        Detecting the requirement up-front lets us approve proactively and
        skip that doomed round-trip.

        Detection is **org-first**: the org's rulesets are enumerated once
        (see :meth:`_org_approval_rulesets`) and their conditions are
        evaluated locally, so a whole org-wide run needs a single ruleset
        query rather than one effective-rules call per repository.  Only
        when a ruleset uses a condition we cannot evaluate locally, or org
        enumeration was not possible, do we fall back to GitHub's
        authoritative per-repo ``rules/branches`` endpoint.  The resolved
        verdict is cached per repo+branch.
        """
        cache_key = f"{owner}/{repo}@{branch}"
        if cache_key in self._branch_approval_cache:
            return self._branch_approval_cache[cache_key]

        async with self._branch_approval_locks_lock:
            if cache_key not in self._branch_approval_locks:
                self._branch_approval_locks[cache_key] = asyncio.Lock()
            branch_lock = self._branch_approval_locks[cache_key]

        async with branch_lock:
            # Re-check after acquiring the per-branch lock (another task
            # may have populated the cache while we waited).
            if cache_key in self._branch_approval_cache:
                return self._branch_approval_cache[cache_key]

            rulesets = await self._org_approval_rulesets(owner)

            requires = False
            # ``None`` means org enumeration failed; consult the per-repo
            # endpoint.  An empty list means the org mandates no approval
            # (repo-level rulesets, if any, are covered by the reactive
            # approve-on-demand safety net).
            need_authoritative = rulesets is None
            for rs in rulesets or []:
                applies = self._ruleset_condition_applies(
                    rs.get("conditions"), repo, branch
                )
                if applies is True:
                    requires = True
                    break
                if applies is None:
                    need_authoritative = True

            if not requires and need_authoritative:
                requires = await self._effective_branch_requires_approval(
                    owner, repo, branch
                )

            self._branch_approval_cache[cache_key] = requires
            if requires:
                self.log.debug(
                    "Branch %s requires an approving review before merge; "
                    "approving proactively",
                    cache_key,
                )
            return requires

    async def _effective_branch_requires_approval(
        self, owner: str, repo: str, branch: str
    ) -> bool:
        """Authoritative per-repo fallback for the approval requirement.

        Uses ``GET /repos/{owner}/{repo}/rules/branches/{branch}`` which
        returns the *effective* rules for the branch — every applicable
        org- and repo-level ruleset, with all conditions already evaluated
        by GitHub.  Used only when the org-first path cannot decide (an
        unrecognised condition type, or org enumeration was unavailable).
        On any error returns ``False`` so the reactive approve-on-demand
        path remains the safety net rather than blocking the merge.
        """
        if not self._github_client:
            return False
        try:
            # Branch names can contain "/" (e.g. "release/v1"); encode the
            # whole segment so it routes to the right endpoint rather than
            # 404ing and being mistaken for "no rules".
            rules = await self._github_client.get(
                f"/repos/{owner}/{repo}/rules/branches/{quote(branch, safe='')}"
            )
            return self._rules_require_approval(rules)
        except Exception as e:
            self.log.debug(
                f"Could not read effective branch rules for {owner}/{repo}@{branch}: {e}"
            )
            return False

    async def _predict_merge_outcome(
        self, owner: str, repo: str, pr_number: int, merge_method: str
    ) -> tuple[bool, str]:
        """Best-effort, read-only prediction of whether a PR would merge.

        This is a **preview-only** probe used to render the dry-run
        evaluation.  It inspects the PR's ``mergeable`` / ``mergeable_state``
        and, for ``blocked`` PRs, consults :meth:`analyze_block_reason` to
        produce a one-line verdict.

        It deliberately has **no authority over the real merge**: GitHub's
        ``mergeable_state`` can lag the true state, and repository rulesets
        are invisible to this code path, so a confident "would block"
        verdict here can still be wrong.  The execution path therefore does
        not gate on this prediction — it attempts the merge and treats
        GitHub's actual response (Step 6 and ``_merge_pr_with_retry``) as
        authoritative.  Only the preview path calls this method.

        Args:
            owner: Repository owner
            repo: Repository name
            pr_number: Pull request number
            merge_method: Merge method to test

        Returns:
            Tuple of (can_merge: bool, reason: str)
        """
        if not self._github_client:
            return False, "GitHub client not initialized"

        try:
            # Check organization-level restrictions (cached per org)
            await self._get_org_settings(owner)

            # Note: Removed DCO signoff check as web_commit_signoff_required only affects
            # web-based commits, not PR merges. DCO enforcement for PRs is handled by
            # status checks/apps, not repository settings.

            pr_data = await self._github_client.get(
                f"/repos/{owner}/{repo}/pulls/{pr_number}"
            )

            if isinstance(pr_data, dict):
                mergeable_state = pr_data.get("mergeable_state", "unknown")
                mergeable = pr_data.get("mergeable")
                head_sha = pr_data.get("head", {}).get("sha", "")

                self.log.debug(
                    f"PR {owner}/{repo}#{pr_number} REST API status: mergeable={mergeable}, mergeable_state={mergeable_state}"
                )

                # Check for specific blocking conditions that indicate protection rules
                if mergeable_state == "blocked" and mergeable is False:
                    # Before declaring the PR unmergeable, analyze WHY it's blocked.
                    # If the only blocker is "requires approval", the tool is about to
                    # provide that approval — so we should allow the merge to proceed.
                    # Note: we only call analyze_block_reason when mergeable is False
                    # to avoid unnecessary API traffic; when mergeable is True/None the
                    # code falls through to the pass-through return at the end.
                    block_reason = ""
                    if head_sha and self._github_client:
                        try:
                            block_reason = (
                                await self._github_client.analyze_block_reason(
                                    owner,
                                    repo,
                                    pr_number,
                                    head_sha,
                                    base_branch=(pr_data.get("base") or {}).get("ref"),
                                )
                            )
                            self.log.debug(
                                f"PR {owner}/{repo}#{pr_number} block reason: {block_reason}"
                            )
                        except Exception as analyze_err:
                            self.log.debug(
                                f"Could not analyze block reason for {owner}/{repo}#{pr_number}: {analyze_err}"
                            )

                    # If the PR is only blocked because it needs approval, allow it
                    # through — the tool will approve it before attempting merge.
                    if "requires approval" in block_reason.lower():
                        self.log.info(
                            f"PR {owner}/{repo}#{pr_number} is blocked pending approval — tool will approve before merge"
                        )
                        return True, "PR blocked pending approval (tool will approve)"

                    # For other blocking reasons, check force level
                    if self.force_level in ["code-owners", "protection-rules", "all"]:
                        self.log.info(
                            f"Force level '{self.force_level}' bypassing branch protection rules for {owner}/{repo}#{pr_number}"
                        )
                        return True, "branch protection bypassed by force level"
                    return (
                        False,
                        f"branch protection rules prevent merge ({block_reason or 'blocked'})",
                    )
                elif mergeable_state == "dirty":
                    return False, "merge conflicts"
                elif mergeable_state == "behind":
                    if not self.fix_out_of_date:
                        return (
                            False,
                            "PR is behind base branch and --no-fix option is set",
                        )
                    # Otherwise it's fixable
                elif mergeable is False and mergeable_state == "unknown":
                    # This often indicates hidden branch protection rules
                    if self.force_level in ["code-owners", "protection-rules", "all"]:
                        self.log.info(
                            f"Force level '{self.force_level}' bypassing hidden branch protection rules for {owner}/{repo}#{pr_number}"
                        )
                        return True, "hidden branch protection bypassed by force level"
                    return (
                        False,
                        "cannot update protected ref - organization or branch protection rules prevent merge",
                    )

            return True, "merge capability test passed"

        except Exception as e:
            error_msg = str(e)
            self.log.debug(
                f"Exception in _predict_merge_outcome for {owner}/{repo}#{pr_number}: {error_msg}"
            )

            # Look for specific DCO-related errors in the GitHub API response
            # DCO errors typically come as 422 validation errors with specific messages
            is_dco_error = False
            if "422" in error_msg and (
                "commit signoff required" in error_msg.lower()
                or "commits must have verified signatures" in error_msg.lower()
                or (
                    "dco" in error_msg.lower()
                    and ("required" in error_msg.lower() or "sign" in error_msg.lower())
                )
            ):
                is_dco_error = True
            elif "commit signoff required" in error_msg.lower():
                # Catch DCO errors that don't include status codes
                is_dco_error = True

            if is_dco_error:
                # This error comes from GitHub API, not our code - but these PRs are actually mergeable
                # The DCO requirement doesn't apply to API merges, only web-based commits
                self.log.info(
                    f"Ignoring DCO-related error for {owner}/{repo}#{pr_number} - API merges are allowed"
                )
                return True, "DCO enforcement not applicable to API merges"

            if (
                "protected ref" in error_msg.lower()
                or "cannot update" in error_msg.lower()
            ):
                if self.force_level in ["code-owners", "protection-rules", "all"]:
                    self.log.info(
                        f"Force level '{self.force_level}' bypassing protected ref error for {owner}/{repo}#{pr_number}"
                    )
                    return True, "protected ref error bypassed by force level"
                return (
                    False,
                    "cannot update protected ref - organization or branch protection rules prevent merge",
                )
            elif "403" in error_msg:
                if self.force_level == "all":
                    self.log.info(
                        f"Force level 'all' attempting to bypass permissions error for {owner}/{repo}#{pr_number}"
                    )
                    return True, "permissions error bypassed by force level"
                return (
                    False,
                    "insufficient permissions or branch protection rules prevent merge",
                )
            elif "405" in error_msg:
                return False, "merge method not allowed by repository settings"
            else:
                # Unknown error during test - assume it's mergeable
                self.log.debug(f"Test merge capability failed with unknown error: {e}")
                return True, "test merge capability failed - assuming mergeable"

    def get_results_summary(self) -> dict[str, Any]:
        """
        Get a summary of merge results.

        Returns:
            Dictionary with merge statistics
        """
        if not self._results:
            return {
                "total": 0,
                "merged": 0,
                "auto_merge_pending": 0,
                "failed": 0,
                "skipped": 0,
                "success_rate": 0.0,
                "average_duration": 0.0,
            }

        total = len(self._results)
        merged = sum(1 for r in self._results if r.status == MergeStatus.MERGED)
        auto_merge_pending = sum(
            1 for r in self._results if r.status == MergeStatus.AUTO_MERGE_PENDING
        )
        failed = sum(1 for r in self._results if r.status == MergeStatus.FAILED)
        skipped = sum(1 for r in self._results if r.status == MergeStatus.SKIPPED)

        success_rate = (merged / total) * 100 if total > 0 else 0.0
        average_duration = (
            sum(r.duration for r in self._results) / total if total > 0 else 0.0
        )

        return {
            "total": total,
            "merged": merged,
            "auto_merge_pending": auto_merge_pending,
            "failed": failed,
            "skipped": skipped,
            "success_rate": success_rate,
            "average_duration": average_duration,
            "results": self._results,
        }

    def get_failed_prs(self) -> list[MergeResult]:
        """
        Get list of failed merge results.

        Returns:
            List of MergeResult objects that failed
        """
        return [r for r in self._results if r.status == MergeStatus.FAILED]

    def get_successful_prs(self) -> list[MergeResult]:
        """
        Get list of successful or auto-merge-pending results.

        "Successful" here covers both PRs that were merged directly
        (``MergeStatus.MERGED``) and PRs where GitHub auto-merge was
        enabled and the PR is expected to merge once all required
        checks pass (``MergeStatus.AUTO_MERGE_PENDING``).

        Returns:
            List of MergeResult objects that were merged successfully
            or have auto-merge pending.
        """
        return [
            r
            for r in self._results
            if r.status in (MergeStatus.MERGED, MergeStatus.AUTO_MERGE_PENDING)
        ]
