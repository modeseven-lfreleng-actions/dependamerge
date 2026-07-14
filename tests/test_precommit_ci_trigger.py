# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

"""
Unit tests for the pre-commit.ci retrigger logic in AsyncMergeManager.

Covers:
- Required check present + status missing -> posts comment
- Status already reported -> no comment
- Preview mode -> no comment (side-effect guard)
- Duplicate trigger comment already exists -> no new comment
- Required check not configured -> no comment
- Polling success / failure / timeout paths (with sleep mocked)
- No GitHub client -> early return
- Ruleset branch filtering (_ruleset_applies_to_branch)
"""

from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from dependamerge.github2gerrit_detector import GitHub2GerritDetectionResult
from dependamerge.github_async import GitHubAsync
from dependamerge.merge_manager import MergeStatus
from dependamerge.models import PullRequestInfo


def _make_pr_info(**overrides):
    """Helper to build a PullRequestInfo with sensible defaults."""
    defaults = {
        "number": 42,
        "title": "Bump foo from 1.0 to 2.0",
        "body": "Dependabot PR",
        "author": "dependabot[bot]",
        "head_sha": "abc123def456",
        "base_branch": "main",
        "head_branch": "dependabot/pip/foo-2.0",
        "state": "open",
        "mergeable": True,
        "mergeable_state": "blocked",
        "behind_by": 0,
        "files_changed": [],
        "repository_full_name": "owner/repo",
        "html_url": "https://github.com/owner/repo/pull/42",
    }
    defaults.update(overrides)
    return PullRequestInfo(**defaults)


def _make_manager(**overrides):
    """Build an AsyncMergeManager with a mocked GitHub client.

    Returns ``(manager, client)`` — see ``tests/conftest.py`` for the
    typed-mock-client pattern and rationale.  Use ``client`` (not
    ``mgr._github_client``) for all mock setup and assertions.
    """
    # Typed mock client pattern — see tests/conftest.py
    from tests.conftest import make_merge_manager

    defaults: dict[str, Any] = {"preview_mode": False}
    defaults.update(overrides)
    return make_merge_manager(**defaults)


# ---------------------------------------------------------------------------
# 1. Required check present + status missing -> posts comment
# ---------------------------------------------------------------------------
class TestTriggerPostsComment:
    """When pre-commit.ci is required but has never reported status."""

    @pytest.mark.asyncio
    async def test_posts_trigger_comment_when_status_missing(self):
        mgr, client = _make_manager()  # typed mock client pattern (see conftest.py)
        pr = _make_pr_info()

        # Required checks include pre-commit.ci
        client.get_required_status_checks = AsyncMock(
            return_value=[{"context": "pre-commit.ci - pr"}]
        )

        # No statuses reported at all
        client.get.side_effect = [
            # commit status endpoint — no statuses
            {"statuses": []},
            # issue comments endpoint — no existing trigger comments
            [],
        ]

        client.post_issue_comment = AsyncMock()

        # Mock sleep so the poll loop doesn't actually wait
        with patch("asyncio.sleep", new_callable=AsyncMock):
            # After posting, first poll returns success
            success_status = {
                "statuses": [{"context": "pre-commit.ci - pr", "state": "success"}]
            }
            # Append the poll response after the initial two get calls
            client.get.side_effect = [
                # 1st call: commit status check (step 2)
                {"statuses": []},
                # 2nd call: issue comments (step 3 - duplicate check)
                [],
                # 3rd call: first poll iteration
                success_status,
            ]

            result = await mgr._trigger_stale_precommit_ci(pr)

        assert result is True
        client.post_issue_comment.assert_called_once_with(
            "owner", "repo", 42, "pre-commit.ci run"
        )


# ---------------------------------------------------------------------------
# 2. Status already reported -> no comment
# ---------------------------------------------------------------------------
class TestStatusAlreadyReported:
    """When pre-commit.ci has already reported a status (any state)."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("state", ["success", "pending", "failure", "error"])
    async def test_no_comment_when_status_exists(self, state):
        mgr, client = _make_manager()  # typed mock client pattern (see conftest.py)
        pr = _make_pr_info()

        client.get_required_status_checks = AsyncMock(
            return_value=[{"context": "pre-commit.ci - pr"}]
        )
        client.get = AsyncMock(
            return_value={
                "statuses": [{"context": "pre-commit.ci - pr", "state": state}]
            }
        )
        client.post_issue_comment = AsyncMock()

        result = await mgr._trigger_stale_precommit_ci(pr)

        assert result is False
        client.post_issue_comment.assert_not_called()


# ---------------------------------------------------------------------------
# 2b. Status pending past the stuck threshold -> retrigger
# ---------------------------------------------------------------------------
class TestStuckPendingRetrigger:
    """A pre-commit.ci status stuck in ``pending`` is retriggered."""

    @staticmethod
    def _iso(seconds_ago: float) -> str:
        ts = datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
        return ts.strftime("%Y-%m-%dT%H:%M:%SZ")

    @pytest.mark.asyncio
    async def test_retriggers_when_pending_past_threshold(self):
        """Pending longer than the stuck threshold posts a fresh trigger."""
        mgr, client = _make_manager()
        pr = _make_pr_info()

        client.get_required_status_checks = AsyncMock(
            return_value=[{"context": "pre-commit.ci - pr"}]
        )
        stuck = {
            "statuses": [
                {
                    "context": "pre-commit.ci - pr",
                    "state": "pending",
                    "updated_at": self._iso(600),
                }
            ]
        }
        success = {"statuses": [{"context": "pre-commit.ci - pr", "state": "success"}]}
        client.get.side_effect = [
            stuck,  # step 2: status check (stuck pending)
            [],  # step 3: duplicate-comment check
            success,  # first poll iteration
        ]
        client.post_issue_comment = AsyncMock()

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await mgr._trigger_stale_precommit_ci(pr)

        assert result is True
        client.post_issue_comment.assert_called_once_with(
            "owner", "repo", 42, "pre-commit.ci run"
        )

    @pytest.mark.asyncio
    async def test_leaves_recent_pending_run_alone(self):
        """A pending run still within its normal window is not retriggered."""
        mgr, client = _make_manager()
        pr = _make_pr_info()

        client.get_required_status_checks = AsyncMock(
            return_value=[{"context": "pre-commit.ci - pr"}]
        )
        client.get = AsyncMock(
            return_value={
                "statuses": [
                    {
                        "context": "pre-commit.ci - pr",
                        "state": "pending",
                        "updated_at": self._iso(30),
                    }
                ]
            }
        )
        client.post_issue_comment = AsyncMock()

        result = await mgr._trigger_stale_precommit_ci(pr)

        assert result is False
        client.post_issue_comment.assert_not_called()

    @pytest.mark.asyncio
    async def test_naive_timestamp_does_not_crash(self):
        """A pending status with a tz-naive timestamp degrades, not crashes.

        A timestamp lacking tz info parses to a naive datetime that
        would raise ``TypeError`` when subtracted from the tz-aware
        ``now``; the detector must fail closed (return ``False``)
        rather than abort the merge run.
        """
        mgr, client = _make_manager()
        pr = _make_pr_info()

        client.get_required_status_checks = AsyncMock(
            return_value=[{"context": "pre-commit.ci - pr"}]
        )
        client.get = AsyncMock(
            return_value={
                "statuses": [
                    {
                        "context": "pre-commit.ci - pr",
                        "state": "pending",
                        # No trailing "Z"/offset -> a naive datetime.
                        "updated_at": "2026-06-08T16:00:00",
                    }
                ]
            }
        )
        client.post_issue_comment = AsyncMock()

        result = await mgr._trigger_stale_precommit_ci(pr)

        assert result is False
        client.post_issue_comment.assert_not_called()


# ---------------------------------------------------------------------------
# 3. Preview mode -> no comment (guarded at call-site, not inside method,
#    but we also test that the call-site guard works)
# ---------------------------------------------------------------------------
class TestPreviewModeGuard:
    """Preview mode must prevent any side effects."""

    @pytest.mark.asyncio
    async def test_preview_mode_skips_trigger_via_merge_single_pr(self):
        """_merge_single_pr must not trigger side effects when preview_mode is True."""
        mgr, client = _make_manager(
            preview_mode=True
        )  # typed mock client pattern (see conftest.py)
        pr = _make_pr_info()

        # Sanity check that we're actually in preview mode.
        assert mgr.preview_mode is True

        # Patch the side-effecting method so we can assert it is not called.
        mgr._trigger_stale_precommit_ci = AsyncMock()
        client.post_issue_comment = AsyncMock()

        # Mock _check_merge_requirements to avoid unawaited-coroutine warnings
        # from the AsyncMock client (the real method would call async methods on
        # the mock whose return-value coroutines are never awaited).
        mgr._check_merge_requirements = AsyncMock(
            return_value=(True, "mocked for test")
        )

        # Execute the merge flow; preview_mode should prevent side effects.
        # _merge_single_pr will proceed through the flow and eventually
        # reach the pre-commit.ci block, which should be guarded.
        await mgr._merge_single_pr(pr)

        # In preview mode, neither the retrigger logic nor comment posting
        # should be invoked.
        mgr._trigger_stale_precommit_ci.assert_not_awaited()
        client.post_issue_comment.assert_not_awaited()


# ---------------------------------------------------------------------------
# 4. Duplicate trigger comment already exists -> no new comment
# ---------------------------------------------------------------------------
class TestDuplicateCommentGuard:
    """Avoid posting duplicate 'pre-commit.ci run' comments."""

    @pytest.mark.asyncio
    async def test_skips_when_trigger_comment_already_exists(self):
        mgr, client = _make_manager()  # typed mock client pattern (see conftest.py)
        pr = _make_pr_info()

        client.get_required_status_checks = AsyncMock(
            return_value=[{"context": "pre-commit.ci - pr"}]
        )
        client.get.side_effect = [
            # commit status — missing
            {"statuses": []},
            # existing issue comments — already has a trigger
            [{"body": "pre-commit.ci run"}],
        ]
        client.post_issue_comment = AsyncMock()

        result = await mgr._trigger_stale_precommit_ci(pr)

        assert result is False
        client.post_issue_comment.assert_not_called()

    @pytest.mark.asyncio
    async def test_posts_when_only_unrelated_comments_exist(self):
        mgr, client = _make_manager()  # typed mock client pattern (see conftest.py)
        pr = _make_pr_info()

        client.get_required_status_checks = AsyncMock(
            return_value=[{"context": "pre-commit.ci - pr"}]
        )
        client.get.side_effect = [
            {"statuses": []},
            # Unrelated comments — no trigger
            [{"body": "LGTM"}, {"body": "Please review"}],
            # Poll returns success
            {"statuses": [{"context": "pre-commit.ci - pr", "state": "success"}]},
        ]
        client.post_issue_comment = AsyncMock()

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await mgr._trigger_stale_precommit_ci(pr)

        assert result is True
        client.post_issue_comment.assert_called_once()


# ---------------------------------------------------------------------------
# 5. Required check not configured -> no comment
# ---------------------------------------------------------------------------
class TestRequiredCheckNotConfigured:
    """When pre-commit.ci is not a required status check."""

    @pytest.mark.asyncio
    async def test_no_comment_when_not_required(self):
        mgr, client = _make_manager()  # typed mock client pattern (see conftest.py)
        pr = _make_pr_info()

        client.get_required_status_checks = AsyncMock(
            return_value=[{"context": "ci/build"}, {"context": "ci/lint"}]
        )
        client.post_issue_comment = AsyncMock()

        result = await mgr._trigger_stale_precommit_ci(pr)

        assert result is False
        client.post_issue_comment.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_comment_when_no_required_checks(self):
        mgr, client = _make_manager()  # typed mock client pattern (see conftest.py)
        pr = _make_pr_info()

        client.get_required_status_checks = AsyncMock(return_value=[])
        client.post_issue_comment = AsyncMock()

        result = await mgr._trigger_stale_precommit_ci(pr)

        assert result is False
        client.post_issue_comment.assert_not_called()


# ---------------------------------------------------------------------------
# 6. Polling: success, failure, and timeout paths
# ---------------------------------------------------------------------------
class TestPollingBehavior:
    """Test the status-polling loop after posting the trigger comment."""

    @pytest.mark.asyncio
    async def test_polling_returns_true_on_success(self):
        mgr, client = _make_manager()  # typed mock client pattern (see conftest.py)
        pr = _make_pr_info()

        client.get_required_status_checks = AsyncMock(
            return_value=[{"context": "pre-commit.ci - pr"}]
        )
        client.post_issue_comment = AsyncMock()

        # Build side effects: status missing, no comments, then 2 pending polls, then success
        pending = {"statuses": [{"context": "pre-commit.ci - pr", "state": "pending"}]}
        success = {"statuses": [{"context": "pre-commit.ci - pr", "state": "success"}]}

        client.get.side_effect = [
            {"statuses": []},  # step 2: status check
            [],  # step 3: duplicate comment check
            pending,  # poll 1
            pending,  # poll 2
            success,  # poll 3
        ]

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await mgr._trigger_stale_precommit_ci(pr)

        assert result is True
        # sleep should have been called for each poll iteration
        assert mock_sleep.call_count == 3

    @pytest.mark.asyncio
    async def test_polling_returns_false_on_failure(self):
        mgr, client = _make_manager()  # typed mock client pattern (see conftest.py)
        pr = _make_pr_info()

        client.get_required_status_checks = AsyncMock(
            return_value=[{"context": "pre-commit.ci - pr"}]
        )
        client.post_issue_comment = AsyncMock()

        failure = {"statuses": [{"context": "pre-commit.ci - pr", "state": "failure"}]}
        client.get.side_effect = [
            {"statuses": []},  # step 2
            [],  # step 3
            failure,  # poll 1: immediate failure
        ]

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await mgr._trigger_stale_precommit_ci(pr)

        assert result is False

    @pytest.mark.asyncio
    async def test_polling_returns_false_on_error_state(self):
        mgr, client = _make_manager()  # typed mock client pattern (see conftest.py)
        pr = _make_pr_info()

        client.get_required_status_checks = AsyncMock(
            return_value=[{"context": "pre-commit.ci - pr"}]
        )
        client.post_issue_comment = AsyncMock()

        error_status = {
            "statuses": [{"context": "pre-commit.ci - pr", "state": "error"}]
        }
        client.get.side_effect = [
            {"statuses": []},
            [],
            error_status,
        ]

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await mgr._trigger_stale_precommit_ci(pr)

        assert result is False

    @pytest.mark.asyncio
    async def test_polling_timeout(self):
        """After max_polls iterations with only pending, returns False."""
        mgr, client = _make_manager()  # typed mock client pattern (see conftest.py)
        pr = _make_pr_info()

        client.get_required_status_checks = AsyncMock(
            return_value=[{"context": "pre-commit.ci - pr"}]
        )
        client.post_issue_comment = AsyncMock()

        pending = {"statuses": [{"context": "pre-commit.ci - pr", "state": "pending"}]}

        # step 2 + step 3 + 30 poll iterations (max_polls = 300s / 10s = 30)
        client.get.side_effect = [
            {"statuses": []},
            [],
        ] + [pending] * 30

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await mgr._trigger_stale_precommit_ci(pr)

        assert result is False
        # Should have slept once per poll
        assert mock_sleep.call_count == 30

    @pytest.mark.asyncio
    async def test_polling_handles_api_errors_gracefully(self):
        """API errors during polling should not crash — polling continues."""
        mgr, client = _make_manager()  # typed mock client pattern (see conftest.py)
        pr = _make_pr_info()

        client.get_required_status_checks = AsyncMock(
            return_value=[{"context": "pre-commit.ci - pr"}]
        )
        client.post_issue_comment = AsyncMock()

        success = {"statuses": [{"context": "pre-commit.ci - pr", "state": "success"}]}
        client.get.side_effect = [
            {"statuses": []},  # step 2
            [],  # step 3
            Exception("API error"),  # poll 1: transient error
            success,  # poll 2: recovered
        ]

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await mgr._trigger_stale_precommit_ci(pr)

        assert result is True


# ---------------------------------------------------------------------------
# 7. No GitHub client -> early return
# ---------------------------------------------------------------------------
class TestNoGitHubClient:
    """When _github_client is None, method returns False immediately."""

    @pytest.mark.asyncio
    async def test_returns_false_without_client(self):
        mgr, _client = _make_manager()  # typed mock client pattern (see conftest.py)
        mgr._github_client = None  # intentionally set to None for this test
        pr = _make_pr_info()

        result = await mgr._trigger_stale_precommit_ci(pr)
        assert result is False


# ---------------------------------------------------------------------------
# 8. Post comment failure -> returns False
# ---------------------------------------------------------------------------
class TestPostCommentFailure:
    """When posting the trigger comment fails."""

    @pytest.mark.asyncio
    async def test_returns_false_on_post_failure(self):
        mgr, client = _make_manager()  # typed mock client pattern (see conftest.py)
        pr = _make_pr_info()

        client.get_required_status_checks = AsyncMock(
            return_value=[{"context": "pre-commit.ci - pr"}]
        )
        client.get.side_effect = [
            {"statuses": []},
            [],  # no existing comments
        ]
        client.post_issue_comment = AsyncMock(
            side_effect=Exception("Permission denied")
        )

        result = await mgr._trigger_stale_precommit_ci(pr)

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_when_required_checks_api_fails(self):
        mgr, client = _make_manager()  # typed mock client pattern (see conftest.py)
        pr = _make_pr_info()

        client.get_required_status_checks = AsyncMock(
            side_effect=Exception("API unavailable")
        )
        client.post_issue_comment = AsyncMock()

        result = await mgr._trigger_stale_precommit_ci(pr)

        assert result is False
        client.post_issue_comment.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_false_when_status_api_fails(self):
        mgr, client = _make_manager()  # typed mock client pattern (see conftest.py)
        pr = _make_pr_info()

        client.get_required_status_checks = AsyncMock(
            return_value=[{"context": "pre-commit.ci - pr"}]
        )
        client.get = AsyncMock(side_effect=Exception("Network error"))
        client.post_issue_comment = AsyncMock()

        result = await mgr._trigger_stale_precommit_ci(pr)

        assert result is False
        client.post_issue_comment.assert_not_called()


# ---------------------------------------------------------------------------
# 9. Ruleset branch filtering (_ruleset_applies_to_branch)
# ---------------------------------------------------------------------------
class TestRulesetAppliesToBranch:
    """Unit tests for the static _ruleset_applies_to_branch helper."""

    _method = staticmethod(GitHubAsync._ruleset_applies_to_branch)

    def test_empty_conditions_returns_true(self):
        """No conditions → assume the ruleset applies (conservative)."""
        assert self._method({}, "main") is True

    def test_missing_ref_name_returns_true(self):
        """conditions dict without ref_name → assume applies."""
        assert self._method({"other_key": 123}, "main") is True

    def test_ref_name_not_dict_returns_true(self):
        """Non-dict ref_name → treat as no conditions."""
        assert self._method({"ref_name": "not-a-dict"}, "main") is True

    def test_tilde_all_matches_any_branch(self):
        assert self._method({"ref_name": {"include": ["~ALL"]}}, "main") is True
        assert self._method({"ref_name": {"include": ["~ALL"]}}, "develop") is True
        assert self._method({"ref_name": {"include": ["~ALL"]}}, "feature/foo") is True

    def test_tilde_default_branch_no_default_conservatively_matches(self):
        """When default_branch is None, ~DEFAULT_BRANCH matches conservatively."""
        cond = {"ref_name": {"include": ["~DEFAULT_BRANCH"]}}
        assert self._method(cond, "main") is True
        assert self._method(cond, "develop") is True
        assert self._method(cond, "anything") is True

    def test_tilde_default_branch_matches_explicit_default(self):
        """When default_branch is provided, ~DEFAULT_BRANCH matches only that branch."""
        cond = {"ref_name": {"include": ["~DEFAULT_BRANCH"]}}
        assert self._method(cond, "main", default_branch="main") is True
        assert self._method(cond, "master", default_branch="master") is True
        assert self._method(cond, "develop", default_branch="develop") is True

    def test_tilde_default_branch_does_not_match_non_default(self):
        """When default_branch is provided, other branches do not match."""
        cond = {"ref_name": {"include": ["~DEFAULT_BRANCH"]}}
        assert self._method(cond, "develop", default_branch="main") is False
        assert self._method(cond, "main", default_branch="develop") is False

    def test_exact_ref_match(self):
        cond = {"ref_name": {"include": ["refs/heads/release"]}}
        assert self._method(cond, "release") is True
        assert self._method(cond, "main") is False

    def test_bare_branch_name_normalised(self):
        """A bare branch name (no refs/heads/ prefix) should still match."""
        cond = {"ref_name": {"include": ["release"]}}
        assert self._method(cond, "release") is True
        assert self._method(cond, "main") is False

    def test_fnmatch_glob_pattern(self):
        cond = {"ref_name": {"include": ["refs/heads/release/*"]}}
        assert self._method(cond, "release/v1") is True
        assert self._method(cond, "release/v2.0") is True
        assert self._method(cond, "main") is False

    def test_exclude_overrides_include(self):
        cond = {"ref_name": {"include": ["~ALL"], "exclude": ["refs/heads/develop"]}}
        assert self._method(cond, "main") is True
        assert self._method(cond, "develop") is False

    def test_exclude_with_glob(self):
        cond = {"ref_name": {"include": ["~ALL"], "exclude": ["refs/heads/feature/*"]}}
        assert self._method(cond, "main") is True
        assert self._method(cond, "feature/foo") is False

    def test_no_include_patterns_returns_true(self):
        """Empty include list → no constraint → applies to all branches."""
        cond = {"ref_name": {"include": [], "exclude": []}}
        assert self._method(cond, "main") is True

    def test_multiple_include_patterns(self):
        cond = {"ref_name": {"include": ["refs/heads/main", "refs/heads/release/*"]}}
        assert self._method(cond, "main") is True
        assert self._method(cond, "release/v1") is True
        assert self._method(cond, "develop") is False

    def test_default_branch_passed_through_to_exclude(self):
        """Exclude with ~DEFAULT_BRANCH respects the explicit default_branch."""
        cond = {
            "ref_name": {
                "include": ["~ALL"],
                "exclude": ["~DEFAULT_BRANCH"],
            }
        }
        assert self._method(cond, "main", default_branch="main") is False
        assert self._method(cond, "feature/x", default_branch="main") is True


# ---------------------------------------------------------------------------
# 9. Post-wait staleness re-check in _merge_single_pr (Step 5.5)
# ---------------------------------------------------------------------------
class TestPostWaitRetrigger:
    """Step 5.5 must re-check pre-commit.ci staleness after its wait.

    Step 0.5 only retriggers a run that was already stale when
    processing started; a run that went pending shortly before the run
    began crosses the stuck threshold *during* the Step 5.5 wait.
    These tests drive ``_merge_single_pr`` with the wait mocked to
    expire while the PR is still blocked and assert the retrigger is
    invoked a second time (post-wait) and the flow routes correctly
    afterwards.
    """

    @staticmethod
    def _step5_5_patches(mgr, trigger_mock, merge_retry_mock):
        """Common patch set to reach Step 5.5 with a blocked PR."""
        no_g2g = GitHub2GerritDetectionResult()
        return (
            patch.object(
                mgr,
                "_detect_github2gerrit",
                new_callable=AsyncMock,
                return_value=no_g2g,
            ),
            patch.object(
                mgr,
                "_get_merge_method_for_repo",
                new_callable=AsyncMock,
                return_value="merge",
            ),
            patch.object(mgr, "_trigger_stale_precommit_ci", trigger_mock),
            patch.object(
                mgr,
                "_check_merge_requirements",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ),
            patch.object(
                mgr,
                "_blocked_pr_needs_rebase",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch.object(
                mgr,
                "_approve_pr",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                mgr,
                "_wait_for_auto_merge",
                new_callable=AsyncMock,
                return_value=(False, False),
            ),
            patch.object(mgr, "_merge_pr_with_retry", merge_retry_mock),
        )

    def _make_blocked_pr(self):
        return _make_pr_info(
            node_id="PR_kwDOTestNode42",
            mergeable_state="blocked",
            mergeable=True,
            state="open",
        )

    @pytest.mark.asyncio
    async def test_retrigger_fires_after_wait_and_detects_auto_merge(self):
        """Post-wait retrigger succeeds; auto-merge landed the PR."""
        mgr, client = _make_manager()
        pr = self._make_blocked_pr()

        client.enable_auto_merge = AsyncMock(return_value=True)
        client.analyze_block_reason = AsyncMock(
            return_value="Blocked by pending required check: pre-commit.ci - pr"
        )
        # Post-retrigger refresh: auto-merge fired the moment the
        # check landed, so the PR is already closed and merged.
        client.get = AsyncMock(return_value={"state": "closed", "merged": True})

        # Step 0.5 finds the run not yet stale (False); the post-wait
        # re-check finds it stuck and recovers it (True).
        trigger = AsyncMock(side_effect=[False, True])
        merge_retry = AsyncMock(return_value=True)

        patches = self._step5_5_patches(mgr, trigger, merge_retry)
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            result = await mgr._merge_single_pr(pr)

        assert trigger.await_count == 2
        assert result.status == MergeStatus.MERGED
        merge_retry.assert_not_called()

    @pytest.mark.asyncio
    async def test_retrigger_failure_still_routes_to_auto_merge_pending(self):
        """Retrigger cannot recover; flow falls through to Step 6."""
        mgr, client = _make_manager()
        pr = self._make_blocked_pr()

        client.enable_auto_merge = AsyncMock(return_value=True)
        client.analyze_block_reason = AsyncMock(
            return_value="Blocked by pending required check: pre-commit.ci - pr"
        )
        client.get = AsyncMock(return_value={})

        trigger = AsyncMock(return_value=False)
        merge_retry = AsyncMock(return_value=True)

        patches = self._step5_5_patches(mgr, trigger, merge_retry)
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            result = await mgr._merge_single_pr(pr)

        # Called at Step 0.5 AND again after the wait expired.
        assert trigger.await_count == 2
        # Auto-merge is armed and the block reason is pending checks,
        # so the run defers to auto-merge rather than failing.
        assert result.status == MergeStatus.AUTO_MERGE_PENDING
        merge_retry.assert_not_called()

    @pytest.mark.asyncio
    async def test_retrigger_success_with_open_pr_proceeds_to_merge(self):
        """Retrigger succeeds and the PR is now clean: manual merge runs."""
        mgr, client = _make_manager()
        pr = self._make_blocked_pr()

        client.enable_auto_merge = AsyncMock(return_value=True)
        client.analyze_block_reason = AsyncMock(
            return_value="Blocked by pending required check: pre-commit.ci - pr"
        )
        # Post-retrigger refresh: still open, now clean.
        client.get = AsyncMock(
            return_value={
                "state": "open",
                "merged": False,
                "mergeable": True,
                "mergeable_state": "clean",
                "head": {"sha": "def789refresh"},
            }
        )

        trigger = AsyncMock(side_effect=[False, True])
        merge_retry = AsyncMock(return_value=True)

        patches = self._step5_5_patches(mgr, trigger, merge_retry)
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            result = await mgr._merge_single_pr(pr)

        assert trigger.await_count == 2
        # The refresh updated the snapshot from the live payload.
        assert pr.mergeable_state == "clean"
        assert pr.head_sha == "def789refresh"
        # Clean state bypasses the auto-merge skip gate: manual merge.
        merge_retry.assert_called_once()
        assert result.status == MergeStatus.MERGED
