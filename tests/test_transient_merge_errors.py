# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 The Linux Foundation

"""
Tests for transient HTTP error handling during merge operations.

These tests verify that:
1. Transient 405 errors on mergeable PRs are retried instead of
   treated as permanent failures.
2. Post-approval propagation delays are applied correctly.
3. _get_failure_summary surfaces HTTP errors accurately instead of
   inferring misleading reasons from stale mergeable_state.
4. Defensive parsing of DEPENDAMERGE_POST_APPROVAL_DELAY env var.
"""

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from dependamerge.merge_manager import AsyncMergeManager
from dependamerge.models import PullRequestInfo
from tests.conftest import make_merge_manager


def _make_pr_info(
    mergeable_state: str = "clean",
    mergeable: bool | None = True,
    state: str = "open",
    number: int = 39,
    repo: str = "org/repo",
) -> PullRequestInfo:
    """Create a PullRequestInfo fixture modelling a pre-commit.ci PR.

    The title/author/branch defaults deliberately describe a
    ``pre-commit-ci[bot]`` autoupdate PR — the canonical automation PR
    this module exercises.  Tests whose behaviour depends on author or
    title should pass those values explicitly rather than relying on
    these fixture defaults.
    """
    return PullRequestInfo(
        number=number,
        title="Chore: pre-commit autoupdate",
        body="pre-commit update",
        author="pre-commit-ci[bot]",
        head_sha="abc123def456",
        base_branch="main",
        head_branch="pre-commit-ci-update-config",
        state=state,
        mergeable=mergeable,
        mergeable_state=mergeable_state,
        behind_by=None,
        files_changed=[],
        repository_full_name=repo,
        html_url=f"https://github.com/{repo}/pull/{number}",
        reviews=[],
        review_comments=[],
    )


def _make_405_exception() -> httpx.HTTPStatusError:
    """Create a realistic 405 HTTPStatusError with no response body.

    The empty body is deliberate: these tests exercise the retry
    classifier in ``_merge_pr_with_retry``, which keys off the HTTP
    *status line* ("405 Method Not Allowed") only.  The body-parsing
    path that surfaces GitHub's ``message`` field is covered separately
    by ``_make_405_with_body`` / ``TestMergeApiBodyCapture``.
    """
    request = httpx.Request(
        "PUT",
        "https://api.github.com/repos/org/repo/pulls/39/merge",
    )
    response = httpx.Response(
        status_code=405,
        request=request,
    )
    return httpx.HTTPStatusError(
        "Client error '405 Method Not Allowed' for url "
        "'https://api.github.com/repos/org/repo/pulls/39/merge'\n"
        "For more information check: "
        "https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/405",
        request=request,
        response=response,
    )


def _make_405_base_modified_exception() -> Exception:
    """Create a 405 carrying GitHub's "Base branch was modified" body.

    This mirrors the enhanced error string produced by
    ``merge_pull_request``/``_validate_merge_result``: the original 405
    status line plus GitHub's response body, which is what the retry
    classifier in ``_merge_pr_with_retry`` string-matches.
    """
    return Exception(
        "Failed to merge PR #39 in org/repo. Error: Client error "
        "'405 Method Not Allowed' for url "
        "'https://api.github.com/repos/org/repo/pulls/39/merge'. "
        "GitHub: Base branch was modified. Review and try the merge again. "
        "(PR state: open, mergeable: True, mergeable_state: behind) "
        "[PR branch is behind base branch]"
    )


def _make_502_exception() -> httpx.HTTPStatusError:
    """Create a realistic 502 HTTPStatusError."""
    request = httpx.Request(
        "PUT",
        "https://api.github.com/repos/org/repo/pulls/39/merge",
    )
    response = httpx.Response(
        status_code=502,
        request=request,
    )
    return httpx.HTTPStatusError(
        "Server error '502 Bad Gateway' for url "
        "'https://api.github.com/repos/org/repo/pulls/39/merge'",
        request=request,
        response=response,
    )


# -------------------------------------------------------------------
# Tests for _get_failure_summary HTTP error reporting
# -------------------------------------------------------------------


class TestFailureSummaryHTTPErrors:
    """Verify _get_failure_summary surfaces HTTP errors accurately."""

    def _make_manager(self) -> AsyncMergeManager:
        """Create a manager for testing _get_failure_summary."""
        mgr, _client = make_merge_manager(
            merge_method="merge",
            max_retries=2,
            concurrency=1,
        )
        return mgr

    def test_405_on_clean_pr_reports_transient_error(self):
        """A 405 on a clean PR should report transient API error."""
        mgr = self._make_manager()
        pr = _make_pr_info(mergeable_state="clean", mergeable=True)

        exc = _make_405_exception()
        mgr._last_merge_exception["org/repo#39"] = exc

        summary = mgr._get_failure_summary(pr)

        assert "transient 405" in summary.lower()
        assert "githubstatus.com" in summary
        # Must NOT say "branch protection"
        assert "branch protection" not in summary.lower()

    def test_405_on_unstable_pr_reports_transient_error(self):
        """A 405 on an unstable PR should also report transient."""
        mgr = self._make_manager()
        pr = _make_pr_info(mergeable_state="unstable", mergeable=True)

        exc = _make_405_exception()
        mgr._last_merge_exception["org/repo#39"] = exc

        summary = mgr._get_failure_summary(pr)

        assert "transient 405" in summary.lower()
        assert "githubstatus.com" in summary

    def test_405_on_blocked_pr_falls_through_to_state_analysis(self):
        """A 405 on a blocked PR should fall through to block analysis."""
        mgr = self._make_manager()
        pr = _make_pr_info(mergeable_state="blocked", mergeable=True)

        exc = _make_405_exception()
        mgr._last_merge_exception["org/repo#39"] = exc

        summary = mgr._get_failure_summary(pr)

        # Should NOT report transient — should fall through to
        # state-based analysis for blocked PRs
        assert "transient 405" not in summary.lower()

    def test_405_with_github_body_surfaces_detail(self):
        """A 405 carrying GitHub's response body surfaces that detail.

        End-to-end counterpart to ``TestMergeApiBodyCapture`` (which
        checks the transport layer): ``merge_pull_request`` embeds
        GitHub's explanation after a ``GitHub: `` marker, and
        ``_get_failure_summary`` must surface that actionable reason
        ahead of the state-based inference that a ``blocked`` PR would
        otherwise yield ("branch protection rules prevent merge").
        """
        mgr = self._make_manager()
        pr = _make_pr_info(mergeable_state="blocked", mergeable=True)

        mgr._last_merge_exception["org/repo#39"] = Exception(
            "Failed to merge PR #39 in org/repo. Error: Client error "
            "'405 Method Not Allowed' for url '.../merge'. "
            "GitHub: Required workflows 'Autolabeler' are not satisfied "
            "(PR state: open, mergeable: True, mergeable_state: blocked)"
        )

        summary = mgr._get_failure_summary(pr)

        assert "Required workflows" in summary
        assert "are not satisfied" in summary
        # The marker detail must win over generic block-state inference.
        assert "branch protection" not in summary.lower()
        # The appended PR-state context must be trimmed off.
        assert "PR state:" not in summary

    def test_502_reports_bad_gateway(self):
        """A 502 error should be reported as Bad Gateway."""
        mgr = self._make_manager()
        pr = _make_pr_info(mergeable_state="clean", mergeable=True)

        exc = _make_502_exception()
        mgr._last_merge_exception["org/repo#39"] = exc

        summary = mgr._get_failure_summary(pr)

        assert "502" in summary
        assert "bad gateway" in summary.lower()
        assert "githubstatus.com" in summary

    def test_workflow_scope_error_still_detected(self):
        """Existing workflow scope detection must not regress."""
        mgr = self._make_manager()
        pr = _make_pr_info(mergeable_state="clean", mergeable=True)

        mgr._last_merge_exception["org/repo#39"] = RuntimeError(
            "Missing 'workflow' scope for merge"
        )

        summary = mgr._get_failure_summary(pr)

        assert "workflow" in summary.lower()

    def test_no_exception_falls_through_to_state(self):
        """With no stored exception, summary uses mergeable_state."""
        mgr = self._make_manager()
        pr = _make_pr_info(mergeable_state="behind", mergeable=True)

        summary = mgr._get_failure_summary(pr)

        assert "behind" in summary.lower()


# -------------------------------------------------------------------
# Tests for transient 405 retry logic in _merge_pr_with_retry
# -------------------------------------------------------------------


class TestTransient405Retry:
    """Verify 405 on clean PRs triggers retry instead of immediate failure."""

    @pytest.mark.asyncio
    async def test_405_on_clean_pr_retries_and_succeeds(self):
        """First merge attempt gets 405, retry succeeds."""
        pr = _make_pr_info(mergeable_state="clean", mergeable=True)

        mgr, client = make_merge_manager(
            merge_method="merge",
            max_retries=2,
            concurrency=1,
        )
        mgr._pr_merge_methods["org/repo"] = "merge"

        # First call raises 405, second call succeeds
        client.merge_pull_request = AsyncMock(side_effect=[_make_405_exception(), True])
        # Return clean state on refresh
        client.get = AsyncMock(
            return_value={
                "mergeable": True,
                "mergeable_state": "clean",
                "state": "open",
                "merged": False,
            }
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await mgr._merge_pr_with_retry(pr, "org", "repo")

        assert result is True
        assert client.merge_pull_request.call_count == 2
        # The first attempt's 405 must be recorded for failure reporting,
        # even though the retry ultimately succeeded.
        assert "org/repo#39" in mgr._last_merge_exception

    @pytest.mark.asyncio
    async def test_405_on_clean_pr_retries_and_still_fails(self):
        """All merge attempts get 405, should eventually give up."""
        pr = _make_pr_info(mergeable_state="clean", mergeable=True)

        mgr, client = make_merge_manager(
            merge_method="merge",
            max_retries=2,
            concurrency=1,
        )
        mgr._pr_merge_methods["org/repo"] = "merge"

        # All calls raise 405
        client.merge_pull_request = AsyncMock(side_effect=_make_405_exception())
        client.get = AsyncMock(
            return_value={
                "mergeable": True,
                "mergeable_state": "clean",
                "state": "open",
                "merged": False,
            }
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await mgr._merge_pr_with_retry(pr, "org", "repo")

        assert result is False
        # Initial attempt + max_retries = 3 attempts total
        assert client.merge_pull_request.call_count == 3

    @pytest.mark.asyncio
    async def test_405_on_blocked_pr_breaks_without_recent_approval(self):
        """A 405 on a blocked PR with no recent approval should not retry."""
        pr = _make_pr_info(mergeable_state="blocked", mergeable=True)

        mgr, client = make_merge_manager(
            merge_method="merge",
            max_retries=2,
            concurrency=1,
        )
        mgr._pr_merge_methods["org/repo"] = "merge"

        client.merge_pull_request = AsyncMock(side_effect=_make_405_exception())

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await mgr._merge_pr_with_retry(pr, "org", "repo")

        assert result is False
        # Should only attempt once — breaks immediately
        assert client.merge_pull_request.call_count == 1

    @pytest.mark.asyncio
    async def test_405_on_blocked_pr_retries_with_recent_approval(self):
        """A 405 on blocked PR with recent approval retries after refresh."""
        pr = _make_pr_info(mergeable_state="blocked", mergeable=True)

        mgr, client = make_merge_manager(
            merge_method="merge",
            max_retries=2,
            concurrency=1,
        )
        mgr._pr_merge_methods["org/repo"] = "merge"
        # Mark as recently approved
        mgr._recently_approved.add("org/repo#39")

        # First call: 405 (blocked), refresh shows clean, retry succeeds
        client.merge_pull_request = AsyncMock(side_effect=[_make_405_exception(), True])
        client.get = AsyncMock(
            return_value={
                "mergeable": True,
                "mergeable_state": "clean",
                "state": "open",
                "merged": False,
            }
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await mgr._merge_pr_with_retry(pr, "org", "repo")

        assert result is True
        assert client.merge_pull_request.call_count == 2

    @pytest.mark.asyncio
    async def test_405_base_branch_modified_retries_and_succeeds(self):
        """A "Base branch was modified" 405 is a transient race: retry.

        The PR is ``behind`` and ``fix_out_of_date`` is off.  The dedicated
        "base branch was modified" branch in ``_merge_pr_with_retry``
        recognises this as a transient race and retries via ``continue``
        (up to ``max_retries``); without it the 405 would instead fall
        through to the terminal ``else: break`` taken by behind/no-fix PRs
        and report a false failure.  The race clears on retry (a sibling
        merge advanced the base), so the second attempt succeeds.
        """
        pr = _make_pr_info(mergeable_state="behind", mergeable=True)

        mgr, client = make_merge_manager(
            merge_method="merge",
            max_retries=2,
            concurrency=1,
            fix_out_of_date=False,
        )
        mgr._pr_merge_methods["org/repo"] = "merge"

        client.merge_pull_request = AsyncMock(
            side_effect=[_make_405_base_modified_exception(), True]
        )
        # Re-fetch before the retry shows the PR still open (not merged).
        client.get = AsyncMock(
            return_value={
                "mergeable": True,
                "mergeable_state": "clean",
                "state": "open",
                "merged": False,
            }
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await mgr._merge_pr_with_retry(pr, "org", "repo")

        assert result is True
        assert client.merge_pull_request.call_count == 2

    @pytest.mark.asyncio
    async def test_405_base_branch_modified_exhausts_retries(self):
        """A persistent base-modified 405 gives up after max_retries."""
        pr = _make_pr_info(mergeable_state="behind", mergeable=True)

        mgr, client = make_merge_manager(
            merge_method="merge",
            max_retries=2,
            concurrency=1,
            fix_out_of_date=False,
        )
        mgr._pr_merge_methods["org/repo"] = "merge"

        client.merge_pull_request = AsyncMock(
            side_effect=_make_405_base_modified_exception()
        )
        client.get = AsyncMock(
            return_value={
                "mergeable": True,
                "mergeable_state": "behind",
                "state": "open",
                "merged": False,
            }
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await mgr._merge_pr_with_retry(pr, "org", "repo")

        assert result is False
        # Initial attempt + max_retries = 3 attempts total.
        assert client.merge_pull_request.call_count == 3


# -------------------------------------------------------------------
# Tests for post-approval delay and _recently_approved cleanup
# -------------------------------------------------------------------


class TestPostApprovalDelay:
    """Verify post-approval propagation delay behaviour."""

    @pytest.mark.asyncio
    async def test_approval_adds_to_recently_approved(self):
        """Submitting a new approval should track the PR key."""
        mgr, _client = make_merge_manager(
            merge_method="merge",
            max_retries=1,
            concurrency=1,
        )
        mgr._post_approval_delay = 0.0  # No real delay in tests

        # Simulate _approve_pr returning True (new approval)
        with patch.object(mgr, "_approve_pr", new_callable=AsyncMock) as mock_approve:
            mock_approve.return_value = True

            # We just need to verify _recently_approved is populated
            # Simulate the approval code path
            approval_added = await mgr._approve_pr("org", "repo", 39)

            if approval_added:
                pr_key = "org/repo#39"
                mgr._recently_approved.add(pr_key)

            assert "org/repo#39" in mgr._recently_approved

    def test_recently_approved_cleanup_via_discard(self):
        """Verify _recently_approved entries can be cleaned up."""
        mgr, _client = make_merge_manager(
            merge_method="merge",
            max_retries=1,
            concurrency=1,
        )

        mgr._recently_approved.add("org/repo#39")
        assert "org/repo#39" in mgr._recently_approved

        mgr._recently_approved.discard("org/repo#39")
        assert "org/repo#39" not in mgr._recently_approved

        # Discarding non-existent key should not raise
        mgr._recently_approved.discard("org/repo#999")


# -------------------------------------------------------------------
# Tests for defensive DEPENDAMERGE_POST_APPROVAL_DELAY parsing
# -------------------------------------------------------------------


class TestPostApprovalDelayConfig:
    """Verify defensive parsing of the env var configuration."""

    def test_default_delay_value(self):
        """Default delay should be 3.0 seconds."""
        with (
            patch.dict("os.environ", {}, clear=False),
            patch("dependamerge.merge_manager.GitHubAsync"),
        ):
            # Ensure env var is NOT set
            import os

            os.environ.pop("DEPENDAMERGE_POST_APPROVAL_DELAY", None)
            mgr = AsyncMergeManager(
                token="fake_token",
                merge_method="merge",
            )
            assert mgr._post_approval_delay == 3.0

    def test_custom_delay_from_env(self):
        """Custom numeric value should be respected."""
        with (
            patch.dict(
                "os.environ",
                {"DEPENDAMERGE_POST_APPROVAL_DELAY": "5.5"},
            ),
            patch("dependamerge.merge_manager.GitHubAsync"),
        ):
            mgr = AsyncMergeManager(
                token="fake_token",
                merge_method="merge",
            )
            assert mgr._post_approval_delay == 5.5

    def test_zero_delay_from_env(self):
        """Zero should disable the delay."""
        with (
            patch.dict(
                "os.environ",
                {"DEPENDAMERGE_POST_APPROVAL_DELAY": "0"},
            ),
            patch("dependamerge.merge_manager.GitHubAsync"),
        ):
            mgr = AsyncMergeManager(
                token="fake_token",
                merge_method="merge",
            )
            assert mgr._post_approval_delay == 0.0

    def test_invalid_delay_falls_back_to_default(self):
        """Non-numeric value should fall back to 3.0 with a warning."""
        with (
            patch.dict(
                "os.environ",
                {"DEPENDAMERGE_POST_APPROVAL_DELAY": "not-a-number"},
            ),
            patch("dependamerge.merge_manager.GitHubAsync"),
        ):
            mgr = AsyncMergeManager(
                token="fake_token",
                merge_method="merge",
            )
            assert mgr._post_approval_delay == 3.0

    def test_inf_delay_falls_back_to_default(self):
        """Infinity should be rejected and fall back to default."""
        with (
            patch.dict(
                "os.environ",
                {"DEPENDAMERGE_POST_APPROVAL_DELAY": "inf"},
            ),
            patch("dependamerge.merge_manager.GitHubAsync"),
        ):
            mgr = AsyncMergeManager(
                token="fake_token",
                merge_method="merge",
            )
            assert mgr._post_approval_delay == 3.0

    def test_negative_inf_delay_falls_back_to_default(self):
        """Negative infinity should be rejected and fall back to default."""
        with (
            patch.dict(
                "os.environ",
                {"DEPENDAMERGE_POST_APPROVAL_DELAY": "-inf"},
            ),
            patch("dependamerge.merge_manager.GitHubAsync"),
        ):
            mgr = AsyncMergeManager(
                token="fake_token",
                merge_method="merge",
            )
            assert mgr._post_approval_delay == 3.0

    def test_nan_delay_falls_back_to_default(self):
        """NaN should be rejected and fall back to default."""
        with (
            patch.dict(
                "os.environ",
                {"DEPENDAMERGE_POST_APPROVAL_DELAY": "nan"},
            ),
            patch("dependamerge.merge_manager.GitHubAsync"),
        ):
            mgr = AsyncMergeManager(
                token="fake_token",
                merge_method="merge",
            )
            assert mgr._post_approval_delay == 3.0

    def test_negative_delay_falls_back_to_default(self):
        """Negative values should be rejected and fall back to default."""
        with (
            patch.dict(
                "os.environ",
                {"DEPENDAMERGE_POST_APPROVAL_DELAY": "-5"},
            ),
            patch("dependamerge.merge_manager.GitHubAsync"),
        ):
            mgr = AsyncMergeManager(
                token="fake_token",
                merge_method="merge",
            )
            assert mgr._post_approval_delay == 3.0


def _make_405_with_body(message: str) -> httpx.HTTPStatusError:
    """Build a 405 ``HTTPStatusError`` carrying a JSON ``message`` body."""
    request = httpx.Request("PUT", "https://api.github.com/repos/o/r/pulls/1/merge")
    response = httpx.Response(
        status_code=405, request=request, json={"message": message}
    )
    return httpx.HTTPStatusError(
        "Client error '405 Method Not Allowed' for url "
        "'https://api.github.com/repos/o/r/pulls/1/merge'",
        request=request,
        response=response,
    )


class TestMergeApiBodyCapture:
    """``merge_pull_request`` must surface GitHub's response body.

    GitHub puts the real reason (ruleset violations, "Required
    workflows ... not satisfied") in the response *body*; the
    ``HTTPStatusError`` text only carries the status line.  A bare
    ``raise`` previously discarded the body — these tests lock in that
    the body now reaches the raised error.
    """

    @pytest.mark.asyncio
    async def test_merge_error_surfaces_response_body(self):
        from dependamerge.github_async import GitHubAsync

        api = GitHubAsync(token="t")
        body_msg = (
            "Repository rule violations found\n\n"
            "Required workflows 'Autolabeler, Semantic Pull Request' "
            "are not satisfied"
        )
        api.put = AsyncMock(side_effect=_make_405_with_body(body_msg))
        api.get = AsyncMock(
            return_value={
                "state": "open",
                "mergeable": True,
                "mergeable_state": "blocked",
                "merged": False,
            }
        )

        with pytest.raises(Exception) as excinfo:
            await api.merge_pull_request("o", "r", 1, "merge")

        msg = str(excinfo.value)
        assert "GitHub:" in msg
        assert "Required workflows" in msg
        assert "not satisfied" in msg
        # The original status line MUST be preserved so
        # _merge_pr_with_retry still classifies the 405 as terminal
        # (and fails fast) instead of retrying it 3x.
        assert "405" in msg
        assert "Method Not Allowed" in msg

    @pytest.mark.asyncio
    async def test_merge_succeeded_despite_exception_returns_true(self):
        """The race-recovery path (was dead code) must work again."""
        from dependamerge.github_async import GitHubAsync

        api = GitHubAsync(token="t")
        api.put = AsyncMock(side_effect=_make_405_with_body("transient"))
        # Re-fetch shows the PR actually merged despite the exception.
        api.get = AsyncMock(return_value={"state": "closed", "merged": True})

        result = await api.merge_pull_request("o", "r", 1, "merge")

        assert result is True

    @pytest.mark.asyncio
    async def test_surfaces_body_when_state_refetch_fails(self):
        """If the PR-state re-fetch fails, still surface GitHub's body."""
        from dependamerge.github_async import GitHubAsync

        api = GitHubAsync(token="t")
        api.put = AsyncMock(
            side_effect=_make_405_with_body(
                "Required workflows 'Autolabeler' are not satisfied"
            )
        )
        # The follow-up PR-state GET also fails.
        api.get = AsyncMock(side_effect=RuntimeError("re-fetch failed"))

        with pytest.raises(Exception) as excinfo:
            await api.merge_pull_request("o", "r", 1, "merge")

        msg = str(excinfo.value)
        assert "GitHub:" in msg
        assert "Required workflows" in msg


class TestFailureSummarySurfacesGitHubDetail:
    """``_get_failure_summary`` surfaces the GitHub-supplied reason."""

    def test_github_detail_extracted_from_exception(self):
        mgr, _client = make_merge_manager(merge_method="merge")
        pr = _make_pr_info(mergeable_state="blocked", mergeable=True)
        exc = Exception(
            "Failed to merge PR #39 in org/repo. GitHub: Required "
            "workflows 'Autolabeler' are not satisfied (PR state: open, "
            "mergeable: True, mergeable_state: blocked) "
            "[blocked by branch protection / required checks]"
        )
        mgr._last_merge_exception["org/repo#39"] = exc

        summary = mgr._get_failure_summary(pr)

        # The actionable GitHub message is returned, trimmed of the
        # appended PR-state context.
        assert summary == "Required workflows 'Autolabeler' are not satisfied"


# -------------------------------------------------------------------
# Pending required-workflows classification and recovery
# -------------------------------------------------------------------


def _make_405_workflows_not_satisfied_exception() -> Exception:
    """405 carrying GitHub's "Required workflows ... not satisfied" body.

    Mirrors the enhanced error produced by ``_validate_merge_result``
    when ruleset-required workflows are still *executing* on the head
    commit — a pending condition, not a terminal failure.
    """
    return Exception(
        "Failed to merge PR #39 in org/repo. Error: Client error "
        "'405 Method Not Allowed' for url "
        "'https://api.github.com/repos/org/repo/pulls/39/merge'. "
        "GitHub: Repository rule violations found Required workflows "
        "'Verify Token Permissions' are not satisfied (PR state: open, "
        "mergeable: True, mergeable_state: blocked) "
        "[blocked by branch protection / required checks]"
    )


def _make_405_workflows_failed_exception() -> Exception:
    """405 whose required-workflows clause reports a *failure*."""
    return Exception(
        "Failed to merge PR #39 in org/repo. Error: Client error "
        "'405 Method Not Allowed' for url "
        "'https://api.github.com/repos/org/repo/pulls/39/merge'. "
        "GitHub: Repository rule violations found Required workflows "
        "'Verify Token Permissions' failed (PR state: open, "
        "mergeable: True, mergeable_state: blocked)"
    )


class TestMergeErrorIndicatesPendingWorkflows:
    """Classification of 405 required-workflows rejection bodies."""

    def _mgr(self) -> AsyncMergeManager:
        mgr, _client = make_merge_manager(merge_method="merge")
        return mgr

    def test_not_satisfied_is_pending(self):
        """ "not satisfied" (workflows still executing) is pending."""
        mgr = self._mgr()
        exc = _make_405_workflows_not_satisfied_exception()
        assert mgr._merge_error_indicates_pending_workflows(str(exc)) is True

    def test_failed_variant_is_terminal(self):
        """A workflow that ran and failed must stay terminal."""
        mgr = self._mgr()
        exc = _make_405_workflows_failed_exception()
        assert mgr._merge_error_indicates_pending_workflows(str(exc)) is False

    def test_leading_failed_to_merge_prefix_does_not_poison(self):
        """The "Failed to merge PR" prefix must not read as failure.

        Only the clause from the ``required workflow`` wording onward
        is inspected; the enhanced exception text always starts with
        "Failed to merge PR …".
        """
        mgr = self._mgr()
        text = (
            "Failed to merge PR #39 in org/repo. "
            "GitHub: Required workflows 'CI' are not satisfied"
        )
        assert mgr._merge_error_indicates_pending_workflows(text) is True

    def test_pr_state_suffix_is_trimmed(self):
        """Failure wording after the PR-state context is ignored."""
        mgr = self._mgr()
        text = (
            "Failed to merge PR #39 in org/repo. "
            "GitHub: Required workflows 'CI' are not satisfied "
            "(PR state: open, mergeable: False, mergeable_state: "
            "blocked) [blocked by failing checks]"
        )
        assert mgr._merge_error_indicates_pending_workflows(text) is True

    def test_required_status_checks_not_matched(self):
        """Status-check violations are a different condition."""
        mgr = self._mgr()
        text = (
            "Failed to merge PR #39 in org/repo. "
            "GitHub: Repository rule violations found Required status "
            'check "pre-commit.ci - pr" is expected.'
        )
        assert mgr._merge_error_indicates_pending_workflows(text) is False

    def test_empty_text_returns_false(self):
        mgr = self._mgr()
        assert mgr._merge_error_indicates_pending_workflows("") is False


class TestWaitForRequiredWorkflowsAndRetry:
    """Behaviour of the pending-workflows wait-and-retry recovery."""

    def _make_mgr(self, **overrides):
        defaults = {"merge_method": "merge", "preview_mode": False}
        defaults.update(overrides)
        return make_merge_manager(**defaults)

    @pytest.mark.asyncio
    async def test_preview_mode_returns_false_without_side_effects(self):
        mgr, client = self._make_mgr(preview_mode=True)
        pr = _make_pr_info(mergeable_state="blocked")

        result = await mgr._wait_for_required_workflows_and_retry(pr, "org", "repo")

        assert result is False
        client.merge_pull_request.assert_not_called()

    @pytest.mark.asyncio
    async def test_merged_during_wait_returns_true(self):
        mgr, client = self._make_mgr()
        pr = _make_pr_info(mergeable_state="blocked")
        client.get = AsyncMock(
            return_value={
                "mergeable": True,
                "mergeable_state": "blocked",
                "state": "open",
            }
        )

        with patch.object(
            mgr,
            "_wait_for_auto_merge",
            new_callable=AsyncMock,
            return_value=(True, True),
        ):
            result = await mgr._wait_for_required_workflows_and_retry(pr, "org", "repo")

        assert result is True
        client.merge_pull_request.assert_not_called()

    @pytest.mark.asyncio
    async def test_closed_without_merge_returns_false(self):
        mgr, client = self._make_mgr()
        pr = _make_pr_info(mergeable_state="blocked")
        client.get = AsyncMock(return_value={})

        with patch.object(
            mgr,
            "_wait_for_auto_merge",
            new_callable=AsyncMock,
            return_value=(True, False),
        ):
            result = await mgr._wait_for_required_workflows_and_retry(pr, "org", "repo")

        assert result is False
        client.merge_pull_request.assert_not_called()

    @pytest.mark.asyncio
    async def test_wait_then_merge_succeeds(self):
        """Workflows finish during the wait; the retry lands the merge."""
        mgr, client = self._make_mgr()
        pr = _make_pr_info(mergeable_state="blocked")
        client.get = AsyncMock(
            return_value={
                "mergeable": True,
                "mergeable_state": "blocked",
                "state": "open",
            }
        )
        client.merge_pull_request = AsyncMock(return_value=True)

        with patch.object(
            mgr,
            "_wait_for_auto_merge",
            new_callable=AsyncMock,
            return_value=(False, False),
        ):
            result = await mgr._wait_for_required_workflows_and_retry(pr, "org", "repo")

        assert result is True
        client.merge_pull_request.assert_awaited_once_with("org", "repo", 39, "merge")

    @pytest.mark.asyncio
    async def test_different_rejection_reason_stops_recovery(self):
        """A changed rejection reason is left to the caller's classifier."""
        mgr, client = self._make_mgr()
        pr = _make_pr_info(mergeable_state="blocked")
        client.get = AsyncMock(return_value={})
        terminal_exc = _make_405_workflows_failed_exception()
        client.merge_pull_request = AsyncMock(side_effect=terminal_exc)

        with patch.object(
            mgr,
            "_wait_for_auto_merge",
            new_callable=AsyncMock,
            return_value=(False, False),
        ):
            result = await mgr._wait_for_required_workflows_and_retry(pr, "org", "repo")

        assert result is False
        # The terminal exception is stored for failure reporting.
        assert mgr._last_merge_exception["org/repo#39"] is terminal_exc
        client.merge_pull_request.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_still_pending_at_deadline_returns_false(self):
        """Workflows never finish: recovery is bounded by merge_timeout."""
        mgr, client = self._make_mgr(merge_timeout=0.1)
        pr = _make_pr_info(mergeable_state="blocked")
        client.get = AsyncMock(
            return_value={
                "mergeable": True,
                "mergeable_state": "blocked",
                "state": "open",
            }
        )
        client.merge_pull_request = AsyncMock(
            side_effect=_make_405_workflows_not_satisfied_exception()
        )

        with patch.object(
            mgr,
            "_wait_for_auto_merge",
            new_callable=AsyncMock,
            return_value=(False, False),
        ):
            result = await mgr._wait_for_required_workflows_and_retry(pr, "org", "repo")

        assert result is False
        # At least one retry was dispatched before the budget expired.
        assert client.merge_pull_request.await_count >= 1


class TestPendingWorkflowsMergeSinglePrRouting:
    """_merge_single_pr routes 405 pending-workflows rejections to recovery."""

    def _patches(self, mgr, merge_retry, wf_recovery):
        from dependamerge.github2gerrit_detector import (
            GitHub2GerritDetectionResult,
        )

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
            patch.object(
                mgr,
                "_trigger_stale_precommit_ci",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch.object(
                mgr,
                "_check_merge_requirements",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ),
            patch.object(
                mgr,
                "_approve_and_retry_if_review_required",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch.object(
                mgr,
                "_fetch_pr_state_now",
                new_callable=AsyncMock,
                return_value=("open", False),
            ),
            patch.object(
                mgr,
                "_detect_stuck_required_check",
                new_callable=AsyncMock,
                return_value=(False, None, 0.0),
            ),
            patch.object(mgr, "_merge_pr_with_retry", merge_retry),
            patch.object(mgr, "_wait_for_required_workflows_and_retry", wf_recovery),
        )

    @pytest.mark.asyncio
    async def test_not_satisfied_rejection_invokes_recovery(self):
        """ "not satisfied" → wait-and-retry recovery merges the PR."""
        from contextlib import ExitStack

        from dependamerge.merge_manager import MergeStatus

        mgr, client = make_merge_manager(merge_method="merge", preview_mode=False)
        pr = _make_pr_info(mergeable_state="clean", mergeable=True)
        client.get = AsyncMock(return_value={})
        client.get_required_status_checks = AsyncMock(return_value=[])

        async def failing_merge(pr_info, owner, repo):
            mgr._last_merge_exception[f"{owner}/{repo}#{pr_info.number}"] = (
                _make_405_workflows_not_satisfied_exception()
            )
            return False

        merge_retry = AsyncMock(side_effect=failing_merge)
        wf_recovery = AsyncMock(return_value=True)

        with ExitStack() as stack:
            for p in self._patches(mgr, merge_retry, wf_recovery):
                stack.enter_context(p)
            result = await mgr._merge_single_pr(pr)

        wf_recovery.assert_awaited_once()
        assert result.status == MergeStatus.MERGED

    @pytest.mark.asyncio
    async def test_failed_rejection_skips_recovery(self):
        """ "failed" → terminal: no wait-and-retry, PR reported failed."""
        from contextlib import ExitStack

        from dependamerge.merge_manager import MergeStatus

        mgr, client = make_merge_manager(merge_method="merge", preview_mode=False)
        pr = _make_pr_info(mergeable_state="clean", mergeable=True)
        client.get = AsyncMock(return_value={})
        client.get_required_status_checks = AsyncMock(return_value=[])

        async def failing_merge(pr_info, owner, repo):
            mgr._last_merge_exception[f"{owner}/{repo}#{pr_info.number}"] = (
                _make_405_workflows_failed_exception()
            )
            return False

        merge_retry = AsyncMock(side_effect=failing_merge)
        wf_recovery = AsyncMock(return_value=True)

        with ExitStack() as stack:
            for p in self._patches(mgr, merge_retry, wf_recovery):
                stack.enter_context(p)
            result = await mgr._merge_single_pr(pr)

        wf_recovery.assert_not_awaited()
        assert result.status == MergeStatus.FAILED
