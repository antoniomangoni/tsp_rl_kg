from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterable
from typing import Any
from urllib import error, request

API_VERSION = "2022-11-28"
MAINTAINER_LOGIN = os.environ.get("MAINTAINER_LOGIN", "antoniomangoni")


def fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def load_event_payload() -> dict[str, Any]:
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not event_path:
        raise RuntimeError("GITHUB_EVENT_PATH is not set.")

    with open(event_path, encoding="utf-8") as event_file:
        payload = json.load(event_file)

    if not isinstance(payload, dict):
        raise RuntimeError("GitHub event payload must be a JSON object.")

    return payload


def get_pull_request(payload: dict[str, Any]) -> dict[str, Any]:
    pull_request = payload.get("pull_request")
    if not isinstance(pull_request, dict):
        raise RuntimeError("GitHub event payload does not contain a pull_request object.")

    return pull_request


def parse_next_link(link_header: str | None) -> str | None:
    if not link_header:
        return None

    for part in link_header.split(","):
        segment = part.strip()
        if 'rel="next"' not in segment:
            continue

        start = segment.find("<")
        end = segment.find(">", start + 1)
        if start != -1 and end != -1:
            return segment[start + 1 : end]

    return None


def fetch_reviews(pull_request_url: str, token: str) -> list[dict[str, Any]]:
    reviews: list[dict[str, Any]] = []
    next_url = f"{pull_request_url}/reviews?per_page=100"

    while next_url:
        api_request = request.Request(
            next_url,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "User-Agent": "tsp-rl-kg-pr-policy",
                "X-GitHub-Api-Version": API_VERSION,
            },
        )

        try:
            with request.urlopen(api_request) as response:
                body = json.load(response)
                link_header = response.headers.get("Link")
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"GitHub API request failed with status {exc.code}: {details}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(f"Unable to reach GitHub API: {exc.reason}") from exc

        if not isinstance(body, list):
            raise RuntimeError("GitHub API returned an unexpected reviews payload.")

        reviews.extend(review for review in body if isinstance(review, dict))
        next_url = parse_next_link(link_header)

    return reviews


def iter_maintainer_reviews(reviews: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for review in reviews:
        user = review.get("user")
        if not isinstance(user, dict):
            continue

        if user.get("login") == MAINTAINER_LOGIN:
            yield review


def get_effective_review_state(
    reviews: Iterable[dict[str, Any]],
) -> tuple[str | None, str | None]:
    state: str | None = None
    commit_id: str | None = None

    for review in iter_maintainer_reviews(reviews):
        review_state = str(review.get("state", "")).upper()
        if review_state not in {"APPROVED", "CHANGES_REQUESTED", "DISMISSED"}:
            continue

        state = review_state
        commit_id = review.get("commit_id")

    return state, commit_id


def main() -> int:
    try:
        payload = load_event_payload()
        pull_request = get_pull_request(payload)
    except RuntimeError as exc:
        return fail(str(exc))

    base = pull_request.get("base")
    if not isinstance(base, dict):
        return fail("Pull request payload is missing base branch information.")

    if base.get("ref") != "main":
        print("Pull request does not target main; policy check passes.")
        return 0

    author = (pull_request.get("user") or {}).get("login")
    if not author:
        return fail("Pull request author login is missing from the event payload.")

    if author == MAINTAINER_LOGIN:
        print(f"Pull request authored by {MAINTAINER_LOGIN}; policy check passes.")
        return 0

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        return fail("GITHUB_TOKEN is required to evaluate maintainer approval policy.")

    pull_request_url = pull_request.get("url")
    if not isinstance(pull_request_url, str) or not pull_request_url:
        return fail("Pull request API URL is missing from the event payload.")

    head = pull_request.get("head")
    if not isinstance(head, dict):
        return fail("Pull request head information is missing from the event payload.")

    head_sha = head.get("sha")
    if not isinstance(head_sha, str) or not head_sha:
        return fail("Pull request head SHA is missing from the event payload.")

    try:
        reviews = fetch_reviews(pull_request_url, token)
    except RuntimeError as exc:
        return fail(str(exc))

    state, commit_id = get_effective_review_state(reviews)

    if state == "APPROVED" and commit_id == head_sha:
        print(
            f"Contributor pull request by {author} has a current approval from {MAINTAINER_LOGIN}."
        )
        return 0

    if state == "APPROVED":
        return fail(
            f"Contributor pull request by {author} requires a fresh approval from "
            f"{MAINTAINER_LOGIN} for commit {head_sha}."
        )

    if state == "CHANGES_REQUESTED":
        return fail(
            f"Contributor pull request by {author} is blocked because {MAINTAINER_LOGIN} "
            "requested changes."
        )

    if state == "DISMISSED":
        return fail(
            f"Contributor pull request by {author} is blocked because the latest effective "
            f"review from {MAINTAINER_LOGIN} was dismissed."
        )

    return fail(
        f"Contributor pull request by {author} requires an approving review from "
        f"{MAINTAINER_LOGIN}."
    )


if __name__ == "__main__":
    raise SystemExit(main())
