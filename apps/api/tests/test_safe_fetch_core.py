from __future__ import annotations

from collections.abc import Mapping, Sequence

import httpx
import pytest

from debate_api.providers.research import DocumentStatus, FetchLimits, SafeDocumentFetcher

PUBLIC_IP = "93.184.216.34"


class FixtureResolver:
    def __init__(self, answers: Mapping[str, Sequence[str]] | Sequence[Sequence[str]]) -> None:
        self.answers = answers
        self.calls = 0

    async def resolve(self, host: str, port: int) -> Sequence[str]:
        del port
        index = self.calls
        self.calls += 1
        if isinstance(self.answers, Mapping):
            return self.answers[host]
        return self.answers[min(index, len(self.answers) - 1)]


@pytest.mark.asyncio
async def test_html_normalization_hash_and_title() -> None:
    resolver = FixtureResolver({"docs.test": [PUBLIC_IP]})
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            headers={"content-type": "text/html"},
            content=(
                b"<html><title>Fixture</title><body><h1>Safe text</h1>"
                b"<p>Published content.</p></body></html>"
            ),
        )
    )
    async with SafeDocumentFetcher(resolver=resolver, transport=transport) as fetcher:
        document = await fetcher.fetch("https://docs.test/")
    assert document.status == DocumentStatus.EXTRACTED
    assert document.text and "Safe text" in document.text
    assert document.content_hash is not None
    assert document.metadata == {"title": "Fixture"}


@pytest.mark.asyncio
async def test_redirect_revalidates_final_destination() -> None:
    resolver = FixtureResolver({"start.test": [PUBLIC_IP], "final.test": [PUBLIC_IP]})

    async def request(request: httpx.Request) -> httpx.Response:
        if request.url.host == "start.test":
            return httpx.Response(302, headers={"location": "https://final.test/article"})
        return httpx.Response(200, headers={"content-type": "text/html"}, content=b"<p>final</p>")

    async with SafeDocumentFetcher(
        resolver=resolver, transport=httpx.MockTransport(request)
    ) as fetcher:
        document = await fetcher.fetch("https://start.test/")
    assert document.status == DocumentStatus.EXTRACTED
    assert document.final_url == "https://final.test/article"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,answer,status",
    [
        ("http://127.0.0.1/", {}, DocumentStatus.UNSAFE_DESTINATION),
        (
            "https://metadata.test/",
            {"metadata.test": ["169.254.169.254"]},
            DocumentStatus.UNSAFE_DESTINATION,
        ),
        ("https://rebind.test/", [[PUBLIC_IP], ["93.184.216.35"]], DocumentStatus.DNS_REBINDING),
        (
            "https://user:pass@public.test/",
            {"public.test": [PUBLIC_IP]},
            DocumentStatus.UNSAFE_DESTINATION,
        ),
    ],
)
async def test_destination_policy_rejects_unsafe_targets(
    url: str, answer: object, status: DocumentStatus
) -> None:
    resolver = FixtureResolver(answer)  # type: ignore[arg-type]
    async with SafeDocumentFetcher(
        resolver=resolver, transport=httpx.MockTransport(lambda _: httpx.Response(200))
    ) as fetcher:
        document = await fetcher.fetch(url)
    assert document.status == status


@pytest.mark.asyncio
async def test_oversized_and_content_type_mismatch_are_explicit() -> None:
    resolver = FixtureResolver({"large.test": [PUBLIC_IP], "mismatch.test": [PUBLIC_IP]})

    async def request(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/large":
            return httpx.Response(200, headers={"content-type": "text/html"}, content=b"x" * 2048)
        return httpx.Response(
            200, headers={"content-type": "application/pdf"}, content=b"<p>not PDF</p>"
        )

    async with SafeDocumentFetcher(
        resolver=resolver,
        limits=FetchLimits(max_bytes=1024),
        transport=httpx.MockTransport(request),
    ) as fetcher:
        large = await fetcher.fetch("https://large.test/large")
        mismatch = await fetcher.fetch("https://mismatch.test/mismatch")
    assert large.status == DocumentStatus.TOO_LARGE
    assert mismatch.status == DocumentStatus.CONTENT_TYPE_MISMATCH


@pytest.mark.asyncio
async def test_malformed_html_and_charset_are_safe_failures_or_normalized() -> None:
    resolver = FixtureResolver({"bad.test": [PUBLIC_IP], "charset.test": [PUBLIC_IP]})

    async def request(request: httpx.Request) -> httpx.Response:
        if request.url.host == "bad.test":
            return httpx.Response(
                200, headers={"content-type": "text/html"}, content=b"<html><body><div>"
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=base64_codec"},
            content=b"<p>safe text</p>",
        )

    async with SafeDocumentFetcher(
        resolver=resolver, transport=httpx.MockTransport(request)
    ) as fetcher:
        malformed = await fetcher.fetch("https://bad.test/")
        charset = await fetcher.fetch("https://charset.test/")
    assert malformed.status == DocumentStatus.MALFORMED_DOCUMENT
    assert charset.status == DocumentStatus.EXTRACTED
    assert charset.text == "safe text"
    assert any("unsupported charset" in warning for warning in charset.warnings)


@pytest.mark.asyncio
async def test_url_length_is_rejected_before_dns() -> None:
    resolver = FixtureResolver({})
    async with SafeDocumentFetcher(
        resolver=resolver, transport=httpx.MockTransport(lambda _: httpx.Response(200))
    ) as fetcher:
        document = await fetcher.fetch("https://example.com/" + "x" * 2100)
    assert document.status == DocumentStatus.UNSAFE_DESTINATION
    assert resolver.calls == 0
