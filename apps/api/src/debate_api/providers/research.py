"""Bounded safe fetch and local document extraction foundation spike."""

from __future__ import annotations

import asyncio
import codecs
import hashlib
import html.parser
import ipaddress
import re
import socket
from collections.abc import AsyncIterator, Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import TracebackType
from typing import Any, Protocol, Self
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpcore
import httpx
from httpcore._backends.auto import AutoBackend
from httpcore._backends.base import SOCKET_OPTION
from pydantic import ConfigDict, Field, field_validator

from debate_api.domain.models import StrictModel


class DocumentType(StrEnum):
    HTML = "html"


class DocumentStatus(StrEnum):
    EXTRACTED = "extracted"
    UNAVAILABLE = "unavailable"
    TIMEOUT = "timeout"
    UNSAFE_DESTINATION = "unsafe_destination"
    DNS_REBINDING = "dns_rebinding"
    REDIRECT_LIMIT = "redirect_limit"
    UNSUPPORTED_MEDIA_TYPE = "unsupported_media_type"
    CONTENT_TYPE_MISMATCH = "content_type_mismatch"
    TOO_LARGE = "too_large"
    MALFORMED_DOCUMENT = "malformed_document"


MAX_URL_LENGTH = 2_048
MAX_TITLE_LENGTH = 500


class ExtractorIdentity(StrictModel):
    name: str = Field(min_length=1, max_length=120)
    version: str | None = Field(default=None, max_length=80)


class ExtractedDocument(StrictModel):
    """Common, bounded shape consumed by later evidence persistence."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)
    source_url: str = Field(min_length=1, max_length=2_048)
    final_url: str = Field(min_length=1, max_length=2_048)
    document_type: DocumentType | None = None
    status: DocumentStatus
    text: str | None = Field(default=None, max_length=500_000)
    metadata: dict[str, str] = Field(default_factory=dict, max_length=8)
    warnings: list[str] = Field(default_factory=list, max_length=32)
    extractor: ExtractorIdentity | None = None
    content_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("warnings")
    @classmethod
    def warning_lengths(cls, values: list[str]) -> list[str]:
        if any(not value or len(value) > 300 for value in values):
            raise ValueError("warnings must be non-empty and at most 300 characters")
        return values


class FetchLimits(StrictModel):
    connect_timeout_seconds: float = Field(default=5.0, gt=0, le=60)
    read_timeout_seconds: float = Field(default=15.0, gt=0, le=120)
    write_timeout_seconds: float = Field(default=15.0, gt=0, le=120)
    pool_timeout_seconds: float = Field(default=5.0, gt=0, le=60)
    total_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    max_bytes: int = Field(default=2_000_000, ge=1_024, le=20_000_000)
    max_extracted_chars: int = Field(default=500_000, ge=1_024, le=2_000_000)
    max_redirects: int = Field(default=3, ge=0, le=10)
    max_concurrency: int = Field(default=4, ge=1, le=32)


class DnsResolver(Protocol):
    async def resolve(self, host: str, port: int) -> Sequence[str]: ...


class FetchProvider(Protocol):
    """Provider-neutral bounded fetch boundary.

    The returned document is normalized and may have a failure status.  A
    fetcher must never return unbounded response bytes or silently bypass the
    destination/content/timeout policies represented by ``FetchLimits``.
    """

    async def fetch(self, source_url: str) -> ExtractedDocument: ...


class DocumentExtractor(Protocol):
    """Extract bounded bytes into the common normalized document shape."""

    async def extract(
        self,
        data: bytes,
        *,
        source_url: str,
        final_url: str,
        content_type: str,
    ) -> ExtractedDocument: ...


class SystemDnsResolver:
    async def resolve(self, host: str, port: int) -> Sequence[str]:
        infos = await asyncio.to_thread(socket.getaddrinfo, host, port, type=socket.SOCK_STREAM)
        return tuple({str(info[4][0]) for info in infos})


class SafeFetchException(RuntimeError):
    def __init__(self, status: DocumentStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class _ValidatedUrl:
    url: str
    hostname: str
    port: int
    addresses: tuple[str, ...]


def _is_forbidden_ip(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return True
    mapped = getattr(address, "ipv4_mapped", None)
    if mapped is not None:
        return _is_forbidden_ip(str(mapped))
    return (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_unspecified
        or address.is_reserved
        or address in ipaddress.ip_network("100.64.0.0/10")
        or address
        in {ipaddress.ip_address("169.254.169.254"), ipaddress.ip_address("100.100.100.200")}
    )


def _canonical_url(url: str) -> _ValidatedUrl:
    if len(url) > MAX_URL_LENGTH:
        raise SafeFetchException(DocumentStatus.UNSAFE_DESTINATION, "URL exceeds maximum length")
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in url):
        raise SafeFetchException(
            DocumentStatus.UNSAFE_DESTINATION, "URL contains control characters"
        )
    try:
        parts = urlsplit(url)
        scheme = parts.scheme.lower()
        if scheme not in {"http", "https"}:
            raise SafeFetchException(
                DocumentStatus.UNSAFE_DESTINATION, "only http and https are allowed"
            )
        if parts.username is not None or parts.password is not None:
            raise SafeFetchException(
                DocumentStatus.UNSAFE_DESTINATION, "URL userinfo is not allowed"
            )
        if not parts.hostname:
            raise SafeFetchException(DocumentStatus.UNSAFE_DESTINATION, "URL host is required")
        hostname = parts.hostname.rstrip(".").encode("idna").decode("ascii").lower()
        if not hostname:
            raise SafeFetchException(DocumentStatus.UNSAFE_DESTINATION, "URL host is invalid")
        if re.fullmatch(r"[0-9.]+", hostname):
            try:
                address = ipaddress.ip_address(hostname)
            except ValueError:
                raise SafeFetchException(
                    DocumentStatus.UNSAFE_DESTINATION, "URL IP address is invalid"
                ) from None
            if str(address) != hostname:
                raise SafeFetchException(
                    DocumentStatus.UNSAFE_DESTINATION, "URL IP address is noncanonical"
                )
        port = parts.port or (443 if scheme == "https" else 80)
        if (
            port not in {80, 443}
            or (scheme == "http" and port != 80)
            or (scheme == "https" and port != 443)
        ):
            raise SafeFetchException(
                DocumentStatus.UNSAFE_DESTINATION, "URL scheme and port are invalid"
            )
        netloc = f"[{hostname}]" if ":" in hostname else hostname
        canonical = urlunsplit((scheme, netloc, parts.path or "/", parts.query, ""))
        if len(canonical) > MAX_URL_LENGTH:
            raise SafeFetchException(
                DocumentStatus.UNSAFE_DESTINATION, "URL exceeds maximum length"
            )
        return _ValidatedUrl(canonical, hostname, port, ())
    except SafeFetchException:
        raise
    except (UnicodeError, ValueError):
        raise SafeFetchException(DocumentStatus.UNSAFE_DESTINATION, "URL is malformed") from None


async def _validate_destination(url: str, resolver: DnsResolver) -> _ValidatedUrl:
    parsed = _canonical_url(url)
    try:
        literal = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        literal = None
    if literal is not None:
        if _is_forbidden_ip(str(literal)):
            raise SafeFetchException(
                DocumentStatus.UNSAFE_DESTINATION, "destination address is not public"
            )
        return _ValidatedUrl(parsed.url, parsed.hostname, parsed.port, (parsed.hostname,))
    try:
        first = tuple(sorted(set(await resolver.resolve(parsed.hostname, parsed.port))))
        second = tuple(sorted(set(await resolver.resolve(parsed.hostname, parsed.port))))
    except (OSError, socket.gaierror, ValueError):
        raise SafeFetchException(
            DocumentStatus.UNAVAILABLE, "destination DNS resolution failed"
        ) from None
    if not first or first != second:
        raise SafeFetchException(DocumentStatus.DNS_REBINDING, "destination DNS answers changed")
    if any(_is_forbidden_ip(value) for value in first):
        raise SafeFetchException(
            DocumentStatus.UNSAFE_DESTINATION, "destination resolves to a blocked address"
        )
    return _ValidatedUrl(parsed.url, parsed.hostname, parsed.port, first)


class _PinnedNetworkBackend(httpcore.AsyncNetworkBackend):
    """Connect only to validated IPs while preserving hostname SNI and Host."""

    def __init__(self) -> None:
        self._backend = AutoBackend()
        self._pins: dict[str, tuple[str, ...]] = {}

    def set_pins(self, hostname: str, addresses: tuple[str, ...]) -> None:
        self._pins[hostname] = addresses

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[SOCKET_OPTION] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        addresses = self._pins.get(host)
        if not addresses:
            raise httpcore.ConnectError("unvalidated destination")
        last_error: Exception | None = None
        for address in addresses:
            try:
                return await self._backend.connect_tcp(
                    address,
                    port,
                    timeout=timeout,
                    local_address=local_address,
                    socket_options=socket_options,
                )
            except Exception as error:
                last_error = error
        raise httpcore.ConnectError("all validated destination addresses failed") from last_error

    async def connect_unix_socket(
        self, *_args: Any, **_kwargs: Any
    ) -> httpcore.AsyncNetworkStream:
        raise httpcore.ConnectError("unix sockets are not supported")

    async def sleep(self, seconds: float) -> None:
        await self._backend.sleep(seconds)


class _PinnedTransport(httpx.AsyncHTTPTransport):
    def __init__(self, limits: FetchLimits, backend: _PinnedNetworkBackend) -> None:
        http_limits = httpx.Limits(
            max_connections=limits.max_concurrency, max_keepalive_connections=limits.max_concurrency
        )
        super().__init__(trust_env=False, limits=http_limits)
        old_pool = self._pool
        self._pool = httpcore.AsyncConnectionPool(
            ssl_context=old_pool._ssl_context,
            max_connections=limits.max_concurrency,
            max_keepalive_connections=limits.max_concurrency,
            network_backend=backend,
        )


class _FallbackHtmlParser(html.parser.HTMLParser):
    def __init__(self, max_chars: int) -> None:
        super().__init__(convert_charrefs=True)
        self.max_chars = max_chars
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self._in_title = False
        self._blocked_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "template"}:
            self._blocked_depth += 1
        elif tag == "title" and not self._blocked_depth:
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "template"} and self._blocked_depth:
            self._blocked_depth -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._blocked_depth:
            return
        if self._in_title:
            remaining_title = MAX_TITLE_LENGTH - sum(len(value) for value in self.title_parts)
            if remaining_title > 0:
                self.title_parts.append(data[:remaining_title])
        remaining = self.max_chars - sum(len(value) for value in self.text_parts)
        if remaining > 0:
            self.text_parts.append(data[:remaining])


def _decode_html(data: bytes, content_type: str) -> tuple[str, list[str]]:
    warnings: list[str] = []
    match = re.search(r"(?:^|;)\s*charset\s*=\s*[\"']?\s*([A-Za-z0-9._-]+)", content_type, re.I)
    encoding = (match.group(1) if match else "utf-8").lower()
    allowed_charsets = {
        "utf-8": "utf-8",
        "utf8": "utf-8",
        "ascii": "ascii",
        "iso-8859-1": "iso-8859-1",
        "latin-1": "iso-8859-1",
        "latin1": "iso-8859-1",
        "windows-1252": "cp1252",
        "cp1252": "cp1252",
    }
    if encoding not in allowed_charsets:
        warnings.append("unsupported charset; decoded as UTF-8 with replacement")
        encoding = "utf-8"
    else:
        encoding = allowed_charsets[encoding]
    try:
        codecs.lookup(encoding)
    except LookupError:  # pragma: no cover - allowlist is intentionally conservative
        warnings.append("unsupported charset; decoded as UTF-8 with replacement")
        encoding = "utf-8"
    return data.decode(encoding, errors="replace"), warnings


def _extract_html(
    data: bytes, content_type: str, max_chars: int
) -> tuple[str | None, dict[str, str], list[str], ExtractorIdentity]:
    decoded, warnings = _decode_html(data, content_type)
    try:
        import trafilatura

        text = trafilatura.extract(decoded, include_comments=False, include_tables=True)
        version = getattr(trafilatura, "__version__", None)
        identity = ExtractorIdentity(name="trafilatura", version=version)
        if text:
            text = text.strip()
            if len(text) > max_chars:
                return None, {}, warnings + ["extracted text exceeded character limit"], identity
            parser = _FallbackHtmlParser(max_chars)
            parser.feed(decoded)
            metadata = (
                {"title": " ".join("".join(parser.title_parts).split())[:MAX_TITLE_LENGTH]}
                if parser.title_parts
                else {}
            )
            return text, metadata, warnings, identity
        warnings.append("trafilatura returned no main text; used stdlib HTML fallback")
        identity = ExtractorIdentity(name="trafilatura+stdlib-fallback", version=version)
    except Exception:
        warnings.append("trafilatura was unavailable or failed; used stdlib HTML fallback")
        identity = ExtractorIdentity(name="trafilatura+stdlib-fallback", version=None)
    parser = _FallbackHtmlParser(max_chars)
    try:
        parser.feed(decoded)
        parser.close()
    except (ValueError, AssertionError):
        warnings.append("HTML parser reported malformed markup")
    text = " ".join(" ".join(parser.text_parts).split())
    fallback_metadata = (
        {"title": " ".join("".join(parser.title_parts).split())[:MAX_TITLE_LENGTH]}
        if parser.title_parts
        else {}
    )
    return text or None, fallback_metadata, warnings, identity


class SafeDocumentExtractor:
    """Local extractor enforcing byte/type limits before parsing any payload."""

    def __init__(
        self,
        *,
        limits: FetchLimits | None = None,
    ) -> None:
        self.limits = limits or FetchLimits()

    async def extract(
        self,
        data: bytes,
        *,
        source_url: str,
        final_url: str,
        content_type: str,
    ) -> ExtractedDocument:
        if not _document_urls_are_bounded(source_url, final_url):
            return _failure_document(
                source_url,
                final_url,
                DocumentStatus.UNSAFE_DESTINATION,
                "document URL metadata is empty or exceeds the maximum length",
            )
        if len(data) > self.limits.max_bytes:
            return _failure_document(
                source_url, final_url, DocumentStatus.TOO_LARGE, "response exceeds byte limit"
            )
        mime = content_type.split(";", 1)[0].strip().lower()
        detected = DocumentType.HTML if b"<" in data[:512] else None
        allowed = {
            "text/html": DocumentType.HTML,
            "application/xhtml+xml": DocumentType.HTML,
        }
        declared = allowed.get(mime)
        if declared is None and detected is None:
            return _failure_document(
                source_url,
                final_url,
                DocumentStatus.UNSUPPORTED_MEDIA_TYPE,
                "content type is not supported",
            )
        if content_type and declared is None:
            return _failure_document(
                source_url,
                final_url,
                DocumentStatus.CONTENT_TYPE_MISMATCH,
                "declared content type is not on the allowlist",
            )
        if declared is not None and detected is not None and declared != detected:
            return _failure_document(
                source_url,
                final_url,
                DocumentStatus.CONTENT_TYPE_MISMATCH,
                "content type does not match document bytes",
            )
        kind = declared or detected
        assert kind is not None
        content_hash = hashlib.sha256(data).hexdigest()
        if kind == DocumentType.HTML:
            text, metadata, warnings, html_extractor = _extract_html(
                data, content_type, self.limits.max_extracted_chars
            )
            status = DocumentStatus.EXTRACTED if text else DocumentStatus.MALFORMED_DOCUMENT
            if not text:
                warnings.append("HTML contained no extractable text")
            return ExtractedDocument(
                source_url=source_url,
                final_url=final_url,
                document_type=kind,
                status=status,
                text=text,
                metadata=metadata,
                warnings=warnings,
                extractor=html_extractor,
                content_hash=content_hash,
            )
        return _failure_document(
            source_url,
            final_url,
            DocumentStatus.UNSUPPORTED_MEDIA_TYPE,
            "only HTML documents are supported",
        )


def _failure_document(
    source_url: object, final_url: object, status: DocumentStatus, warning: str
) -> ExtractedDocument:
    def safe_url(value: object) -> str:
        if isinstance(value, str):
            if 0 < len(value) <= MAX_URL_LENGTH:
                return value
        return "about:blank"

    return ExtractedDocument(
        source_url=safe_url(source_url),
        final_url=safe_url(final_url),
        status=status,
        warnings=[warning[:300]],
    )


def _document_urls_are_bounded(source_url: object, final_url: object) -> bool:
    return all(
        isinstance(value, str) and 0 < len(value) <= MAX_URL_LENGTH
        for value in (source_url, final_url)
    )


class SafeDocumentFetcher:
    def __init__(
        self,
        *,
        limits: FetchLimits | None = None,
        resolver: DnsResolver | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.limits = limits or FetchLimits()
        self.resolver = resolver or SystemDnsResolver()
        self._backend = _PinnedNetworkBackend()
        self._transport = transport or _PinnedTransport(self.limits, self._backend)
        self._client: httpx.AsyncClient | None = None
        self._extractor = SafeDocumentExtractor(limits=self.limits)
        self._semaphore = asyncio.Semaphore(self.limits.max_concurrency)

    async def __aenter__(self) -> Self:
        self._client = httpx.AsyncClient(
            follow_redirects=False,
            trust_env=False,
            timeout=httpx.Timeout(
                timeout=self.limits.total_timeout_seconds,
                connect=self.limits.connect_timeout_seconds,
                read=self.limits.read_timeout_seconds,
                write=self.limits.write_timeout_seconds,
                pool=self.limits.pool_timeout_seconds,
            ),
            limits=httpx.Limits(
                max_connections=self.limits.max_concurrency,
                max_keepalive_connections=self.limits.max_concurrency,
            ),
            transport=self._transport,
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def fetch(self, source_url: str) -> ExtractedDocument:
        if self._client is None:
            raise RuntimeError("SafeDocumentFetcher must be used as an async context manager")
        async with self._semaphore:
            try:
                return await asyncio.wait_for(
                    self._fetch(source_url), self.limits.total_timeout_seconds
                )
            except asyncio.CancelledError:
                raise
            except (TimeoutError, httpx.TimeoutException):
                return self._failure(
                    source_url, source_url, DocumentStatus.TIMEOUT, "fetch timed out"
                )

    async def _fetch(self, source_url: str) -> ExtractedDocument:
        client = self._client
        if client is None:
            raise RuntimeError("SafeDocumentFetcher must be used as an async context manager")
        current = source_url
        redirects = 0
        previous_scheme: str | None = None
        try:
            while True:
                validated = await _validate_destination(current, self.resolver)
                if previous_scheme == "https" and validated.url.startswith("http://"):
                    return self._failure(
                        source_url,
                        validated.url,
                        DocumentStatus.UNSAFE_DESTINATION,
                        "HTTPS downgrade redirect rejected",
                    )
                previous_scheme = "https" if validated.url.startswith("https://") else "http"
                self._backend.set_pins(validated.hostname, validated.addresses)
                try:
                    async with client.stream("GET", validated.url) as response:
                        if response.is_redirect:
                            if redirects >= self.limits.max_redirects:
                                return self._failure(
                                    source_url,
                                    validated.url,
                                    DocumentStatus.REDIRECT_LIMIT,
                                    "redirect limit exceeded",
                                )
                            location = response.headers.get("location")
                            if not location:
                                return self._failure(
                                    source_url,
                                    validated.url,
                                    DocumentStatus.UNAVAILABLE,
                                    "redirect omitted location",
                                )
                            current = urljoin(validated.url, location)
                            redirects += 1
                            continue
                        if response.status_code < 200 or response.status_code >= 300:
                            return self._failure(
                                source_url,
                                validated.url,
                                DocumentStatus.UNAVAILABLE,
                                "source returned a non-success status",
                            )
                        return await self._read_and_extract(source_url, validated.url, response)
                except asyncio.CancelledError:
                    raise
                except (httpx.TimeoutException, TimeoutError):
                    return self._failure(
                        source_url, validated.url, DocumentStatus.TIMEOUT, "fetch timed out"
                    )
                except (httpx.NetworkError, httpx.ProtocolError):
                    return self._failure(
                        source_url, validated.url, DocumentStatus.UNAVAILABLE, "fetch failed"
                    )
        except SafeFetchException as error:
            return self._failure(source_url, current, error.status, str(error))

    async def _read_and_extract(
        self, source_url: str, final_url: str, response: httpx.Response
    ) -> ExtractedDocument:
        content_length = response.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > self.limits.max_bytes:
                    return _failure_document(
                        source_url,
                        final_url,
                        DocumentStatus.TOO_LARGE,
                        "response exceeds byte limit",
                    )
            except ValueError:
                pass
        data = bytearray()
        async for chunk in response.aiter_bytes():
            if len(chunk) > self.limits.max_bytes - len(data):
                return _failure_document(
                    source_url, final_url, DocumentStatus.TOO_LARGE, "response exceeds byte limit"
                )
            data.extend(chunk)
        return await self._extractor.extract(
            bytes(data),
            source_url=source_url,
            final_url=final_url,
            content_type=response.headers.get("content-type", "").strip(),
        )

    @staticmethod
    def _failure(
        source_url: str, final_url: str, status: DocumentStatus, warning: str
    ) -> ExtractedDocument:
        return _failure_document(source_url, final_url, status, warning)


async def iter_response_bytes(response: httpx.Response, max_bytes: int) -> AsyncIterator[bytes]:
    total = 0
    async for chunk in response.aiter_bytes():
        if len(chunk) > max_bytes - total:
            raise SafeFetchException(DocumentStatus.TOO_LARGE, "response exceeds byte limit")
        total += len(chunk)
        yield chunk
