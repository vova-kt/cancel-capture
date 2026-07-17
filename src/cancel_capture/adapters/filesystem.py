from __future__ import annotations

import hashlib
import mimetypes
import os
import tempfile
from pathlib import Path

from cancel_capture.errors import ImageTooLargeError
from cancel_capture.models import StoredAsset

_MAGIC_TYPES: tuple[tuple[bytes, str, str], ...] = (
    (b"\xff\xd8\xff", "image/jpeg", ".jpg"),
    (b"\x89PNG\r\n\x1a\n", "image/png", ".png"),
    (b"II*\x00", "image/tiff", ".tif"),
    (b"MM\x00*", "image/tiff", ".tif"),
)


def _sniff_type(data: bytes, declared: str | None) -> tuple[str, str]:
    for prefix, media_type, extension in _MAGIC_TYPES:
        if data.startswith(prefix):
            return media_type, extension
    if len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp", ".webp"
    if len(data) >= 12 and data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand in {b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1"}:
            return "image/heic", ".heic"
    if declared is not None and declared.startswith("image/"):
        extension = mimetypes.guess_extension(declared, strict=False) or ".bin"
        return declared, extension
    return "application/octet-stream", ".bin"


class ContentAddressedAssetStore:
    def __init__(self, root: Path, max_upload_bytes: int) -> None:
        self._root = root.resolve()
        self._max_upload_bytes = max_upload_bytes
        self._root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._root.chmod(0o700)

    def save_original(
        self, data: bytes, filename: str | None, declared_media_type: str | None
    ) -> StoredAsset:
        if len(data) > self._max_upload_bytes:
            raise ImageTooLargeError(
                f"File is {len(data)} bytes; limit is {self._max_upload_bytes} bytes"
            )
        sha256 = hashlib.sha256(data).hexdigest()
        media_type, extension = _sniff_type(data, declared_media_type)
        relative_path = Path("assets") / "originals" / sha256[:2] / f"{sha256}{extension}"
        self._write_once(relative_path, data)
        clean_filename = Path(filename).name if filename else None
        return StoredAsset(
            sha256=sha256,
            relative_path=relative_path.as_posix(),
            media_type=media_type,
            byte_size=len(data),
            width=None,
            height=None,
            original_filename=clean_filename,
        )

    def save_crop(self, data: bytes, width: int, height: int) -> StoredAsset:
        sha256 = hashlib.sha256(data).hexdigest()
        relative_path = Path("assets") / "crops" / sha256[:2] / f"{sha256}.jpg"
        self._write_once(relative_path, data)
        return StoredAsset(
            sha256=sha256,
            relative_path=relative_path.as_posix(),
            media_type="image/jpeg",
            byte_size=len(data),
            width=width,
            height=height,
            original_filename=None,
        )

    def resolve(self, relative_path: str) -> Path:
        candidate = (self._root / relative_path).resolve()
        if not candidate.is_relative_to(self._root):
            raise ValueError("Asset path escapes the configured data directory")
        return candidate

    def _write_once(self, relative_path: Path, data: bytes) -> None:
        destination = self.resolve(relative_path.as_posix())
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if destination.exists():
            return
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent, prefix=".asset-", delete=False
            ) as temporary:
                temporary_name = temporary.name
                temporary.write(data)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.chmod(temporary_name, 0o600)
            os.replace(temporary_name, destination)
            temporary_name = None
        finally:
            if temporary_name is not None:
                Path(temporary_name).unlink(missing_ok=True)
