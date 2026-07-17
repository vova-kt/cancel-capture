from __future__ import annotations

import json
import math
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import cast

from PIL import ExifTags, Image

from cancel_capture.models import ImageMetadata

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]


def _json_safe(value: object) -> JsonValue:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, Fraction):
        return float(value)
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {str(key): _json_safe(item) for key, item in mapping.items()}
    if isinstance(value, Sequence):
        sequence = cast(Sequence[object], value)
        return [_json_safe(item) for item in sequence]
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(value)


def _find(data: Mapping[str, JsonValue], names: tuple[str, ...]) -> JsonValue | None:
    for name in names:
        for key, value in data.items():
            if key == name or key.endswith(f":{name}"):
                return value
    return None


def _as_float(value: JsonValue | None) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, str):
        try:
            result = float(value)
            return result if math.isfinite(result) else None
        except ValueError:
            return None
    return None


def _as_int(value: JsonValue | None) -> int | None:
    number = _as_float(value)
    return int(number) if number is not None else None


def _as_str(value: JsonValue | None) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _group_value(
    data: Mapping[str, JsonValue], groups: tuple[str, ...], name: str
) -> JsonValue | None:
    for group in groups:
        for key, value in data.items():
            prefix, separator, suffix = key.partition(":")
            if separator and prefix.casefold().startswith(group.casefold()) and suffix == name:
                return value
    return None


def _signed_gps_coordinate(
    data: Mapping[str, JsonValue], name: str, negative_reference: str
) -> float | None:
    composite = _as_float(data.get(f"Composite:{name}"))
    if composite is not None:
        return composite
    xmp = _as_float(_group_value(data, ("XMP",), name))
    if xmp is not None:
        return xmp
    value = _as_float(_group_value(data, ("GPS", "EXIF"), name))
    if value is None:
        return None
    reference = _group_value(data, ("GPS", "EXIF"), f"{name}Ref")
    if isinstance(reference, str) and reference.strip().casefold() == negative_reference.casefold():
        return -abs(value)
    if name == "GPSAltitude" and isinstance(reference, str) and "below" in reference.casefold():
        return -abs(value)
    if name == "GPSAltitude" and _as_int(reference) == 1:
        return -abs(value)
    return value


class BestEffortMetadataExtractor:
    def __init__(self, exiftool: str | None = None) -> None:
        self._exiftool = exiftool if exiftool is not None else shutil.which("exiftool")

    def extract(self, path: Path) -> ImageMetadata:
        if self._exiftool is not None:
            try:
                data = self._with_exiftool(path)
                return self._normalize(data, "exiftool")
            except (
                OSError,
                subprocess.SubprocessError,
                ValueError,
                json.JSONDecodeError,
            ) as error:
                data = self._with_pillow(path)
                data["CancelCapture:ExifToolError"] = (
                    f"{type(error).__name__}; ExifTool extraction failed and Pillow fallback "
                    "was used"
                )
                return self._normalize(data, "pillow-fallback")
        return self._normalize(self._with_pillow(path), "pillow")

    def _with_exiftool(self, path: Path) -> dict[str, JsonValue]:
        completed = subprocess.run(
            [self._exiftool or "exiftool", "-json", "-G", "-n", str(path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        decoded = cast(object, json.loads(completed.stdout))
        if not isinstance(decoded, list) or not decoded or not isinstance(decoded[0], dict):
            raise ValueError("ExifTool returned an unexpected payload")
        raw = cast(dict[object, object], decoded[0])
        private_path_keys = {"SourceFile", "File:Directory", "File:FilePermissions"}
        return {
            str(key): _json_safe(value)
            for key, value in raw.items()
            if str(key) not in private_path_keys
        }

    @staticmethod
    def _with_pillow(path: Path) -> dict[str, JsonValue]:
        data: dict[str, JsonValue] = {}
        with Image.open(path) as image:
            data["File:ImageWidth"] = image.width
            data["File:ImageHeight"] = image.height
            data["File:ImageFormat"] = image.format
            exif = image.getexif()
            for tag_id, value in exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                data[f"EXIF:{tag_name}"] = _json_safe(value)
            try:
                gps = cast(
                    dict[int, object],
                    exif.get_ifd(  # pyright: ignore[reportUnknownMemberType]
                        ExifTags.IFD.GPSInfo
                    ),
                )
            except (KeyError, TypeError):
                gps = {}
            for tag_id, value in gps.items():
                tag_name = ExifTags.GPSTAGS.get(tag_id, str(tag_id))
                data[f"GPS:{tag_name}"] = _json_safe(value)
        BestEffortMetadataExtractor._normalize_pillow_gps(data)
        return data

    @staticmethod
    def _normalize_pillow_gps(data: dict[str, JsonValue]) -> None:
        latitude = data.get("GPS:GPSLatitude")
        longitude = data.get("GPS:GPSLongitude")
        if isinstance(latitude, list) and len(latitude) == 3:
            values = [_as_float(value) for value in latitude]
            if all(value is not None for value in values):
                degrees = cast(list[float], values)
                result = degrees[0] + degrees[1] / 60.0 + degrees[2] / 3600.0
                if data.get("GPS:GPSLatitudeRef") == "S":
                    result *= -1
                data["Composite:GPSLatitude"] = result
        if isinstance(longitude, list) and len(longitude) == 3:
            values = [_as_float(value) for value in longitude]
            if all(value is not None for value in values):
                degrees = cast(list[float], values)
                result = degrees[0] + degrees[1] / 60.0 + degrees[2] / 3600.0
                if data.get("GPS:GPSLongitudeRef") == "W":
                    result *= -1
                data["Composite:GPSLongitude"] = result
        altitude = _as_float(data.get("GPS:GPSAltitude"))
        if altitude is not None:
            if _as_int(data.get("GPS:GPSAltitudeRef")) == 1:
                altitude *= -1
            data["Composite:GPSAltitude"] = altitude

    @staticmethod
    def _normalize(data: dict[str, JsonValue], extractor: str) -> ImageMetadata:
        data["CancelCapture:Extractor"] = extractor
        raw_json = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return ImageMetadata(
            raw_json=raw_json,
            captured_at=_as_str(
                _find(data, ("SubSecDateTimeOriginal", "DateTimeOriginal", "CreateDate"))
            ),
            latitude=_signed_gps_coordinate(data, "GPSLatitude", "S"),
            longitude=_signed_gps_coordinate(data, "GPSLongitude", "W"),
            altitude_m=_signed_gps_coordinate(data, "GPSAltitude", "below"),
            camera_make=_as_str(_find(data, ("Make",))),
            camera_model=_as_str(_find(data, ("Model",))),
            lens_model=_as_str(_find(data, ("LensModel",))),
            software=_as_str(_find(data, ("Software",))),
            orientation=_as_int(_find(data, ("Orientation",))),
            extractor=extractor,
        )
