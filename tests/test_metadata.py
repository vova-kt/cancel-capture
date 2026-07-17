import json
from typing import cast

from PIL import Image

from cancel_capture.adapters.metadata import BestEffortMetadataExtractor


def test_pillow_fallback_preserves_raw_and_normalized_exif(tmp_path) -> None:
    source = tmp_path / "metadata.jpg"
    image = Image.new("RGB", (20, 10), "white")
    exif = Image.Exif()
    exif[271] = "Archive Camera"
    exif[272] = "Model 7"
    exif[36867] = "2026:07:17 12:34:56"
    exif[305] = "Camera Software"
    image.save(source, exif=exif)

    metadata = BestEffortMetadataExtractor(exiftool="/definitely/missing/exiftool").extract(source)
    raw = cast(dict[str, object], json.loads(metadata.raw_json))

    assert metadata.extractor == "pillow-fallback"
    assert metadata.camera_make == "Archive Camera"
    assert metadata.camera_model == "Model 7"
    assert metadata.captured_at == "2026:07:17 12:34:56"
    assert raw["EXIF:Software"] == "Camera Software"
    assert raw["File:ImageWidth"] == 20
    extraction_error = cast(str, raw["CancelCapture:ExifToolError"])
    assert "FileNotFoundError" in extraction_error
    assert str(source) not in extraction_error
    assert "/definitely/missing/exiftool" not in extraction_error


def test_composite_gps_wins_over_unsigned_exif_values() -> None:
    metadata = BestEffortMetadataExtractor._normalize(
        {
            "EXIF:GPSLatitude": 33.9,
            "EXIF:GPSLatitudeRef": "S",
            "EXIF:GPSLongitude": 18.4,
            "EXIF:GPSLongitudeRef": "W",
            "EXIF:GPSAltitude": 12.5,
            "EXIF:GPSAltitudeRef": 1,
            "Composite:GPSLatitude": -33.9,
            "Composite:GPSLongitude": -18.4,
            "Composite:GPSAltitude": -12.5,
        },
        "test",
    )

    assert metadata.latitude == -33.9
    assert metadata.longitude == -18.4
    assert metadata.altitude_m == -12.5


def test_exif_gps_references_apply_when_composite_is_absent() -> None:
    metadata = BestEffortMetadataExtractor._normalize(
        {
            "EXIF:GPSLatitude": 33.9,
            "EXIF:GPSLatitudeRef": "S",
            "EXIF:GPSLongitude": 18.4,
            "EXIF:GPSLongitudeRef": "W",
            "EXIF:GPSAltitude": 12.5,
            "EXIF:GPSAltitudeRef": 1,
        },
        "test",
    )

    assert metadata.latitude == -33.9
    assert metadata.longitude == -18.4
    assert metadata.altitude_m == -12.5
