"""Upload validation and in-memory image decoding.

Nothing is ever written to disk: the request body is read into a ``BytesIO``,
decoded, and discarded once inference finishes.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import PurePath

from PIL import Image, UnidentifiedImageError

from app.config import Settings

LOGGER = logging.getLogger(__name__)


class ImageValidationError(ValueError):
    """Raised when an upload is not an acceptable image.

    ``status_code`` lets the route layer map the failure onto the right HTTP
    status without inspecting error strings.
    """

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class ValidatedImage:
    """A decoded, trusted image plus the metadata worth reporting."""

    image: Image.Image
    content_type: str
    byte_size: int
    width: int
    height: int

    @property
    def megapixels(self) -> float:
        return (self.width * self.height) / 1_000_000


def validate_extension(filename: str | None, settings: Settings) -> str:
    """Check the filename extension against the allow-list.

    Returns the lower-cased extension. Raises :class:`ImageValidationError`.
    """
    if not filename:
        raise ImageValidationError("No filename was provided.", 400)
    extension = PurePath(filename).suffix.lower()
    if not extension:
        raise ImageValidationError("The file has no extension.", 400)
    if extension not in settings.allowed_extensions:
        allowed = ", ".join(settings.allowed_extensions)
        raise ImageValidationError(
            f"Unsupported file type '{extension}'. Allowed types: {allowed}."
        )
    return extension


def validate_content_type(content_type: str | None, settings: Settings) -> str:
    """Check the declared MIME type against the allow-list."""
    if not content_type:
        raise ImageValidationError("The upload has no content type.", 400)
    normalised = content_type.split(";")[0].strip().lower()
    if normalised not in settings.allowed_mime_types:
        allowed = ", ".join(settings.allowed_mime_types)
        raise ImageValidationError(
            f"Unsupported content type '{normalised}'. Allowed types: {allowed}."
        )
    return normalised


def validate_size(size: int, settings: Settings) -> None:
    """Reject empty and oversized uploads before any decoding happens."""
    if size <= 0:
        raise ImageValidationError("The uploaded file is empty.", 400)
    if size > settings.max_upload_bytes:
        limit_mb = settings.max_upload_bytes / (1024 * 1024)
        raise ImageValidationError(
            f"The image is larger than the {limit_mb:.0f} MB limit.", 413
        )


def decode_image(payload: bytes, settings: Settings) -> ValidatedImage:
    """Decode bytes into a verified RGB image.

    Guards against decompression bombs, tiny images and corrupt files, and
    normalises the mode to RGB. Every failure mode becomes a
    :class:`ImageValidationError` with a user-safe message - the underlying
    exception is logged but never returned to the client.
    """
    # Keep Pillow's own bomb guard as a hard backstop, but do the size check
    # explicitly below so a just-under-the-limit image is rejected too.
    previous_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = settings.max_image_pixels * 2
    try:
        with Image.open(io.BytesIO(payload)) as handle:
            detected_format = (handle.format or "").upper()
            width, height = handle.size

            # Reject before `load()` so Pillow never allocates the buffer.
            if width * height > settings.max_image_pixels:
                LOGGER.warning(
                    "Rejected oversized image: %dx%d = %.1f MP (limit %.1f MP)",
                    width, height, width * height / 1e6,
                    settings.max_image_pixels / 1e6,
                )
                raise ImageValidationError(
                    "The image has unrealistic dimensions and was rejected.", 413
                )

            handle.load()
            image = handle.convert("RGB")
    except Image.DecompressionBombError as exc:
        LOGGER.warning("Rejected decompression bomb: %s", exc)
        raise ImageValidationError(
            "The image has unrealistic dimensions and was rejected.", 413
        ) from exc
    except UnidentifiedImageError as exc:
        LOGGER.info("Unidentified image: %s", exc)
        raise ImageValidationError(
            "That file could not be read as an image. It may be corrupt.", 400
        ) from exc
    except (OSError, ValueError, SyntaxError) as exc:
        LOGGER.info("Failed to decode image: %s: %s", type(exc).__name__, exc)
        raise ImageValidationError(
            "That image appears to be corrupt and could not be decoded.", 400
        ) from exc
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit

    if min(width, height) < settings.min_image_side:
        raise ImageValidationError(
            f"The image is too small ({width}x{height}px). "
            f"Minimum is {settings.min_image_side}px on the shortest side.",
            400,
        )

    LOGGER.info(
        "Decoded %s %dx%d (%.2f MP, %.1f KB)",
        detected_format, width, height,
        (width * height) / 1_000_000, len(payload) / 1024,
    )
    return ValidatedImage(
        image=image,
        content_type=detected_format or "unknown",
        byte_size=len(payload),
        width=width,
        height=height,
    )


def validate_upload(
    *,
    payload: bytes,
    filename: str | None,
    content_type: str | None,
    settings: Settings,
) -> ValidatedImage:
    """Run every check in order and return the decoded image.

    Order matters: size first (cheapest, and stops oversized bodies being
    decoded), then declared MIME type, then extension, then actual decoding.
    """
    validate_size(len(payload), settings)
    validate_content_type(content_type, settings)
    validate_extension(filename, settings)
    return decode_image(payload, settings)
