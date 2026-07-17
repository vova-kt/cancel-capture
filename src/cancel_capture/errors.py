class CancelCaptureError(Exception):
    pass


class ConfigurationError(CancelCaptureError):
    pass


class UnsupportedImageError(CancelCaptureError):
    pass


class ImageTooLargeError(CancelCaptureError):
    pass


class ProviderResponseError(CancelCaptureError):
    pass


class DuplicateSourceError(CancelCaptureError):
    pass


class ReviewConflictError(CancelCaptureError):
    pass


class CandidateNotFoundError(CancelCaptureError):
    pass
