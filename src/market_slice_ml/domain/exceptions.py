"""Package-specific exceptions."""


class MarketSliceError(Exception):
    """Base package error."""


class DataLeakError(MarketSliceError):
    """Raised when validation time overlaps the training interval."""


class ProviderDisabledError(MarketSliceError):
    """Raised when an explicitly requested optional provider is unavailable."""


class SchemaValidationError(MarketSliceError):
    """Raised when data cannot satisfy the canonical schema."""
