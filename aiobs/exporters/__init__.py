"""Exporters for aiobs observability data.

Exporters handle the serialization and transport of observability data
to various destinations.

Supported exporters:
    - GCSExporter: Export to Google Cloud Storage
    - S3Exporter: Export to Amazon S3
    - CustomExporter: User-defined export logic via callback
    - CompositeExporter: Export to multiple destinations

Example usage:
    # Using GCSExporter
    from aiobs import observer
    from aiobs.exporters import GCSExporter

    exporter = GCSExporter(
        bucket="my-observability-bucket",
        prefix="traces/",
        project="my-gcp-project",
    )

    observer.observe()
    # ... your agent code ...
    observer.end()
    observer.flush(exporter=exporter)

    # Using S3Exporter
    from aiobs.exporters import S3Exporter

    s3_exporter = S3Exporter(
        bucket="my-traces-bucket",
        region="us-east-1",
        prefix="shepherd/",
    )
    observer.flush(exporter=s3_exporter)

    # Using CustomExporter
    from aiobs.exporters import CustomExporter, ExportResult

    def my_export_handler(data, options):
        # Send to custom API, database, etc.
        # (implementation depends on your needs)
        return ExportResult(
            success=True,
            destination="custom://destination",
            metadata={"custom": "metadata"},
        )

    custom_exporter = CustomExporter(handler=my_export_handler)
    observer.flush(exporter=custom_exporter)
"""

from .base import BaseExporter, ExportResult, ExportError
from .custom import CustomExporter, CompositeExporter

__all__ = [
    # Base classes
    "BaseExporter",
    "ExportResult",
    "ExportError",
    # Built-in exporters
    "GCSExporter",
    "S3Exporter",
    "CustomExporter",
    "CompositeExporter",
]


# Lazy imports for optional cloud exporters (to avoid requiring deps at import time)
def __getattr__(name: str):
    if name == "GCSExporter":
        from .gcs import GCSExporter
        return GCSExporter
    if name == "S3Exporter":
        from .s3 import S3Exporter
        return S3Exporter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

