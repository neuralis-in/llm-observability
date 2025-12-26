"""Amazon S3 exporter for aiobs observability data.

Exports observability data to Amazon S3 buckets.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

from .base import BaseExporter, ExportResult, ExportError
from ..models import ObservabilityExport


class S3Exporter(BaseExporter):
    """Export observability data to Amazon S3.

    Example usage:
        from aiobs.exporters import S3Exporter

        exporter = S3Exporter(
            bucket="my-traces-bucket",
            region="us-east-1",
            prefix="shepherd/",
        )
        observer.flush(exporter=exporter)

    Authentication:
        The exporter uses AWS's default authentication chain:
        1. AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables
        2. AWS credentials file (~/.aws/credentials)
        3. IAM role credentials (when running on EC2/ECS/Lambda)
        4. AWS_PROFILE environment variable for named profiles

    Args:
        bucket: The S3 bucket name (required).
        region: AWS region name (e.g., "us-east-1"). Defaults to "us-east-1".
        prefix: Path prefix within the bucket (e.g., "shepherd/"). Defaults to "".
        aws_access_key_id: AWS access key ID. If not provided, uses default auth chain.
        aws_secret_access_key: AWS secret access key. If not provided, uses default auth chain.
        aws_session_token: AWS session token for temporary credentials.
        filename_template: Template for the output filename. Supports placeholders:
                          - {session_id}: First session ID
                          - {timestamp}: Unix timestamp
                          - {date}: Date in YYYY-MM-DD format
                          Defaults to "{session_id}.json"
        content_type: Content-Type for uploaded files. Defaults to "application/json".
    """

    name: str = "s3"

    def __init__(
        self,
        bucket: str,
        region: str = "us-east-1",
        prefix: str = "",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        filename_template: str = "{session_id}.json",
        content_type: str = "application/json",
    ) -> None:
        self.bucket = bucket
        self.region = region
        self.prefix = prefix.rstrip("/") + "/" if prefix and not prefix.endswith("/") else prefix
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token
        self.filename_template = filename_template
        self.content_type = content_type
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazily initialize the S3 client."""
        if self._client is None:
            try:
                import boto3
            except ImportError as e:
                raise ExportError(
                    "boto3 is required for S3Exporter. "
                    "Install it with: pip install boto3",
                    cause=e,
                )

            client_kwargs = {"region_name": self.region}
            
            if self.aws_access_key_id and self.aws_secret_access_key:
                client_kwargs["aws_access_key_id"] = self.aws_access_key_id
                client_kwargs["aws_secret_access_key"] = self.aws_secret_access_key
                if self.aws_session_token:
                    client_kwargs["aws_session_token"] = self.aws_session_token
            
            self._client = boto3.client("s3", **client_kwargs)

        return self._client

    def _generate_filename(self, data: ObservabilityExport) -> str:
        """Generate the filename from the template."""
        session_id = "unknown"
        if data.sessions:
            session_id = data.sessions[0].id

        timestamp = int(time.time())
        date = time.strftime("%Y-%m-%d")

        return self.filename_template.format(
            session_id=session_id,
            timestamp=timestamp,
            date=date,
        )

    def export(self, data: ObservabilityExport, **kwargs: Any) -> ExportResult:
        """Export observability data to S3.

        Args:
            data: The ObservabilityExport object to export.
            **kwargs: Additional options:
                - filename: Override the filename (ignores template).
                - metadata: Dict of custom metadata to attach to the object.

        Returns:
            ExportResult with the S3 URI and export metadata.

        Raises:
            ExportError: If upload fails.
        """
        self.validate(data)

        try:
            client = self._get_client()

            # Generate filename
            filename = kwargs.get("filename") or self._generate_filename(data)
            key = f"{self.prefix}{filename}"

            # Serialize data
            json_data = json.dumps(data.model_dump(), ensure_ascii=False, indent=2)
            bytes_data = json_data.encode("utf-8")

            # Prepare upload parameters
            upload_kwargs = {
                "Bucket": self.bucket,
                "Key": key,
                "Body": bytes_data,
                "ContentType": self.content_type,
            }

            # Add custom metadata if provided
            if metadata := kwargs.get("metadata"):
                upload_kwargs["Metadata"] = {
                    str(k): str(v) for k, v in metadata.items()
                }

            # Upload
            client.put_object(**upload_kwargs)

            s3_uri = f"s3://{self.bucket}/{key}"

            return ExportResult(
                success=True,
                destination=s3_uri,
                bytes_written=len(bytes_data),
                metadata={
                    "bucket": self.bucket,
                    "key": key,
                    "region": self.region,
                    "content_type": self.content_type,
                    "sessions_count": len(data.sessions),
                    "events_count": len(data.events),
                    "function_events_count": len(data.function_events),
                },
            )

        except ExportError:
            raise
        except Exception as e:
            raise ExportError(f"Failed to export to S3: {e}", cause=e)

    @classmethod
    def from_env(
        cls,
        bucket_env: str = "AIOBS_S3_BUCKET",
        region_env: str = "AIOBS_S3_REGION",
        prefix_env: str = "AIOBS_S3_PREFIX",
        access_key_env: str = "AWS_ACCESS_KEY_ID",
        secret_key_env: str = "AWS_SECRET_ACCESS_KEY",
    ) -> "S3Exporter":
        """Create an S3Exporter from environment variables.

        Args:
            bucket_env: Environment variable name for bucket.
            region_env: Environment variable name for region.
            prefix_env: Environment variable name for prefix.
            access_key_env: Environment variable name for AWS access key ID.
            secret_key_env: Environment variable name for AWS secret access key.

        Returns:
            Configured S3Exporter instance.

        Raises:
            ExportError: If required environment variables are missing.
        """
        bucket = os.getenv(bucket_env)
        if not bucket:
            raise ExportError(f"Environment variable {bucket_env} is required")

        return cls(
            bucket=bucket,
            region=os.getenv(region_env, "us-east-1"),
            prefix=os.getenv(prefix_env, ""),
            aws_access_key_id=os.getenv(access_key_env),
            aws_secret_access_key=os.getenv(secret_key_env),
        )

