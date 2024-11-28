# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Exceptions related to the cloud client."""

from urllib.parse import urlparse

from requests import Response

from aihwkit.exceptions import CloudError


class ResponseError(CloudError):
    """Error retrieving a response."""

    def __init__(self, response: Response):
        self.response = response
        self.url = self._sanitize_url(response.url)

        super().__init__(str(self))

    def __str__(self) -> str:
        return "{} {} for url: {} {}".format(
            self.response.status_code, self.response.reason, self.response.request.method, self.url
        )

    @staticmethod
    def _sanitize_url(url: str) -> str:
        """Remove sensitive parts from an url."""
        return url


class ApiResponseError(ResponseError):
    """Error retrieving a response (object storage)."""

    @staticmethod
    def _sanitize_url(url: str) -> str:
        """Remove sensitive parts from an url."""
        parts = urlparse(url)

        return "{}{}".format(parts.path, "?..." if parts.query else "")


class InvalidResponseFieldError(CloudError):
    """Invalid or unsupported field in the response."""


class ExperimentStatusError(CloudError):
    """Error dependent on to the Experiment status."""


class CredentialsError(CloudError):
    """Errors related to cloud credentials."""
