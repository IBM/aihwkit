# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

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
