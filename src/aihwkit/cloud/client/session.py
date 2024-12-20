# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Session handler for the AIHW Composer API."""

from typing import Any, Text, Union, TYPE_CHECKING

from requests import HTTPError, Session
import urllib3
from urllib3.exceptions import InsecureRequestWarning

from aihwkit.version import __version__
from aihwkit.cloud.client.exceptions import ApiResponseError, ResponseError

if TYPE_CHECKING:
    from typing import Optional


class ObjectStorageSession(Session):
    """Session handler for requests to object storage."""

    def request(  # type: ignore
        self, method: str, url: Union[str, bytes, Text], *args: Any, **kwargs: Any
    ) -> Any:
        """Construct a Request, prepares it and sends it.

        Args:
            method: method for the new ``Request`` object.
            url: URL for the new ``Request`` object.
            args: additional arguments for the original ``requests`` method.
            kwargs: additional arguments for the original ``requests`` method.

        Returns:
            A new ``Response`` object.

        Raises:
            ResponseError: if the response did not have a valid status code.
        """
        # pylint: disable=signature-differs
        response = super().request(method, url, *args, **kwargs)

        try:
            response.raise_for_status()
        except HTTPError:
            raise ResponseError(response) from None

        return response


class ApiSession(Session):
    """Session handler for requests to the AIHW Composer API.

    Custom ``Session`` for interfacing with the AIHW Composer API, using:

    * authorization based on jwt token.
    * custom user agent for the requests.

    Additionally, this class stores information about the API URL and base
    token.
    """

    def __init__(self, api_url: str, api_token: str, verify: bool = True):
        super().__init__()

        self.api_url = api_url
        self.api_token = api_token
        self.verify = verify
        if not verify:
            urllib3.disable_warnings(InsecureRequestWarning)  # type: ignore[no-untyped-call]

        self.jwt_token = None  # type: Optional[str]

        self.headers.update({"User-Agent": "aihwkit/{}".format(__version__)})

    def update_jwt_token(self, jwt_token: str) -> None:
        """Set the jwt token for the session."""
        self.jwt_token = jwt_token
        self.headers.update({"Authorization": "Bearer {}".format(jwt_token)})

    def request(  # type: ignore
        self, method: str, url: Union[str, bytes, Text], *args: Any, **kwargs: Any
    ) -> Any:
        """Construct a Request, prepares it and sends it.

        Args:
            method: method for the new ``Request`` object.
            url: URL for the new ``Request`` object.
            args: additional arguments for the original ``requests`` method.
            kwargs: additional arguments for the original ``requests`` method.

        Returns:
            A new ``Response`` object.

        Raises:
            ApiResponseError: if the response did not have a valid status code.
        """
        # pylint: disable=signature-differs
        full_url = "{}/{}".format(self.api_url, str(url))

        response = super().request(method, full_url, *args, **kwargs)

        try:
            response.raise_for_status()
        except HTTPError:
            raise ApiResponseError(response) from None

        return response
