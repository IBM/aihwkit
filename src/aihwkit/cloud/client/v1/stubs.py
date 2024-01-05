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

"""API stubs for the AIHW Composer API."""

from collections import namedtuple
from typing import Dict

from aihwkit.cloud.client.session import ApiSession


Endpoint = namedtuple("Endpoint", ["url", "method"])


class ApiStub:
    """Base API stub for the AIHW Composer.

    API stub for use in the client for the AIHW Composer in order to interact
    with REST endpoints.

    Subclasses should inherit from this class, customizing ``base_url`` and
    applying any extra changes. By default, the stub assumes three operations
    for the entities:

        * ``get`` (via ``self.get()``): a ``GET`` operation returning a single
          object from an id.
        * ``post`` (via ``self.post()``): a ``POST`` operation.
        * ``list`` (via ``self.list()``): a ``GET`` operation returning
          multiple objects.
    """

    base_url = ""
    """Base url to be used in the endpoints."""

    def __init__(self, session: ApiSession):
        """Create a new ``ApiStub``.

        Args:
            session: the requests session to be used.
        """
        self.session = session
        self.endpoints = self._endpoint_map()

    def _endpoint_map(self) -> Dict[str, Endpoint]:
        """Generate the mappings to the endpoints.

        Returns:
            A dictionary of strings to endpoints.
        """
        return {
            "list": Endpoint(self.base_url, "GET"),
            "get": Endpoint("{}/{{}}".format(self.base_url), "GET"),
            "post": Endpoint(self.base_url, "POST"),
        }

    def list(self) -> Dict:
        """Return a list of entities.

        Returns:
            A list of entities.
        """
        endpoint = self.endpoints["list"]

        return self.session.request(method=endpoint.method, url=endpoint.url).json()

    def get(self, object_id: str) -> Dict:
        """Return a single entity by id.

        Args:
            object_id: the id of the entity.

        Returns:
            A dictionary with the entity.
        """
        endpoint = self.endpoints["get"]

        return self.session.request(
            method=endpoint.method, url=endpoint.url.format(object_id)
        ).json()

    def post(self, content: Dict) -> Dict:
        """Create a single entity.

        Args:
            content: the content of the entity.

        Returns:
            A dictionary with the API response.
        """
        endpoint = self.endpoints["post"]

        return self.session.request(method=endpoint.method, url=endpoint.url, json=content).json()


class ExperimentStub(ApiStub):
    """Stub for ``experiment``."""

    base_url = "experiments"

    def _endpoint_map(self) -> Dict[str, Endpoint]:
        """Generate the mappings to the endpoints.

        Returns:
            A dictionary of strings to endpoints.
        """
        ret = super()._endpoint_map()

        # Use a different url for listing.
        ret["list"] = Endpoint("{}/me".format(self.base_url), "GET")

        return ret


class InputStub(ApiStub):
    """Stub for ``input``."""

    base_url = "inputs"

    def _endpoint_map(self) -> Dict[str, Endpoint]:
        """Generate the mappings to the endpoints.

        Returns:
            A dictionary of strings to endpoints.
        """
        ret = super()._endpoint_map()

        # Use a different url for getter.
        ret["get"] = Endpoint("{}/{{}}/file".format(self.base_url), "GET")

        return ret


class OutputStub(ApiStub):
    """Stub for ``output``."""

    base_url = "outputs"

    def _endpoint_map(self) -> Dict[str, Endpoint]:
        """Generate the mappings to the endpoints.

        Returns:
            A dictionary of strings to endpoints.
        """
        ret = super()._endpoint_map()

        # Use a different url for getter.
        ret["get"] = Endpoint("{}/{{}}/file".format(self.base_url), "GET")

        return ret


class JobStub(ApiStub):
    """Stub for ``job``."""

    base_url = "jobs"


class LoginStub(ApiStub):
    """Stub for ``login``."""

    base_url = "token/login"
