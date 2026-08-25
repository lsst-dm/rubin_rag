#
# This file is part of rubin_rag.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Weaviate client factory.

Routes to the correct ``weaviate.connect_to_*`` function based on
``config["weaviate"]["connection_mode"]``.

Supported modes
---------------
custom
    Self-hosted server (e.g. Kubernetes). Reads ``http_host``,
    ``http_port``, ``http_secure``, ``grpc_host``, ``grpc_port``,
    ``grpc_secure`` from config. Uses ``WEAVIATE_API_KEY`` env var for
    auth if set.
weaviate_cloud
    Weaviate Cloud. Reads ``cluster_url`` from config. Requires
    ``WEAVIATE_API_KEY`` env var.
local
    Local Docker. Reads optional ``port`` and ``grpc_port`` from config
    (defaults: 8080 / 50051). No auth required.
"""

import os
import sys
from pathlib import Path

import weaviate
from dotenv import load_dotenv
from weaviate.classes.init import AdditionalConfig, Auth, Timeout

from ...utils import load_config

# Re-export so callers can build AdditionalConfig without a separate import.
__all__ = ["AdditionalConfig", "Timeout", "connect"]

_DEFAULT_CLOUD_ADDITIONAL_CONFIG = AdditionalConfig(
    timeout=Timeout(init=30, query=60, insert=120)
)


def connect(
    config: dict,
    headers: dict[str, str] | None = None,
    additional_config: AdditionalConfig | None = None,
) -> weaviate.WeaviateClient:
    """Create and return an open Weaviate client.

    Parameters
    ----------
    config : dict
        Top-level config dict. Must contain a ``weaviate`` key with at
        minimum ``connection_mode``.
    headers : dict[str, str] or None
        Extra HTTP headers forwarded to every request (e.g. inference API
        keys). Passed to all connection modes.
    additional_config : AdditionalConfig or None
        Weaviate connection tuning (timeouts, proxies, gRPC config, …).
        Passed to all connection modes. ``weaviate_cloud`` applies a
        default timeout if ``None`` is passed. Construct with
        ``AdditionalConfig`` re-exported from this module.

    Returns
    -------
    weaviate.WeaviateClient
        An open client. The caller is responsible for closing it (or use
        it as a context manager).

    Raises
    ------
    ValueError
        If ``connection_mode`` is unknown or required config keys are
        missing.
    """
    load_dotenv(override=True)
    weaviate_cfg = config["weaviate"]
    mode = weaviate_cfg["connection_mode"]

    if mode == "custom":
        return _connect_custom(
            weaviate_cfg, headers=headers, additional_config=additional_config
        )
    elif mode == "weaviate_cloud":
        return _connect_weaviate_cloud(
            weaviate_cfg, headers=headers, additional_config=additional_config
        )
    elif mode == "local":
        return _connect_local(
            weaviate_cfg, headers=headers, additional_config=additional_config
        )
    else:
        raise ValueError(
            f"Unknown weaviate connection_mode '{mode}'. "
            "Must be one of: custom, weaviate_cloud, local."
        )


def _connect_custom(
    weaviate_cfg: dict,
    headers: dict[str, str] | None = None,
    additional_config: AdditionalConfig | None = None,
) -> weaviate.WeaviateClient:
    api_key = os.getenv("WEAVIATE_API_KEY")
    auth = Auth.api_key(api_key) if api_key else None
    return weaviate.connect_to_custom(
        http_host=weaviate_cfg["http_host"],
        http_port=weaviate_cfg["http_port"],
        http_secure=weaviate_cfg["http_secure"],
        grpc_host=weaviate_cfg["grpc_host"],
        grpc_port=weaviate_cfg["grpc_port"],
        grpc_secure=weaviate_cfg["grpc_secure"],
        auth_credentials=auth,
        headers=headers,
        additional_config=additional_config,
    )


def _connect_weaviate_cloud(
    weaviate_cfg: dict,
    headers: dict[str, str] | None = None,
    additional_config: AdditionalConfig | None = None,
) -> weaviate.WeaviateClient:
    api_key = os.getenv("WEAVIATE_API_KEY")
    if not api_key:
        raise ValueError(
            "WEAVIATE_API_KEY environment variable is required for "
            "weaviate_cloud connection mode."
        )
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_cfg["cluster_url"],
        auth_credentials=Auth.api_key(api_key),
        headers=headers,
        additional_config=additional_config
        or _DEFAULT_CLOUD_ADDITIONAL_CONFIG,
    )


def _connect_local(
    weaviate_cfg: dict,
    headers: dict[str, str] | None = None,
    additional_config: AdditionalConfig | None = None,
) -> weaviate.WeaviateClient:
    return weaviate.connect_to_local(
        port=weaviate_cfg.get("port", 8080),
        grpc_port=weaviate_cfg.get("grpc_port", 50051),
        headers=headers,
        additional_config=additional_config,
    )


if __name__ == "__main__":
    config = load_config(Path(sys.argv[1]))
    client = connect(config)
    try:
        print(client.is_ready())  # noqa: T201
    finally:
        client.close()
