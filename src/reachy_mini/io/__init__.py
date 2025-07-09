"""Provide the Zenoh client and server as default implementation for the Reachy Mini project."""

from .zenoh_client import ZenohClient
from .zenoh_server import ZenohServer

Client = ZenohClient
Server = ZenohServer
