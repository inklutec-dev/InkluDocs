"""Fremde Webadressen abrufen, ohne einen Weg ins interne Netz (SSRF) — fuer den Seitenscanner (main.scan_url) und alles, was
ein Nutzer als Adresse angibt.

Pruefung 13.10.2026 (Befund S11). Vorher pruefte main._pruefe_url_sicher den Namen, und der Abruf loeste ihn danach noch einmal
auf: eine eigene Domain konnte erst eine oeffentliche, dann eine interne Adresse liefern (DNS-Rebinding). Dazu galten Bereiche
wie 100.64.0.0/10 (Carrier-NAT, z. B. Tailscale) nicht als intern, die Groessengrenze griff erst nach dem vollstaendigen Laden,
und die Aufloesung blockierte die Ereignisschleife. Jetzt:

- adresse_erlaubt(ip): nur global erreichbare Unicast-Adressen (ipaddress.is_global), auch hinter IPv4-in-IPv6-Formen.
- client(): ein httpx.AsyncClient, der nur mit einer Adresse verbindet, die die Pruefung IM SELBEN SCHRITT freigegeben hat
  (_PruefendesNetz). TLS prueft weiter gegen den Namen aus der Adresse (SNI und Zertifikat), nicht gegen die IP.
- lesen_bis(antwort, grenze): gestreamt lesen und bei Ueberschreitung abbrechen.
Weiterleitungen folgt der Aufrufer selbst Schritt fuer Schritt (main._safe_fetch); jede neue Verbindung geht durch dieselbe
Pruefung.
"""
from __future__ import annotations

import asyncio
import ipaddress
import os
import socket
from typing import Optional

import httpcore
import httpx
from fastapi import HTTPException

_NAT64 = ipaddress.ip_network("64:ff9b::/96")


class AdresseNichtErlaubt(httpx.ConnectError):
    """Die Adresse fuehrt ins interne Netz (oder laesst sich nicht aufloesen)."""


def adresse_erlaubt(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(addr, ipaddress.IPv6Address):
        eingebettet = addr.ipv4_mapped or addr.sixtofour or (addr.teredo[1] if addr.teredo else None)
        if eingebettet is None and addr in _NAT64:
            eingebettet = ipaddress.IPv4Address(int(addr) & 0xFFFFFFFF)
        if eingebettet is not None:
            addr = eingebettet
    # is_global allein laesst Multicast (224.0.0.0/4) durch
    return addr.is_global and not addr.is_multicast


def _loopback_erlaubt(host: str) -> bool:
    # Testgeschirr (19.08.2026): Pruefseiten von 127.0.0.1 nur mit ausdruecklicher Freigabe (Staging), auf Prod/Demo nie
    return os.environ.get("SCAN_ERLAUBE_LOOPBACK") == "1" and host in ("127.0.0.1", "localhost")


async def erlaubte_adresse(host: str, port: int) -> str:
    """Den Namen aufloesen (ohne die Ereignisschleife zu blockieren) und eine Adresse liefern, mit der verbunden werden darf. Fuehrt
    auch nur EINE der Antworten ins interne Netz, gilt der Name als nicht erlaubt."""
    if _loopback_erlaubt(host):
        return "127.0.0.1"
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as e:
        raise AdresseNichtErlaubt(f"Adresse nicht aufloesbar: {host}") from e
    adressen = [info[4][0] for info in infos]
    if not adressen or not all(adresse_erlaubt(a) for a in adressen):
        raise AdresseNichtErlaubt(f"Adresse nicht erlaubt: {host}")
    return adressen[0]


class _PruefendesNetz(httpcore.AsyncNetworkBackend):
    """Netzwerk-Schicht des Clients: jede Verbindung nur zu einer gerade geprueften Adresse."""

    def __init__(self) -> None:
        self._netz = httpcore.AnyIOBackend()

    async def connect_tcp(self, host: str, port: int, timeout: Optional[float] = None, local_address: Optional[str] = None,
                          socket_options=None) -> httpcore.AsyncNetworkStream:
        ip = await erlaubte_adresse(host, port)
        return await self._netz.connect_tcp(ip, port, timeout=timeout, local_address=local_address, socket_options=socket_options)

    async def connect_unix_socket(self, path: str, timeout: Optional[float] = None, socket_options=None):
        raise AdresseNichtErlaubt("Unix-Sockets sind nicht erlaubt")

    async def sleep(self, seconds: float) -> None:
        await self._netz.sleep(seconds)


class _Transport(httpx.AsyncHTTPTransport):
    """httpx-Transport mit der pruefenden Netzwerk-Schicht. httpx 0.27 (requirements.txt) nimmt sie nicht als Argument; der
    Verbindungs-Pool wird deshalb mit den Vorgaben von httpx (Zertifikatspruefung, Grenzen) und dieser Schicht gebaut.
    tests/test_sicherheit_pruefung_1310.py prueft, dass der Transport sie wirklich nutzt — faellt das bei einem httpx-Update um,
    wird der Test rot statt der Schutz still unwirksam."""

    def __init__(self) -> None:
        super().__init__(trust_env=False)
        grenzen = httpx.Limits(max_connections=100, max_keepalive_connections=20, keepalive_expiry=5.0)
        self._pool = httpcore.AsyncConnectionPool(
            ssl_context=httpx.create_ssl_context(trust_env=False), max_connections=grenzen.max_connections,
            max_keepalive_connections=grenzen.max_keepalive_connections, keepalive_expiry=grenzen.keepalive_expiry,
            network_backend=_PruefendesNetz())


def client(**kwargs) -> httpx.AsyncClient:
    """AsyncClient fuer fremde Adressen: pruefende Verbindungen, keine Proxys aus der Umgebung, Weiterleitungen nur ueber den
    Aufrufer (follow_redirects bleibt aus)."""
    kwargs.pop("follow_redirects", None)
    return httpx.AsyncClient(transport=_Transport(), trust_env=False, follow_redirects=False, **kwargs)


async def lesen_bis(antwort: httpx.Response, grenze: Optional[int]) -> bytes:
    """Den Inhalt einer gestreamten Antwort lesen und abbrechen, sobald er die Grenze ueberschreitet (413)."""
    teile: list[bytes] = []
    n = 0
    async for stueck in antwort.aiter_bytes():
        n += len(stueck)
        if grenze is not None and n > grenze:
            raise HTTPException(status_code=413, detail="Datei zu gross")
        teile.append(stueck)
    return b"".join(teile)
