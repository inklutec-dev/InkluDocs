"""Fremde Webadressen abrufen, ohne einen Weg ins interne Netz (SSRF) — fuer den Seitenscanner (main.scan_url) und alles, was
ein Nutzer als Adresse angibt.

Pruefung 13.10.2026 (Befund S11). Vorher pruefte main._pruefe_url_sicher den Namen, und der Abruf loeste ihn danach noch einmal
auf: eine eigene Domain konnte erst eine oeffentliche, dann eine interne Adresse liefern (DNS-Rebinding). Dazu galten Bereiche
wie 100.64.0.0/10 (Carrier-NAT, z. B. Tailscale) nicht als intern, die Groessengrenze griff erst nach dem vollstaendigen Laden,
und die Aufloesung blockierte die Ereignisschleife. Jetzt:

- adresse_erlaubt(ip): nur global erreichbare Unicast-Adressen (ipaddress.is_global), auch hinter IPv4-in-IPv6-Formen. IPv6
  nur aus 2000::/3 (globaler Unicast); IPv4-kompatible (::/96), -abgebildete (::ffff:0:0/96), -uebersetzte (::ffff:0:0:0/96),
  NAT64 (64:ff9b::/96), 6to4 und Teredo zaehlen als die IPv4-Adresse darin (Runde 2, S-N7: ::7f00:1 = 127.0.0.1, fec0::/10
  und andere galten je nach Python-Version als global).
- client(): ein httpx.AsyncClient, der nur mit Adressen verbindet, die die Pruefung IM SELBEN SCHRITT freigegeben hat
  (_PruefendesNetz), der Reihe nach, bis eine antwortet (Runde 2, S-N12: vorher nur die erste — war das IPv6 und nicht
  erreichbar, scheiterte der Abruf). TLS prueft weiter gegen den Namen aus der Adresse (SNI und Zertifikat), nicht gegen die IP.
- lesen_bis(antwort, grenze): gestreamt lesen, selbst entpacken und nie mehr als bis zur Grenze (Runde 2, S-N12: httpx entpackte
  ein Stueck von 64 KiB vor jeder Pruefung auf bis zu 64 MB).
Weiterleitungen folgt der Aufrufer selbst Schritt fuer Schritt (main._safe_fetch); jede neue Verbindung geht durch dieselbe
Pruefung.
"""
from __future__ import annotations

import asyncio
import ipaddress
import os
import socket
import zlib
from typing import Optional

import httpcore
import httpx
from fastapi import HTTPException

_IPV6_GLOBAL = ipaddress.ip_network("2000::/3")
# Formen, deren untere 32 Bit eine IPv4-Adresse sind: NAT64 (RFC 6052), IPv4-uebersetzt (RFC 2765), IPv4-kompatibel (RFC 4291,
# veraltet — dazu gehoeren auch :: und ::1, die so als 0.0.0.0 und 0.0.0.1 gesperrt bleiben)
_IPV4_UNTEN = tuple(ipaddress.ip_network(n) for n in ("64:ff9b::/96", "::ffff:0:0:0/96", "::/96"))


class AdresseNichtErlaubt(httpx.ConnectError):
    """Die Adresse fuehrt ins interne Netz (oder laesst sich nicht aufloesen)."""


def adresse_erlaubt(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(addr, ipaddress.IPv6Address):
        eingebettet = addr.ipv4_mapped or addr.sixtofour or (addr.teredo[1] if addr.teredo else None)
        if eingebettet is None and any(addr in netz for netz in _IPV4_UNTEN):
            eingebettet = ipaddress.IPv4Address(int(addr) & 0xFFFFFFFF)
        if eingebettet is not None:
            addr = eingebettet
        elif addr not in _IPV6_GLOBAL:   # site-lokal (fec0::/10), ULA, link-lokal, Multicast und alles Kuenftige
            return False
    # is_global allein laesst Multicast (224.0.0.0/4) durch
    return addr.is_global and not addr.is_multicast


def _loopback_erlaubt(host: str) -> bool:
    # Testgeschirr (19.08.2026): Pruefseiten von 127.0.0.1 nur mit ausdruecklicher Freigabe (Staging), auf Prod/Demo nie
    return os.environ.get("SCAN_ERLAUBE_LOOPBACK") == "1" and host in ("127.0.0.1", "localhost")


async def erlaubte_adressen(host: str, port: int) -> list[str]:
    """Den Namen aufloesen (ohne die Ereignisschleife zu blockieren) und die Adressen liefern, mit denen verbunden werden darf, in
    der Reihenfolge des Systems und ohne Doppelte. Fuehrt auch nur EINE der Antworten ins interne Netz, gilt der Name als nicht
    erlaubt."""
    if _loopback_erlaubt(host):
        return ["127.0.0.1"]
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as e:
        raise AdresseNichtErlaubt(f"Adresse nicht aufloesbar: {host}") from e
    adressen = list(dict.fromkeys(info[4][0] for info in infos))
    if not adressen or not all(adresse_erlaubt(a) for a in adressen):
        raise AdresseNichtErlaubt(f"Adresse nicht erlaubt: {host}")
    return adressen


async def erlaubte_adresse(host: str, port: int) -> str:
    """Die erste erlaubte Adresse (fuer eine fruehe Pruefung mit verstaendlicher Meldung, main._pruefe_url_sicher)."""
    return (await erlaubte_adressen(host, port))[0]


class _PruefendesNetz(httpcore.AsyncNetworkBackend):
    """Netzwerk-Schicht des Clients: jede Verbindung nur zu einer gerade geprueften Adresse."""

    def __init__(self) -> None:
        self._netz = httpcore.AnyIOBackend()

    async def connect_tcp(self, host: str, port: int, timeout: Optional[float] = None, local_address: Optional[str] = None,
                          socket_options=None) -> httpcore.AsyncNetworkStream:
        letzter: Optional[Exception] = None
        for ip in await erlaubte_adressen(host, port):
            try:
                return await self._netz.connect_tcp(ip, port, timeout=timeout, local_address=local_address,
                                                    socket_options=socket_options)
            except (httpcore.ConnectError, httpcore.ConnectTimeout) as e:   # naechste gepruefte Adresse
                letzter = e
        raise letzter or AdresseNichtErlaubt(f"Keine Verbindung: {host}")

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


class _Entpacker:
    """gzip und deflate Stueck fuer Stueck entpacken, je Stueck hoechstens bis zu einer Laenge (zlib max_length). deflate wie httpx:
    erst mit zlib-Rahmen (RFC 9110), sonst roh."""

    def __init__(self, art: str) -> None:
        self.art = art
        self._z = zlib.decompressobj(16 + zlib.MAX_WBITS if art in ("gzip", "x-gzip") else zlib.MAX_WBITS)
        self._erstes = True

    def stueck(self, daten: bytes, hoechstens: int) -> bytes:
        try:
            out = self._z.decompress(daten, hoechstens)
        except zlib.error as e:
            if not (self.art == "deflate" and self._erstes):
                raise httpx.DecodingError(f"Inhalt nicht entpackbar ({self.art})") from e
            self._z = zlib.decompressobj(-zlib.MAX_WBITS)
            out = self._z.decompress(daten, hoechstens)
        self._erstes = False
        return out

    def rest(self) -> bytes:
        return self._z.flush()


_ENTPACKBAR = ("gzip", "x-gzip", "deflate")


async def lesen_bis(antwort: httpx.Response, grenze: Optional[int]) -> bytes:
    """Den Inhalt einer gestreamten Antwort lesen und abbrechen, sobald er ENTPACKT die Grenze ueberschreitet (413). Entpackt wird
    hier, nicht von httpx: je Stueck nie mehr als ein Byte ueber die Grenze. Andere Kodierungen (identity, Unbekanntes) bleiben
    wie bei httpx unveraendert; mehrere Kodierungen hintereinander werden nicht gelesen."""
    arten = [k.strip().lower() for k in antwort.headers.get_list("content-encoding", split_commas=True)]
    arten = [k for k in arten if k in _ENTPACKBAR]
    if len(arten) > 1:
        raise httpx.DecodingError("Mehrfach kodierter Inhalt wird nicht gelesen")
    entpacker = _Entpacker(arten[0]) if arten else None
    teile: list[bytes] = []
    n = 0

    def aufnehmen(stueck: bytes) -> None:
        nonlocal n
        n += len(stueck)
        if grenze is not None and n > grenze:
            raise HTTPException(status_code=413, detail="Datei zu gross")
        teile.append(stueck)

    async for roh in antwort.aiter_raw():
        # max_length 0 = unbegrenzt; mit Grenze genau ein Byte mehr als noch Platz ist, damit die Ueberschreitung auffaellt
        aufnehmen(entpacker.stueck(roh, 0 if grenze is None else grenze - n + 1) if entpacker else roh)
    if entpacker:
        aufnehmen(entpacker.rest())
    return b"".join(teile)
