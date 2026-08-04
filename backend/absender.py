"""Absender-Erkennung: welche IP-Adresse gehoert wirklich zu dieser Anfrage?

Warum es diese Datei gibt (02.08.2026)
--------------------------------------
Vor InkluDocs steht genau EIN eigener Proxy, der nginx-proxy-manager (NPM).
Er setzt die Kopfzeile mit `$proxy_add_x_forwarded_for`, das heisst: er HAENGT
die Adresse, von der die Anfrage bei ihm ankam, RECHTS an das an, was der
Besucher selbst schon mitgeschickt hat. Hinter NPM sieht eine Anfrage so aus:

    X-Forwarded-For: <frei erfunden>, <frei erfunden>, <echte Adresse>
                     '------- vom Besucher --------'  '--- von NPM ---'

Wer den LINKEN Eintrag nimmt, glaubt dem Besucher. Jede Ratenbegrenzung ist
dann mit einer wechselnden Kopfzeile umgehbar — belegt am Kontaktformular
(10 von 10 Einsendungen durch, statt 429 nach der fuenften) und an der
Login-Bremse des Admin (10 Fehlversuche ohne jedes 429). Wer den RECHTEN
Eintrag nimmt, liest genau das, was unser eigener Proxy geschrieben hat, und
der Besucher kann nichts weiter rechts anhaengen: nginx haengt immer zuletzt
an.

Genau ein Proxy-Sprung
----------------------
`INKLUDOCS_PROXY_SPRUENGE` sagt, wie viele eigene Proxys eine Adresse
anhaengen. Bei uns ist das genau einer (NPM), darum die Vorgabe 1: gesucht ist
der letzte Eintrag der Kette. Kaeme spaeter ein weiterer eigener Proxy davor
(z.B. ein CDN), waere es 2 — dann ist der zweitletzte Eintrag der Besucher.
`0` heisst "kein eigener Proxy davor": die Kopfzeile wird dann NIE geglaubt.

Wem wir die Kopfzeile glauben
-----------------------------
Die Kopfzeile wird nur gelesen, wenn die Anfrage ueber einen unserer Proxys
kam. Dafuer gibt es zwei Merkmale:

1. uvicorn hat `scope["client"]` schon selbst ersetzt. Das tut es nur, wenn
   die Gegenstelle in `--forwarded-allow-ips` steht, und es setzt dabei den
   Port immer auf 0 (gemessen mit uvicorn 0.34.0). Port 0 ist also das
   verlaessliche Zeichen "uvicorn hat der Gegenstelle vertraut".
2. Die Gegenstelle liegt in einem der Netze aus `INKLUDOCS_PROXY_NETZE`
   (Rueckfall: `FORWARDED_ALLOW_IPS`, dann die privaten Bereiche). Das greift,
   wenn uvicorn ohne `--proxy-headers` laeuft.

Alles andere — Direktzugriff auf den Container-Port aus einem fremden Netz —
gilt als "kein Proxy": dann zaehlt die echte Gegenstelle, die Kopfzeile wird
verworfen.

WARNUNG: `--forwarded-allow-ips *` macht die Faelschung wieder moeglich, weil
uvicorn dann den LINKEN Eintrag nimmt (gemessen 02.08.2026) und ausserdem jede
Gegenstelle akzeptiert. Diese Datei faengt das ab, weil sie die Kette selbst
liest und immer von rechts zaehlt — aber die Startoption gehoert trotzdem auf
das Proxy-Netz eingeschraenkt. Guertel UND Hosentraeger.
"""

from __future__ import annotations

import ipaddress
import logging
import os

log = logging.getLogger("inkludocs.absender")

# Rueckgabewert, wenn wirklich nichts zu erkennen ist. Bewusst KEINE Adresse:
# ein Platzhalter wie "0.0.0.0" wuerde in Protokollen wie eine echte Angabe
# aussehen. Alle Zaehler behandeln ihn wie einen ganz normalen Schluessel.
UNBEKANNT = "unbekannt"

# Private Bereiche als letzter Rueckfall fuer die Proxy-Netze. Ein eigener
# Proxy liegt praktisch immer auf einer privaten Adresse; ein Besucher aus dem
# Internet nie. Die Vorgabe ist also sicher, aber weiter als noetig — im
# Container schraenkt docker-compose.yml sie auf das NPM-Netz ein.
STANDARD_PROXY_NETZE: tuple[str, ...] = (
    "127.0.0.0/8", "::1/128",
    "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "fc00::/7",
)

# Zaehl-Koerbchen fuer IPv6: ein einzelner Anschluss bekommt heute mindestens
# ein /64 und koennte darin beliebig viele Adressen durchprobieren. Fuer die
# Ratenbegrenzung zaehlt darum das /64, nicht die einzelne Adresse. Fuer
# Protokolle und Archiv bleibt die genaue Adresse erhalten.
IPV6_KORB = 64

_NetzTyp = ipaddress.IPv4Network | ipaddress.IPv6Network


# ---------------------------------------------------------------------------
# Einstellungen lesen
# ---------------------------------------------------------------------------

def lies_spruenge(roh: str | None = None) -> int:
    """Anzahl eigener Proxys, die eine Adresse anhaengen (Vorgabe 1)."""
    wert = (os.environ.get("INKLUDOCS_PROXY_SPRUENGE", "") if roh is None
            else roh).strip()
    if not wert:
        return 1
    try:
        zahl = int(wert)
    except ValueError:
        log.warning("INKLUDOCS_PROXY_SPRUENGE=%r ist keine Zahl — nehme 1",
                    wert)
        return 1
    if zahl < 0:
        log.warning("INKLUDOCS_PROXY_SPRUENGE=%r ist negativ — nehme 1", wert)
        return 1
    return zahl


def netze_aus_text(roh: str) -> tuple[_NetzTyp, ...]:
    """Kommaliste aus Netzen und einzelnen Adressen in Netz-Objekte.

    Einzelne Adressen werden zu /32 bzw. /128. Unbrauchbare Angaben werden
    protokolliert und uebersprungen — eine kaputte Einstellung darf den Dienst
    nicht am Start hindern, aber sie darf auch nicht stillschweigend alles
    vertrauen."""
    netze: list[_NetzTyp] = []
    for teil in (roh or "").split(","):
        teil = teil.strip()
        if not teil:
            continue
        if teil == "*":
            log.warning("Proxy-Netze: '*' ist hier nicht erlaubt (es wuerde "
                        "jede Gegenstelle vertrauen) — Eintrag ignoriert")
            continue
        try:
            netze.append(ipaddress.ip_network(teil, strict=False))
        except ValueError:
            log.warning("Proxy-Netze: %r ist kein Netz und keine Adresse — "
                        "Eintrag ignoriert", teil)
    return tuple(netze)


def lies_netze(umgebung: dict[str, str] | None = None) -> tuple[_NetzTyp, ...]:
    """Netze, aus denen wir X-Forwarded-For glauben.

    Reihenfolge: `INKLUDOCS_PROXY_NETZE`, dann uvicorns eigenes
    `FORWARDED_ALLOW_IPS` (damit die Angabe nur an EINER Stelle stehen muss),
    dann die privaten Bereiche."""
    quelle = umgebung if umgebung is not None else os.environ
    for name in ("INKLUDOCS_PROXY_NETZE", "FORWARDED_ALLOW_IPS"):
        roh = (quelle.get(name) or "").strip()
        if not roh:
            continue
        if roh == "*":
            log.warning(
                "%s=* — uvicorn nimmt damit den LINKEN, faelschbaren Eintrag "
                "von X-Forwarded-For. Die Absender-Erkennung dieser Datei "
                "bleibt korrekt (sie zaehlt von rechts), aber die Startoption "
                "gehoert auf das Proxy-Netz eingeschraenkt.", name)
            continue
        netze = netze_aus_text(roh)
        if netze:
            return netze
        log.warning("%s=%r ergab kein brauchbares Netz — nehme die privaten "
                    "Bereiche", name, roh)
        break
    return netze_aus_text(", ".join(STANDARD_PROXY_NETZE))


PROXY_SPRUENGE: int = lies_spruenge()
PROXY_NETZE: tuple[_NetzTyp, ...] = lies_netze()


def einstellungen_text() -> str:
    """Eine Zeile fuer das Startprotokoll — was gilt gerade?"""
    return (f"Absender-Erkennung: {PROXY_SPRUENGE} Proxy-Sprung(=Eintrag von "
            f"rechts), vertraute Proxy-Netze: "
            f"{', '.join(str(n) for n in PROXY_NETZE)}")


# ---------------------------------------------------------------------------
# Bausteine (rein, ohne Request — darum leicht zu pruefen)
# ---------------------------------------------------------------------------

def als_ip(roh: str | None) -> str:
    """Normalisierte Adresse als Text, oder "" wenn es keine Adresse ist.

    Normalisiert absichtlich: `2001:db8::0001` und `2001:db8::1` sind dieselbe
    Adresse und muessen im selben Zaehler landen, sonst waere die
    Ratenbegrenzung durch andere Schreibweisen umgehbar. Erkennt zusaetzlich
    die Formen, die Proxys manchmal schreiben: `[::1]`, `[::1]:443`,
    `1.2.3.4:5678`, `fe80::1%eth0` und die IPv4-in-IPv6-Schreibweise
    `::ffff:1.2.3.4`."""
    wert = (roh or "").strip()
    if not wert:
        return ""
    if wert.startswith("["):                      # [2001:db8::1] / [::1]:443
        ende = wert.find("]")
        if ende < 1:
            return ""
        wert = wert[1:ende]
    elif wert.count(":") == 1 and "." in wert:     # 1.2.3.4:5678
        wert = wert.split(":", 1)[0]
    if "%" in wert:                                # fe80::1%eth0
        wert = wert.split("%", 1)[0]
    try:
        adresse = ipaddress.ip_address(wert)
    except ValueError:
        return ""
    if adresse.version == 6 and adresse.ipv4_mapped:
        return str(adresse.ipv4_mapped)
    return str(adresse)


def kette_aus_header(*rohwerte: str | None) -> list[str]:
    """Alle brauchbaren Adressen der X-Forwarded-For-Kette, links nach rechts.

    Mehrere Kopfzeilen gleichen Namens werden hintereinander gehaengt (nginx
    fasst sie zwar zu einer zusammen, aber wir wollen uns nicht darauf
    verlassen). Muell fliegt raus: leere Felder, `unknown`, Rechnernamen,
    Unsinn. Das ist sicher, weil unser Proxy seine Adresse immer ZULETZT
    anhaengt — Muell kann also nur links vom echten Eintrag stehen."""
    kette: list[str] = []
    for rohwert in rohwerte:
        for teil in (rohwert or "").split(","):
            ip = als_ip(teil)
            if ip:
                kette.append(ip)
    return kette


def darf_kopfzeile_glauben(peer: str, peer_port: int | None,
                          netze: tuple[_NetzTyp, ...] | None = None) -> bool:
    """Kam die Anfrage ueber einen unserer Proxys?

    Port 0 = uvicorns ProxyHeadersMiddleware hat `scope["client"]` ersetzt;
    das tut sie nur fuer Gegenstellen aus `--forwarded-allow-ips`. Sonst muss
    die Gegenstelle selbst in einem Proxy-Netz liegen. Eine Gegenstelle, die
    keine IP-Adresse ist (Unix-Socket, Testclient), gilt bewusst als NICHT
    vertraut: dann zaehlen alle Anfragen in einem Topf, was hoechstens
    unbequem, aber nie unsicher ist."""
    if peer_port == 0:
        return True
    if not peer:
        return False
    try:
        adresse = ipaddress.ip_address(peer)
    except ValueError:
        return False
    return any(adresse in netz
               for netz in (PROXY_NETZE if netze is None else netze))


def echte_ip(peer_host: str | None, peer_port: int | None,
             xff: str | list[str] | tuple[str, ...] | None,
             spruenge: int | None = None,
             netze: tuple[_NetzTyp, ...] | None = None) -> str:
    """Die echte Besucher-Adresse — Kernstueck, ohne FastAPI/Starlette.

    Hinter unserem Proxy ist das der `spruenge`-letzte Eintrag der Kette;
    ohne brauchbare Kette (oder ohne Proxy davor) die Gegenstelle selbst;
    und wenn auch die nichts hergibt, `UNBEKANNT`."""
    schritte = PROXY_SPRUENGE if spruenge is None else spruenge
    peer = als_ip(peer_host)
    kette = kette_aus_header(*(xff if isinstance(xff, (list, tuple))
                               else (xff,)))
    if (schritte >= 1 and len(kette) >= schritte
            and darf_kopfzeile_glauben(peer, peer_port, netze)):
        return kette[-schritte]
    return peer or UNBEKANNT


def korb(ip: str) -> str:
    """Zaehl-Schluessel zu einer Adresse: IPv4 genau, IPv6 als /64-Netz."""
    try:
        adresse = ipaddress.ip_address(ip)
    except ValueError:
        return ip or UNBEKANNT
    if adresse.version == 6:
        return str(ipaddress.ip_network(f"{adresse}/{IPV6_KORB}",
                                        strict=False))
    return str(adresse)


# ---------------------------------------------------------------------------
# Anschluss an Starlette/FastAPI
# ---------------------------------------------------------------------------

def _gegenstelle(request) -> tuple[str | None, int | None]:
    stelle = getattr(request, "client", None)
    if stelle is None:
        return None, None
    return getattr(stelle, "host", None), getattr(stelle, "port", None)


def _xff_kopfzeilen(request) -> list[str]:
    kopf = getattr(request, "headers", None)
    if kopf is None:
        return []
    holen = getattr(kopf, "getlist", None)
    if callable(holen):
        return list(holen("x-forwarded-for"))
    wert = kopf.get("x-forwarded-for")
    return [wert] if wert else []


def absender_ip(request) -> str:
    """Echte Adresse des Besuchers — fuer Protokolle und Anzeige."""
    host, port = _gegenstelle(request)
    return echte_ip(host, port, _xff_kopfzeilen(request))


def limit_schluessel(request) -> str:
    """Schluessel fuer JEDE Ratenbegrenzung (slowapi `key_func` und der
    eigene Zaehler des Renderers). Verschiedene echte Netze bleiben
    getrennt; verschiedene Adressen im selben IPv6-/64 nicht."""
    return korb(absender_ip(request))


def protokolliere_einstellungen() -> None:
    """Beim Start eine Zeile ins Container-Protokoll — damit im Betrieb ohne
    Code-Lesen nachvollziehbar ist, welche Vertrauensgrenze gerade gilt.
    Geht bewusst ueber uvicorns eigenen Logger: nur der ist im Container
    eingerichtet, ein eigener INFO-Satz wuerde sonst verschluckt."""
    logging.getLogger("uvicorn.error").info("%s", einstellungen_text())
    log.info("%s", einstellungen_text())


protokolliere_einstellungen()
