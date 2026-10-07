"""Ruecksprung nach der Anmeldung (Express Runde 7, 06.10.2026).

Wer ohne Sitzung eine geschuetzte Seite aufruft (typisch: Link aus einer Mail), landet auf /login?weiter=<Pfad> und nach
der Anmeldung wieder dort. Erlaubt sind NUR interne, relative Pfade — sonst waere /login ein offener Redirect
(/login?weiter=https://boese.example). Geprueft wird hier (Server) und noch einmal im Anmeldeformular (index.html).
"""
import posixpath
from urllib.parse import quote, urlsplit

STANDARD = "/app"
MAX_LAENGE = 1000


def sicheres_ziel(weiter) -> str:
    """Interner Pfad aus `weiter` oder "" (dann gilt das Standardziel). Abgelehnt: alles ohne fuehrenden einzelnen
    Schraegstrich, //host, Backslashes, Schema/Host, Steuer- und Leerzeichen, /login selbst (Schleife), zu lang."""
    z = str(weiter or "")
    if not z or len(z) > MAX_LAENGE or not z.startswith("/") or z.startswith("//"):
        return ""
    if "\\" in z or any(ord(c) < 33 or ord(c) == 127 for c in z):
        return ""
    teile = urlsplit(z)
    if teile.scheme or teile.netloc or not teile.path.startswith("/"):
        return ""
    # Pfad normalisieren (Nachkontrolle Runde 7, Hinweis): „/..//host“ wird zu „/host“, nie zu „//host“; „.“ und „..“
    # verschwinden. Ein abschliessender Schraegstrich bleibt erhalten.
    pfad = posixpath.normpath(teile.path)
    if pfad.startswith("//"):
        return ""
    if teile.path.endswith("/") and pfad != "/":
        pfad += "/"
    if pfad.rstrip("/") in ("/login", "/logout", "/api/logout"):
        return ""
    return pfad + (f"?{teile.query}" if teile.query else "") + (f"#{teile.fragment}" if teile.fragment else "")


def login_adresse(pfad: str, abfrage: str = "") -> str:
    """/login mit Ruecksprung auf pfad?abfrage (nur wenn das ein sicheres internes Ziel ist)."""
    ziel = sicheres_ziel(pfad + (f"?{abfrage}" if abfrage else ""))
    if not ziel or ziel in ("/", STANDARD):
        return "/login"
    return "/login?weiter=" + quote(ziel, safe="")
