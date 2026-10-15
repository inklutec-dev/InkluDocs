"""Schnappschuesse des fertigen InkluAgent-Systemprompts (Schubfaecher-Umbau, Steve 14./15.10.2026).

Der Prompt entsteht aus vielen Teilen (Projektart, Kosten und Zustimmung, Hochladen, Konto, Begleiter, Karten, Rueckmeldung,
Auftraege, Demo-Leitplanke), je nach Schalter. Dieser Test baut ihn fuer feste Lagen genau so, wie das Modell ihn bekommt —
ueber agent_loop.run_agent mit einem Ersatzmodell, das nur den Systemprompt mitschreibt — und vergleicht ihn Byte fuer Byte mit
tests/snapshots/agent_prompts/<lage>.md. Jede Aenderung am Prompt zeigt sich damit als Diff der Schnappschuesse.

Schnappschuesse neu schreiben (nur bewusst, nach einer gewollten Prompt-Aenderung):
    AGENT_PROMPTS_SCHREIBEN=1 python3 -m unittest tests/test_agent_prompt_snapshots.py

Alle Schalter werden je Lage ausdruecklich gesetzt; die Umgebung des Rechners spielt keine Rolle. Wegwerf-Datenbank (/tmp),
fiktives Konto, kein Modell, keine Mails.
"""
import os
import sys
import tempfile
import time
import unittest
from unittest import mock

TMP = tempfile.mkdtemp(prefix="agent_prompts_")
os.environ["INKLUDOCS_DB"] = os.path.join(TMP, "test.db")
HIER = os.path.dirname(os.path.abspath(__file__))
for kandidat in (os.path.normpath(os.path.join(HIER, "..", "backend")), "/app"):
    if os.path.isdir(kandidat) and kandidat not in sys.path:
        sys.path.insert(0, kandidat)

import database  # noqa: E402
database.init_db()

import funktionen  # noqa: E402
from inkluagent import agent_loop, begleiter, storage  # noqa: E402

try:
    import stripe_zahlung  # noqa: E402
except Exception:  # noqa: BLE001
    stripe_zahlung = None

ORDNER = os.path.join(HIER, "snapshots", "agent_prompts")
SCHREIBEN = (os.environ.get("AGENT_PROMPTS_SCHREIBEN") or "").strip() in ("1", "an", "ja")

# Alle Schalter, von denen der Systemprompt abhaengt (direkt oder ueber an()-Funktionen der Module)
SCHALTER = ("TEXT_ZURUECK", "TAGGING_PROFESSIONELL", "KETTE", "KI_PRUEFUNG", "KORREKTUR", "EXPRESS", "AGENT_SICHERHEIT",
            "AGENT_BILD_WERKZEUGE", "AGENT_HOCHLADEN", "AGENT_HOCHLADEFELD", "AGENT_ARBEITSFENSTER", "AGENT_ANSICHT",
            "AGENT_UEBERALL", "AGENT_KONTO", "AGENT_KONTO_WERKZEUGE", "AGENT_GESPRAECHE", "AGENT_BEGLEITER", "AGENT_KARTEN",
            "AGENT_KAUFEN", "AGENT_RUECKMELDUNG", "AGENT_AUFTRAEGE")
# Staging am 15.10.2026 (docker-compose.staging.yml, .env.staging): alle Agent-Schalter an ausser dem Hochladefeld,
# professionelles Tagging und Express an, Stripe eingerichtet; KI-Pruefung, Korrektur, Kette und Zuruecksetzen sind im Code aus.
PROFILE = {
    "staging": dict({s: s.startswith("AGENT_") for s in SCHALTER}, AGENT_HOCHLADEFELD=False, TAGGING_PROFESSIONELL=True,
                    EXPRESS=True, _stripe=True),
    "alle_an": dict({s: True for s in SCHALTER}, _stripe=True),
    "alle_aus": dict({s: False for s in SCHALTER}, _stripe=False),
}
# Wie Staging, aber ohne Express-Service (so koennte Prod mit Karten aussehen) — Waechter fuer Werkzeuge hinter EXPRESS
PROFILE["staging_ohne_express"] = dict(PROFILE["staging"], EXPRESS=False)
# Projektart -> (project_type, tool); "konto" = Seite ohne Projekt
ARTEN = {"pdf": ("pdf", "pdf"), "word": ("docx", "word"), "grafik": ("images", "grafik"), "web": ("url", "web"),
         "formular": ("pdf", "formular"), "konto": None}
# Lage: (Profil, Projektart, Gespraech des Kontos mit Lauf?, Demo-Leitplanke?, Agentenansicht?)
LAGEN = {f"{art}_{profil}": (profil, art, True, False, False) for profil in PROFILE for art in ARTEN}
LAGEN.update({
    "pdf_demo": ("staging", "pdf", False, True, False),           # Demo: ohne Lauf und Gespraech, mit Leitplanke
    "pdf_staging_agentenansicht": ("staging", "pdf", True, False, True),
    "konto_staging_agentenansicht": ("staging", "konto", True, False, True),
    "pdf_staging_ohne_express": ("staging_ohne_express", "pdf", True, False, False),
})


class _Mitschreiber:
    """Ersatzmodell: schreibt den Systemprompt mit und antwortet ohne Werkzeug."""
    kennt_frist = False
    kann_streamen = False

    def __init__(self):
        self.system = None
        self.werkzeuge = set()

    def invoke_with_tools(self, anthropic_messages, tools, system, **kw):
        self.system = system
        self.werkzeuge = {t["name"] for t in tools}
        return {"content": [{"type": "text", "text": "Gern (fiktiv)."}], "stop_reason": "end_turn"}


def _nutzer(agentenansicht: bool) -> int:
    mail = f"prompt-schnappschuss-{time.time_ns()}@beispiel.invalid"
    uid = database.create_user(mail, "geheim-12345", "Erika Mustermann (fiktiv)", email_verified=1)
    if agentenansicht:
        c = database.get_db()
        c.execute("UPDATE users SET oberflaeche = 'agent' WHERE id = ?", (uid,))
        c.commit()
        c.close()
    return uid


def _projekt(uid: int, art: str):
    if ARTEN[art] is None:
        return None, None
    project_type, tool = ARTEN[art]
    c = database.get_db()
    pid = c.execute("INSERT INTO projects (user_id, filename, original_path, name, tool, project_type, status) "
                    "VALUES (?, 'brief-fiktiv', '', 'Vereinsbrief (fiktiv)', ?, ?, 'extracted')", (uid, tool, project_type)).lastrowid
    c.commit()
    p = dict(c.execute("SELECT * FROM projects WHERE id = ?", (pid,)).fetchone())
    c.close()
    return pid, p


def prompt_der_lage(name: str) -> str:
    """Der Systemprompt, den das Modell in dieser Lage bekommt."""
    return lage_bauen(name)[0]


def lage_bauen(name: str) -> tuple:
    """(Systemprompt, Namen der Werkzeuge), wie das Modell sie in dieser Lage bekommt."""
    profil, art, mit_lauf, demo, agentenansicht = LAGEN[name]
    werte = PROFILE[profil]
    patches = [mock.patch.object(funktionen, s, werte[s]) for s in SCHALTER]
    if stripe_zahlung is not None:
        patches.append(mock.patch.object(stripe_zahlung, "AKTIV", werte["_stripe"]))
    for p in patches:
        p.start()
    try:
        uid = _nutzer(agentenansicht)
        pid, projekt = _projekt(uid, art)
        gid = storage.aktuelles_gespraech(uid) if mit_lauf else None
        lauf = begleiter.Lauf(ui_sprache="de", nutzer_text="Was kannst du hier? (fiktiv)") if mit_lauf else None
        suffix = None
        if demo:
            import demo as demo_mod
            suffix = demo_mod.DEMO_GUARDRAIL
        m = _Mitschreiber()
        agent_loop.run_agent(pid, uid, "Was kannst du hier? (fiktiv)", projekt, m, system_suffix=suffix, konto=True,
                             begleiter_lauf=lauf, gespraech_id=gid)
        return m.system, m.werkzeuge
    finally:
        for p in reversed(patches):
            p.stop()


def alle_werkzeugnamen() -> set:
    """Jedes Werkzeug, das es irgendwo gibt (ungefiltert, auch hinter Schaltern)."""
    from inkluagent import auftraege, begleiter as bg, karten, konto, rueckmeldung
    from inkluagent.tools import definitions, definitions_formular, definitions_oberflaeche, definitions_pdf
    namen = set()
    for mod in (definitions, definitions_formular, definitions_pdf, definitions_oberflaeche, konto):
        for k, v in vars(mod).items():
            if k.startswith("TOOL_DEFINITION"):
                namen |= {d["name"] for d in (v if isinstance(v, list) else [v]) if isinstance(d, dict) and "name" in d}
    return namen | set(bg.NAMEN) | set(karten.NAMEN) | set(rueckmeldung.NAMEN) | set(auftraege.NAMEN)


class Waechter(unittest.TestCase):
    """Widerspruchs-Runde 15.10.2026: der Prompt nennt nur, was es in der Lage gibt."""

    @classmethod
    def setUpClass(cls):
        cls.lagen = {name: lage_bauen(name) for name in sorted(LAGEN)}
        cls.alle = alle_werkzeugnamen()

    def test_kein_werkzeug_das_es_hier_nicht_gibt(self):
        import re
        for name, (system, werkzeuge) in self.lagen.items():
            with self.subTest(lage=name):
                genannt = {w for w in self.alle if re.search(r"(?<!\w)" + re.escape(w) + r"(?!\w)", system)}
                self.assertEqual(sorted(genannt - werkzeuge), [], "Werkzeug im Text, das es in dieser Lage nicht gibt")

    def test_keine_arbeit_hinter_einem_ausgeschalteten_schalter(self):
        for name, (system, werkzeuge) in self.lagen.items():
            profil = PROFILE[LAGEN[name][0]]
            with self.subTest(lage=name):
                if not profil["KI_PRUEFUNG"]:
                    self.assertNotIn("KI-basierte Prüfung", system)
                if not profil["KETTE"]:
                    self.assertNotIn("Komplett barrierefrei", system)

    def test_mit_kauf_karte_kein_pauschales_nie_selbst(self):
        from inkluagent import begleiter as bg
        geprueft = 0
        for name, (system, werkzeuge) in self.lagen.items():
            if "credits_kaufen_vorbereiten" in werkzeuge and "hinfuehren" in werkzeuge:
                geprueft += 1
                with self.subTest(lage=name):
                    self.assertNotIn(bg._NIE_SELBST, system)
                    self.assertIn("Credits nachkaufen oder ein Abo buchen geht im Gespräch nur über die Karte", system)
        self.assertGreater(geprueft, 0, "mindestens eine Lage mit Begleiter und Kauf-Karte")

    def test_keine_werkzeugzahl_und_keine_reine_pdf_identitaet(self):
        for name, (system, _w) in self.lagen.items():
            with self.subTest(lage=name):
                self.assertNotIn("sieben Tools", system)
                self.assertNotIn("Alternativ-Texte in PDF-Dokumenten", system)


class Schnappschuesse(unittest.TestCase):
    def test_jede_lage_byte_gleich(self):
        os.makedirs(ORDNER, exist_ok=True)
        for name in sorted(LAGEN):
            with self.subTest(lage=name):
                text = prompt_der_lage(name)
                self.assertIsInstance(text, str)
                pfad = os.path.join(ORDNER, name + ".md")
                if SCHREIBEN:
                    with open(pfad, "w", encoding="utf-8", newline="") as f:
                        f.write(text)
                    continue
                self.assertTrue(os.path.exists(pfad), f"Schnappschuss fehlt: {pfad} (AGENT_PROMPTS_SCHREIBEN=1)")
                with open(pfad, encoding="utf-8", newline="") as f:
                    erwartet = f.read()
                if text != erwartet:
                    # die erste abweichende Stelle zeigen, nicht 40.000 Zeichen
                    i = next((k for k, (a, b) in enumerate(zip(text, erwartet)) if a != b), min(len(text), len(erwartet)))
                    self.fail(f"{name}: Prompt weicht ab bei Zeichen {i} (Laenge jetzt {len(text)}, Schnappschuss {len(erwartet)}):\n"
                              f"jetzt:       {text[max(0, i - 120):i + 200]!r}\nSchnappschuss: {erwartet[max(0, i - 120):i + 200]!r}")

    def test_keine_verwaisten_schnappschuesse(self):
        if SCHREIBEN or not os.path.isdir(ORDNER):
            return
        uebrig = sorted(f for f in os.listdir(ORDNER) if f.endswith(".md") and f[:-3] not in LAGEN)
        self.assertEqual(uebrig, [], "Schnappschuss ohne Lage — Lage geloescht? Dann die Datei auch")


if __name__ == "__main__":
    unittest.main()
