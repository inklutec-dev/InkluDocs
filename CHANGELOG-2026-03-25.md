# InkluDocs Changelog – 25. Maerz 2026

Erstellt von: Claude Code (Steves Mac)

---

## Fix: Bild-Download schlaegt fehl bei Websites mit Hotlink-Protection (403 Forbidden)

### Problem
Stephan Raithel (LEUCHTKRAFT) meldete, dass der Website-Scan fuer dc-tischlermeister.de
keine Bilder herunterladen konnte. Fehlermeldung: "Keine Bilder konnten heruntergeladen werden".

### Ursache
Viele WordPress-Websites (besonders mit Sicherheits-Plugins wie "All In One WP Security"
oder Cloudflare) blockieren Bild-Downloads, wenn der HTTP-Request keinen Referer-Header
oder einen verdaechtigen User-Agent hat. Das nennt sich "Hotlink Protection".

Unser Code hat die HTML-Seite zwar mit einem User-Agent geladen, aber beim anschliessenden
Bild-Download wurde ein neuer HTTP-Client OHNE jegliche Header erstellt. Der Webserver
der Schreinerei hat das als Bot erkannt und alle Bilder mit 403 Forbidden blockiert.

### Fix (backend/main.py)
Beim Bild-Download werden jetzt Browser-aehnliche HTTP-Headers mitgeschickt:
- **Referer:** Die URL der gescannten Seite (zeigt dem Server: "ich komme von deiner Seite")
- **User-Agent:** Chrome-Browser-String (wie ein normaler Besucher)
- **Accept:** Standard-Browser-Accept-Header fuer Bilder

Zusaetzlich wurde der User-Agent beim Seiten-Fetch (HTML laden) ebenfalls auf einen
Browser-String umgestellt, da manche Websites auch die HTML-Seite fuer Bots blockieren.

### Auswirkung auf bestehende Funktionalitaet
Keine. Der Fix aendert NUR die HTTP-Headers beim Download. Die gesamte Pipeline dahinter
(Bildanalyse, KI-Beschreibung, Kontext-Engine, Export) bleibt komplett unveraendert.
Websites die vorher funktioniert haben, funktionieren weiterhin – der Referer-Header
wird von Servern ohne Hotlink-Protection einfach ignoriert.

### Test-Ergebnis

MIT Browser-Headers (neuer Code):
- 200  7582 bytes   kachel-privatkunden.jpg
- 200  69534 bytes  schreinerei1_750-731x1024.jpg
- 200  6669 bytes   kroell-cremerius_logo.svg

OHNE Headers (alter Code):
- 403  239 bytes    kachel-privatkunden.jpg
- 403  239 bytes    schreinerei1_750-731x1024.jpg
- 403  239 bytes    kroell-cremerius_logo.svg

Regressions-Test Wikipedia: alle Bilder weiterhin 200 OK.

### Geaenderte Datei
- backend/main.py (Zeile ~733): httpx.AsyncClient bekommt headers-Parameter
- backend/main.py (Zeile ~698): User-Agent beim Seiten-Fetch aktualisiert
- Backup: backend/main.py.bak-2026-03-25

### Fuer Nexus
- Der Fix ist deployed (Container neu gebaut und gestartet)
- Stephan Raithel kann dc-tischlermeister.de jetzt nochmal scannen
- Falls aehnliche 403-Fehler bei anderen Seiten auftreten: Logs pruefen
  (docker logs inkludocs | grep 403) – moeglicherweise brauchen manche
  Seiten zusaetzlich Cookies oder JavaScript-Rendering (anderes Problem)
