"""Teil von nachweis.sh: Chromium (wie im Klicktest) schickt je Szenario eine 25-s-Anfrage ueber HTTP/2;
2 s danach aendert sich das Netz des Rechners. Ausgabe: Ergebnis der Anfrage und Abbruchgrund."""
import subprocess, sys, time
from playwright.sync_api import sync_playwright
SZENARIEN = [
    ("ohne Netzwechsel", None),
    ("Bruecke + veth kommen dazu (wie docker network create + Container-Start)",
     "ip link add br9 type bridge && ip addr add 10.9.9.1/24 dev br9 && ip link set br9 up && "
     "ip link add v9 type veth peer name v9b && ip link set v9 master br9 && ip link set v9 up && ip link set v9b up"),
    ("veth verschwindet (wie docker rm)", "ip link del v9"),
    ("Bruecke verschwindet (wie docker network rm)", "ip link del br9"),
]
with sync_playwright() as p:
    args = [f"--log-net-log={sys.argv[2]}", "--net-log-capture-mode=Everything"] if len(sys.argv) > 2 else []
    br = p.chromium.launch(args=args)
    pg = br.new_context(ignore_https_errors=True).new_page()
    abbrueche = []
    pg.on("requestfailed", lambda r: abbrueche.append(r.failure))
    pg.goto(sys.argv[1])
    print("Protokoll der Verbindung:", pg.evaluate("performance.getEntriesByType('navigation')[0].nextHopProtocol"))
    for i, (name, befehl) in enumerate(SZENARIEN):
        abbrueche.clear()
        pg.evaluate(f"window._r{i} = (async () => {{ const t0 = performance.now(); try {{ const r = await fetch('/langsam?{i}', {{method: 'POST'}}); "
                    f"return 'HTTP ' + r.status + ' nach ' + Math.round(performance.now() - t0) + ' ms'; }} catch (e) {{ "
                    f"return 'FEHLER ' + e.message + ' nach ' + Math.round(performance.now() - t0) + ' ms'; }} }})()")
        time.sleep(2)
        if befehl:
            subprocess.run(befehl, shell=True, check=True)
        print(f"- {name}: {pg.evaluate(f'window._r{i}')}" + (f", Abbruchgrund {abbrueche}" if abbrueche else ""))
    br.close()
