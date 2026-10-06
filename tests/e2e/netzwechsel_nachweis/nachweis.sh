#!/bin/bash
# Nachweis (06.10.2026): Ein Netzwechsel auf dem Rechner bricht laufende HTTP/2-Anfragen des Klicktest-Chromiums ab
# (net::ERR_NETWORK_CHANGED) — so wie am 06.10. „docker rm“ bzw. ein neues Docker-Netz die KI-Anfragen von ui_word
# und ui_formular abbrachen. Laeuft komplett in zwei eigenen Netz-Namensraeumen: am Netz des Servers aendert sich
# nichts, kein Staging, keine KI. A = „Server“ mit Chromium, B = „Container“ mit HTTP/2-Server; Zugang wie bei
# Docker per DNAT auf A's eigene Adresse.
# Aufruf (root, braucht ip, iptables, node, Playwright):  sudo bash tests/e2e/netzwechsel_nachweis/nachweis.sh
#   optional INKLUDOCS_E2E_PYTHON_PW=/pfad/python (mit Playwright), PLAYWRIGHT_BROWSERS_PATH, NETLOG=/pfad/netlog.json,
#   OHNE_IPV6=1 (Gegenprobe)
set -u
H="$(cd "$(dirname "$0")" && pwd)"; PY="${INKLUDOCS_E2E_PYTHON_PW:-python3}"
A=nachweis-a-$$; B=nachweis-b-$$; T=$(mktemp -d)
aufraeumen() { kill "${SRV:-0}" 2>/dev/null; ip netns del $A 2>/dev/null; ip netns del $B 2>/dev/null; rm -rf "$T"; }
trap aufraeumen EXIT
openssl req -x509 -newkey rsa:2048 -nodes -keyout "$T/k.pem" -out "$T/c.pem" -days 1 -subj "/CN=nachweis" 2>/dev/null
ip netns add $A; ip netns add $B
# Wie auf dem Server: Docker-Schnittstellen tragen IPv6-Link-lokal-Adressen (fe80::…). Gerade deren Kommen und
# Gehen meldet Chromium als Netzwechsel. Gegenprobe mit OHNE_IPV6=1: dann bricht nichts ab.
for N in $A $B; do
  [ "${OHNE_IPV6:-0}" = 1 ] && ip netns exec $N sysctl -qw net.ipv6.conf.all.disable_ipv6=1 net.ipv6.conf.default.disable_ipv6=1
  ip -n $N link set lo up
done
ip -n $A link add e0 type dummy; ip -n $A addr add 10.9.8.1/24 dev e0; ip -n $A link set e0 up; ip -n $A route add default dev e0
ip -n $A link add br0 type bridge; ip -n $A addr add 10.9.7.1/24 dev br0; ip -n $A link set br0 up
ip -n $A link add vA type veth peer name vB netns $B; ip -n $A link set vA master br0; ip -n $A link set vA up
ip -n $B addr add 10.9.7.2/24 dev vB; ip -n $B link set vB up; ip -n $B route add default via 10.9.7.1
ip netns exec $A sysctl -qw net.ipv4.ip_forward=1
ip netns exec $A iptables -t nat -A OUTPUT -d 10.9.8.1/32 -p tcp --dport 8443 -j DNAT --to-destination 10.9.7.2:8443
ip netns exec $B node "$H/h2server.js" "$T/k.pem" "$T/c.pem" 10.9.7.2 & SRV=$!
sleep 6   # Netz zur Ruhe kommen lassen (IPv6-Adresspruefung der neuen Schnittstellen), bevor Chromium startet
ip netns exec $A env HOME="${HOME:-/root}" "$PY" "$H/abbruch.py" https://10.9.8.1:8443/ ${NETLOG:-}
