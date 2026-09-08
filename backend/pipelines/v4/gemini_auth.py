"""Zugang zur Gemini-API: Entwickler-API (Schlüssel) oder Vertex AI (Dienstkonto, EU-Region).

GEMINI_AUTH=key (Vorgabe): Gemini Developer API mit GEMINI_API_KEY über GEMINI_ENDPOINT
    (https://generativelanguage.googleapis.com/v1beta). Für Staging und Messungen.
GEMINI_AUTH=vertex: Vertex AI mit einem Google-Cloud-Dienstkonto, Region festgelegt (EU):
    VERTEX_PROJECT      Google-Cloud-Projekt-ID
    VERTEX_REGION       z. B. europe-west1 (Belgien) oder europe-west3 (Frankfurt)
    VERTEX_REGION_MAP   optional je Modell: gemini-3.5-flash=europe-west3,gemini-2.5-pro=europe-west1
    GOOGLE_SERVICE_ACCOUNT_FILE  Pfad zur JSON-Schlüsseldatei des Dienstkontos (im Container)
    oder GOOGLE_SERVICE_ACCOUNT_JSON_B64 (Base64 der Datei, für Umgebungen ohne Datei)
Das Zugangstoken wird aus dem Dienstkonto per signiertem JWT (RS256, cryptography) beim
OAuth-Token-Endpunkt geholt und bis kurz vor Ablauf wiederverwendet. Keine google-auth-Abhängigkeit.
"""
from __future__ import annotations

import base64
import json
import os
import threading
import time
import urllib.parse
import urllib.request

_lock = threading.Lock()
_token: dict = {'wert': '', 'ablauf': 0.0}


def modus() -> str:
    return 'vertex' if os.environ.get('GEMINI_AUTH', 'key').strip().lower() == 'vertex' else 'key'


def endpunkt(model: str) -> str:
    if modus() == 'vertex':
        projekt = os.environ['VERTEX_PROJECT']
        region = _region_fuer(model)
        host = 'aiplatform.googleapis.com' if region == 'global' else f'{region}-aiplatform.googleapis.com'
        return f'https://{host}/v1/projects/{projekt}/locations/{region}/publishers/google/models/{model}:generateContent'
    basis = os.environ.get('GEMINI_ENDPOINT', 'https://generativelanguage.googleapis.com/v1beta').rstrip('/')
    return f'{basis}/models/{model}:generateContent'


def _region_fuer(model: str) -> str:
    """Region je Modell: VERTEX_REGION_MAP="gemini-3.5-flash=europe-west3,gemini-2.5-pro=europe-west1",
    sonst VERTEX_REGION. Nötig, weil nicht jedes Modell in jeder EU-Region liegt."""
    karte = os.environ.get('VERTEX_REGION_MAP', '')
    for eintrag in karte.split(','):
        if '=' in eintrag:
            m, r = eintrag.split('=', 1)
            if m.strip() == model:
                return r.strip()
    return os.environ.get('VERTEX_REGION', 'europe-west1')


def kopfzeilen() -> dict:
    if modus() == 'vertex':
        return {'Authorization': 'Bearer ' + _zugangstoken(), 'Content-Type': 'application/json'}
    key = os.environ.get('GEMINI_API_KEY', '').strip()
    if not key:
        raise RuntimeError('GEMINI_API_KEY fehlt in der Umgebung')
    return {'x-goog-api-key': key, 'Content-Type': 'application/json'}


def _dienstkonto() -> dict:
    pfad = os.environ.get('GOOGLE_SERVICE_ACCOUNT_FILE', '').strip()
    if pfad:
        with open(pfad, encoding='utf-8') as f:
            return json.load(f)
    b64 = os.environ.get('GOOGLE_SERVICE_ACCOUNT_JSON_B64', '').strip()
    if b64:
        return json.loads(base64.b64decode(b64).decode('utf-8'))
    raise RuntimeError('Dienstkonto fehlt: GOOGLE_SERVICE_ACCOUNT_FILE oder GOOGLE_SERVICE_ACCOUNT_JSON_B64 setzen')


def _b64url(daten: bytes) -> str:
    return base64.urlsafe_b64encode(daten).rstrip(b'=').decode('ascii')


def _zugangstoken() -> str:
    with _lock:
        if _token['wert'] and time.time() < _token['ablauf'] - 120:
            return _token['wert']
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding
        konto = _dienstkonto()
        jetzt = int(time.time())
        kopf = _b64url(json.dumps({'alg': 'RS256', 'typ': 'JWT'}).encode())
        nutzlast = _b64url(json.dumps({
            'iss': konto['client_email'], 'scope': 'https://www.googleapis.com/auth/cloud-platform',
            'aud': 'https://oauth2.googleapis.com/token', 'iat': jetzt, 'exp': jetzt + 3600,
        }).encode())
        schluessel = serialization.load_pem_private_key(konto['private_key'].encode(), password=None)
        signatur = schluessel.sign(f'{kopf}.{nutzlast}'.encode(), padding.PKCS1v15(), hashes.SHA256())
        jwt = f'{kopf}.{nutzlast}.{_b64url(signatur)}'
        body = urllib.parse.urlencode({'grant_type': 'urn:ietf:params:oauth:grant-type:jwt-bearer', 'assertion': jwt}).encode()
        req = urllib.request.Request('https://oauth2.googleapis.com/token', data=body,
                                     headers={'Content-Type': 'application/x-www-form-urlencoded'})
        with urllib.request.urlopen(req, timeout=30) as r:
            antwort = json.load(r)
        _token['wert'] = antwort['access_token']
        _token['ablauf'] = time.time() + int(antwort.get('expires_in', 3600))
        return _token['wert']
