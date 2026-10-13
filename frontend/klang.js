/* Klangspur des InkluAgent (Paket 1, Steve 13.10.2026; Schalter funktionen.AGENT_KLANG, inkluagent/klang.py).
 *
 * Zustaende hoerbar statt angesagt: kurze, leise Klaenge, die eine Ansage nie ersetzen (die Information steht immer auch im Text),
 * aber den Stand schneller erfassbar machen und die Flut an Ansagen bei langen Laeufen ueberfluessig (Pruefung 13.10.2026, B9).
 * Raeumlich: was im Arbeitsfenster passiert, klingt von rechts; was im Gespraech ankommt, aus der Mitte. Eine Karte, die ins
 * Gespraech fliegt, rauscht von rechts in die Mitte — wie sie auf dem Bildschirm fliegt.
 *
 * Erzeugt mit Web Audio, ohne Dateien; spaeter ersetzen Steves eigene Klaenge (Pro Tools) die Synthese — dieselben Namen bleiben.
 * Ob es klingt, entscheidet das Konto (<body data-toene="an|aus">, Einstellungen „Töne“). Browser erlauben Ton erst nach einer
 * Bedienung der Seite: der erste Klang nach dem Laden kann ausbleiben, alles danach (Senden, Klick) klingt.
 *
 * Klang.spielen(name): 'karte' (Wusch rechts -> Mitte), 'schritt' (Tick rechts), 'fertig' (Mitte), 'frage' (Zweiklang: eine Karte
 * wartet auf dich), 'fehler' (Mitte, tief).
 */
(function () {
    'use strict';
    let ctx = null;
    let zuletzt = {};           // Name -> Zeit: derselbe Klang hoechstens alle 120 ms (gebuendelte Schritte klingen nicht als Rattern)
    const LAUTSTAERKE = 0.18;   // leise: unter der Stimme des Screenreaders

    function an() {
        return document.body && document.body.getAttribute('data-toene') === 'an';
    }
    function kontext() {
        const K = window.AudioContext || window.webkitAudioContext;
        if (!K) return null;
        if (!ctx) {
            try { ctx = new K(); } catch (e) { return null; }
        }
        if (ctx.state === 'suspended') ctx.resume().catch(() => {});
        return ctx;
    }
    // Ausgang mit Lautstaerke und Raum (-1 links … 0 Mitte … 1 rechts); Safari ohne StereoPanner bekommt Mitte
    function ausgang(c, pan) {
        const g = c.createGain();
        g.gain.value = 0;
        let ziel = g;
        if (c.createStereoPanner) {
            const p = c.createStereoPanner();
            p.pan.value = pan || 0;
            g.connect(p);
            p.connect(c.destination);
            ziel = { gain: g.gain, knoten: g, pan: p.pan };
        } else {
            g.connect(c.destination);
            ziel = { gain: g.gain, knoten: g, pan: null };
        }
        return ziel;
    }
    function ton(c, frequenz, start, dauer, pan, staerke, form) {
        const o = c.createOscillator();
        o.type = form || 'sine';
        o.frequency.value = frequenz;
        const a = ausgang(c, pan);
        o.connect(a.knoten);
        const v = LAUTSTAERKE * (staerke || 1);
        a.gain.setValueAtTime(0, start);
        a.gain.linearRampToValueAtTime(v, start + 0.012);
        a.gain.exponentialRampToValueAtTime(0.0001, start + dauer);
        o.start(start);
        o.stop(start + dauer + 0.02);
    }
    // Wusch: gefiltertes Rauschen, das von rechts in die Mitte wandert und dabei heller wird
    function wusch(c, t0) {
        const dauer = 0.42;
        const puffer = c.createBuffer(1, Math.ceil(c.sampleRate * dauer), c.sampleRate);
        const d = puffer.getChannelData(0);
        for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
        const q = c.createBufferSource();
        q.buffer = puffer;
        const f = c.createBiquadFilter();
        f.type = 'bandpass';
        f.Q.value = 1.4;
        f.frequency.setValueAtTime(500, t0);
        f.frequency.exponentialRampToValueAtTime(2600, t0 + dauer * 0.8);
        const a = ausgang(c, 0.85);
        q.connect(f);
        f.connect(a.knoten);
        a.gain.setValueAtTime(0, t0);
        a.gain.linearRampToValueAtTime(LAUTSTAERKE * 1.6, t0 + dauer * 0.55);
        a.gain.exponentialRampToValueAtTime(0.0001, t0 + dauer);
        if (a.pan) {
            a.pan.setValueAtTime(0.85, t0);
            a.pan.linearRampToValueAtTime(0, t0 + dauer * 0.9);
        }
        q.start(t0);
        q.stop(t0 + dauer + 0.02);
        ton(c, 1046.5, t0 + dauer * 0.8, 0.18, 0, 0.5);   // kleiner Abschluss in der Mitte: „angekommen“
    }
    const KLAENGE = {
        karte: (c, t) => wusch(c, t),
        schritt: (c, t) => ton(c, 1568, t, 0.07, 0.7, 0.6),
        fertig: (c, t) => { ton(c, 784, t, 0.16, 0, 0.8); ton(c, 1175, t + 0.11, 0.26, 0, 0.8); },
        frage: (c, t) => { ton(c, 880, t, 0.14, 0, 0.8); ton(c, 988, t + 0.16, 0.2, 0, 0.8); },
        fehler: (c, t) => { ton(c, 330, t, 0.18, 0, 0.9, 'triangle'); ton(c, 262, t + 0.16, 0.28, 0, 0.9, 'triangle'); },
    };

    function spielen(name) {
        if (!an() || !KLAENGE[name]) return false;
        const jetzt = Date.now();
        if (zuletzt[name] && jetzt - zuletzt[name] < 120) return false;
        zuletzt[name] = jetzt;
        const c = kontext();
        if (!c) return false;
        try { KLAENGE[name](c, c.currentTime + 0.01); } catch (e) { return false; }
        return true;
    }
    // Erste Bedienung der Seite: Ton freischalten (Browser-Regel), damit spaetere Klaenge ohne eigenen Klick klingen
    function freischalten() {
        if (an()) kontext();
        document.removeEventListener('pointerdown', freischalten, true);
        document.removeEventListener('keydown', freischalten, true);
    }
    document.addEventListener('pointerdown', freischalten, true);
    document.addEventListener('keydown', freischalten, true);

    window.Klang = { spielen: spielen, an: an, namen: Object.keys(KLAENGE) };
})();
