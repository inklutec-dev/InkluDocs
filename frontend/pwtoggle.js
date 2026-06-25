/*
 * pwtoggle.js — barrierefreies "Passwort anzeigen" fuer alle Passwortfelder
 * -------------------------------------------------------------------------
 * Zweck: Haengt hinter JEDES <input type="password"> eine native Checkbox
 *        "Passwort anzeigen". Angekreuzt = Passwort im Klartext sichtbar (und
 *        mit dem Screenreader Zeichen fuer Zeichen lesbar), abgehakt = wieder
 *        verborgen.
 *
 * Progressive Enhancement: Die Checkbox wird per JavaScript erzeugt. Ohne JS
 *        bleibt das Feld ein ganz normales Passwortfeld — kein toter Schalter.
 *
 * Barrierefreiheit: bewusst eine native <input type="checkbox"> + zugehoeriges
 *        <label> (ueber for/id verknuepft). Eine Checkbox sagt im Screenreader
 *        von sich aus "Kontrollkaestchen, angekreuzt/nicht angekreuzt" an, daher
 *        ist KEIN zusaetzliches ARIA noetig. Tastatur funktioniert nativ
 *        (Tab zum Kaestchen, Leertaste zum Umschalten).
 *
 * Einbindung: <script src="/static/pwtoggle.js" defer></script> auf jeder Seite
 *        mit Passwortfeldern. Das Skript findet die Felder selbst und ist
 *        idempotent (mehrfaches Laden schadet nicht).
 */
(function () {
  "use strict";

  function enhance(input, index) {
    // Doppelte Initialisierung desselben Feldes vermeiden.
    if (input.dataset.pwToggle === "done") return;
    input.dataset.pwToggle = "done";

    // Eindeutige id fuer die Checkbox aus der Feld-id ableiten (Fallback: Index).
    var cbId = (input.id || ("pwfield-" + index)) + "-show";

    var wrap = document.createElement("div");
    wrap.className = "pw-toggle";

    var cb = document.createElement("input");
    cb.type = "checkbox";
    cb.id = cbId;

    var label = document.createElement("label");
    label.setAttribute("for", cbId);
    label.textContent = "Passwort anzeigen";

    // Umschalten: nur der Anzeigetyp aendert sich, der Wert bleibt erhalten.
    cb.addEventListener("change", function () {
      input.type = cb.checked ? "text" : "password";
    });

    wrap.appendChild(cb);
    wrap.appendChild(label);
    input.insertAdjacentElement("afterend", wrap);
  }

  function init() {
    var fields = document.querySelectorAll('input[type="password"]');
    for (var i = 0; i < fields.length; i++) enhance(fields[i], i);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
