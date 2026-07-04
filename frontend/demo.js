/* ─────────────────────────────────────────────────────────────────────────
   InkluDocs Demo — Client-Logik. Spricht ausschliesslich die anonymen
   /api/demo/* -Endpunkte an (Sitzung laeuft ueber das HttpOnly-Cookie,
   der Browser sendet es automatisch mit). Alle Statusaenderungen werden
   per Live-Region fuer Screenreader angesagt.

   04.07.2026: fuenfsprachig ueber den t()-Helfer aus demo-shell.js
   (window.I18N liefert base_demo.html); ASCII-Umlaute der sichtbaren Texte
   dabei zu echten Umlauten korrigiert (msgid = Laufzeittext). NEU ausserdem:
   das Sprachmenue #altLangSelect (Ausgabesprache der Alt-Texte) wird beim
   Generieren als language-Feld mitgeschickt.
   ───────────────────────────────────────────────────────────────────────── */
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const els = {};
  ["srStatus", "quota", "uploadSection", "resultSection", "upsellSection", "upsellText",
   "dropzone", "fileInput", "resultImage", "badges", "altField", "langField", "langWrap",
   "deleteBtn", "chatLog", "chatForm", "chatInput",
   "generateArea", "generateBtn", "genStatus", "resultBody", "altLangSelect",
   "deleteConfirmDialog", "deleteCancel", "deleteConfirm"].forEach((k) => (els[k] = $(k)));

  let pollTimer = null;
  let saveTimer = null;
  let busy = false;
  let selectedFile = null; // gewaehltes Bild, das auf den Generieren-Klick wartet

  function announce(msg) {
    if (!els.srStatus) return;
    // Live-Region erst leeren, dann setzen — sonst sagt VoiceOver eine
    // identische Meldung (z. B. zweimal "kopiert") nicht erneut an.
    els.srStatus.textContent = "";
    setTimeout(() => { els.srStatus.textContent = msg; }, 40);
  }

  function setQuota(limits) {
    if (!limits || !els.quota) return;
    els.quota.textContent = t("Noch {bilder_frei} von {bilder_tag} kostenlosen Analysen heute · {chat_frei} von {chat_tag} Chat-Nachrichten.", {
      bilder_frei: limits.images_remaining, bilder_tag: limits.images_per_day,
      chat_frei: limits.chat_remaining, chat_tag: limits.chat_per_day,
    });
  }

  function show(el) { if (el) el.hidden = false; }
  function hide(el) { if (el) el.hidden = true; }

  function showUpsell(text) {
    if (els.upsellText) els.upsellText.textContent = text ||
      t("Deine kostenlosen Analysen für heute sind aufgebraucht.");
    show(els.upsellSection);
    announce(text || t("Kostenloses Kontingent aufgebraucht."));
  }

  async function api(path, opts) {
    const res = await fetch(path, opts || {});
    let data = null;
    try { data = await res.json(); } catch (_) { /* z. B. Datei-Antwort */ }
    return { res, data };
  }

  // ── Upload ────────────────────────────────────────────────────────────────
  function validImage(file) {
    return file && /^image\/(jpeg|png|gif|webp)$/.test(file.type);
  }

  // Ein Bild wurde gewaehlt: Vorschau zeigen und auf den Generieren-Klick warten.
  // Bewusst KEIN automatischer Start mehr — der Nutzer loest die Analyse selbst
  // ueber den Button aus (wie im eingeloggten Werkzeug), damit sichtbar ist,
  // dass und wann etwas passiert.
  function handleFile(file) {
    if (busy) return;
    if (!validImage(file)) {
      announce(t("Bitte ein Bild im Format JPG, PNG, GIF oder WebP wählen."));
      return;
    }
    selectedFile = file;
    hide(els.upsellSection);
    // Sofortige lokale Vorschau (ohne Serverabruf)
    els.resultImage.src = URL.createObjectURL(file);
    els.resultImage.alt = t("Vorschau deines ausgewählten Bildes");
    els.altField.value = "";
    els.langField.value = "";
    els.badges.textContent = "";
    els.genStatus.textContent = "";
    // Ergebnis-Inhalte verbergen, Generieren-Bereich zeigen
    hide(els.resultBody);
    els.generateBtn.disabled = false;
    els.generateBtn.textContent = t("Alternativtext generieren");
    show(els.generateArea);
    show(els.resultSection);
    hide(els.uploadSection); // Upload-Flaeche ausblenden, solange ein Bild gewaehlt ist
    announce(t("Bild ausgewählt. Auf „Alternativtext generieren\" klicken, um die Analyse zu starten."));
    els.generateBtn.focus();
  }

  // Startet die Analyse des gewaehlten Bildes. Sichtbarer Status am Button und
  // im Status-Text plus Live-Meldung — derselbe Ablauf wie der Generieren-Klick
  // im eingeloggten Werkzeug.
  async function startGeneration() {
    if (busy || !selectedFile) return;
    busy = true;
    els.generateBtn.disabled = true;
    els.generateBtn.textContent = t("Generiere …");
    els.genStatus.textContent = t("Bild wird analysiert …");
    announce(t("Bild wird hochgeladen und analysiert. Das dauert ein paar Sekunden."));

    const fd = new FormData();
    fd.append("file", selectedFile);
    // Ausgabesprache der Alt-Texte (Sprachmenue neben dem Generieren-Knopf).
    fd.append("language", els.altLangSelect ? els.altLangSelect.value : "de");
    const { res, data } = await api("/api/demo/generate", { method: "POST", body: fd });
    if (res.status === 429) {
      busy = false;
      hide(els.resultSection);
      showUpsell(data && data.detail);
      return;
    }
    if (!res.ok) {
      busy = false;
      els.generateBtn.disabled = false;
      els.generateBtn.textContent = t("Alternativtext generieren");
      els.genStatus.textContent = t("Es ist ein Fehler aufgetreten. Bitte erneut versuchen.");
      announce((data && data.detail) || t("Upload fehlgeschlagen."));
      return;
    }
    poll();
  }

  function poll() {
    clearTimeout(pollTimer);
    const tick = async () => {
      const { res, data } = await api("/api/demo/result");
      if (!res.ok || !data) { busy = false; return; }
      if (data.limits) setQuota(data.limits);
      const img = data.image;
      if (!img) { busy = false; return; }
      if (img.status === "done") { fillResult(img); busy = false; return; }
      if (img.status === "error") {
        busy = false;
        els.resultImage.alt = t("Analyse fehlgeschlagen");
        // Generieren-Button wieder freigeben, damit der Nutzer es erneut versuchen kann
        els.generateBtn.disabled = false;
        els.generateBtn.textContent = t("Alternativtext generieren");
        els.genStatus.textContent = t("Analyse fehlgeschlagen.");
        announce(t("Die Analyse ist fehlgeschlagen. Bitte versuche es noch einmal."));
        return;
      }
      pollTimer = setTimeout(tick, 2500); // weiter warten (processing)
    };
    pollTimer = setTimeout(tick, 1500);
  }

  function badge(text, kind) {
    return `<span class="badge badge-${kind}">${text}</span>`;
  }

  function fillResult(img) {
    els.altField.value = img.alt_text || "";
    els.langField.value = img.langbeschreibung || "";
    // Langbeschreibung nur zeigen, wenn vorhanden
    els.langWrap.style.display = (img.langbeschreibung && img.langbeschreibung.trim()) ? "" : "none";
    // Kurze, neutrale Bildbeschriftung — der eigentliche Alt-Text steht NUR EINMAL
    // im beschrifteten Feld darunter. Sonst laese ein Screenreader ihn doppelt vor
    // (Steve-Befund 13.06.).
    els.resultImage.alt = t("Vorschau deines hochgeladenen Bildes");
    let b = "";
    // Bildtyp und Konfidenz-Stufe sind Server-Daten und bleiben deutsch
    // (Entscheidung 7b, wie im eingeloggten Werkzeug) — nur die Labels
    // davor werden uebersetzt.
    if (img.image_type) b += badge(t("Typ: {typ}", { typ: img.image_type }), "ready");
    if (img.konfidenz) {
      const k = img.konfidenz === "hoch" ? "done" : img.konfidenz === "niedrig" ? "error" : "processing";
      b += badge(t("Konfidenz: {wert}", { wert: img.konfidenz }), k);
    }
    els.badges.innerHTML = b;
    // Generieren-Bereich ausblenden, Ergebnis-Inhalte einblenden
    els.genStatus.textContent = "";
    hide(els.generateArea);
    show(els.resultBody);
    announce(t("Analyse fertig. Alternativtext und Langbeschreibung stehen bereit."));
    els.altField.focus();
  }

  // ── Kopieren ──────────────────────────────────────────────────────────────
  document.addEventListener("click", async (e) => {
    const btn = e.target.closest("[data-copy]");
    if (!btn) return;
    const field = $(btn.getAttribute("data-copy"));
    if (!field) return;
    // data-copy-label kommt bereits uebersetzt aus dem Template.
    const label = btn.getAttribute("data-copy-label") || t("Text");
    try {
      await navigator.clipboard.writeText(field.value);
      announce(t("{was} kopiert.", { was: label }));
      const old = btn.textContent;
      btn.textContent = t("Kopiert ✓");
      setTimeout(() => (btn.textContent = old), 1500);
    } catch (_) {
      field.select();
      announce(t("Bitte mit Strg+C kopieren."));
    }
  });

  // ── Auto-Speichern manueller Aenderungen (damit der Chatbot sie sieht) ──────
  function scheduleSave() {
    clearTimeout(saveTimer);
    saveTimer = setTimeout(async () => {
      await api("/api/demo/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ alt_text: els.altField.value, langbeschreibung: els.langField.value }),
      });
    }, 800);
  }
  els.altField.addEventListener("input", scheduleSave);
  els.langField.addEventListener("input", scheduleSave);

  // ── Loeschen ────────────────────────────────────────────────────────────────
  // Der eigentliche Loeschvorgang — wird erst nach Bestaetigung im Dialog ausgefuehrt.
  async function performDelete() {
    await api("/api/demo/session", { method: "DELETE" });
    selectedFile = null;
    hide(els.resultSection);
    hide(els.generateArea);
    hide(els.resultBody);
    show(els.uploadSection); // Upload-Flaeche wieder zeigen
    els.fileInput.value = "";
    announce(t("Bild und Ergebnis wurden gelöscht. Du kannst ein neues Bild hochladen."));
    refreshQuota();
    els.fileInput.focus();
  }

  // Loesch-Button oeffnet den barrierefreien Bestaetigungsdialog (natives <dialog>);
  // geloescht wird erst nach „Loeschen". „Abbrechen"/Escape schliessen ihn folgenlos.
  els.deleteBtn.addEventListener("click", () => els.deleteConfirmDialog.showModal());
  els.deleteCancel.addEventListener("click", () => els.deleteConfirmDialog.close());
  els.deleteConfirm.addEventListener("click", () => {
    els.deleteConfirmDialog.close();
    performDelete();
  });

  // ── Chat ────────────────────────────────────────────────────────────────────
  function addMsg(role, text) {
    const p = document.createElement("p");
    p.className = "demo-msg " + (role === "user" ? "demo-msg-user" : "demo-msg-bot");
    p.textContent = text;
    els.chatLog.appendChild(p);
    els.chatLog.scrollTop = els.chatLog.scrollHeight;
  }

  els.chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const msg = els.chatInput.value.trim();
    if (!msg) return;
    addMsg("user", msg);
    els.chatInput.value = "";
    announce(t("Assistent denkt nach …"));
    const { res, data } = await api("/api/demo/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: msg }),
    });
    if (res.status === 429) { showUpsell(data && data.detail); return; }
    if (!res.ok || !data) { addMsg("bot", (data && data.detail) || t("Entschuldigung, das hat nicht geklappt.")); return; }
    addMsg("bot", data.reply || "");
    announce(t("Antwort des Assistenten ist da."));
    if (data.limits) setQuota(data.limits);
    // Der Assistent kann den Alt-Text geaendert haben — frisch laden
    refreshResultTexts();
  });

  async function refreshResultTexts() {
    const { res, data } = await api("/api/demo/result");
    if (res.ok && data && data.image) {
      els.altField.value = data.image.alt_text || els.altField.value;
      if (data.image.langbeschreibung) els.langField.value = data.image.langbeschreibung;
    }
  }

  // ── Init ────────────────────────────────────────────────────────────────────
  async function refreshQuota() {
    const { res, data } = await api("/api/demo/limits");
    if (res.ok && data && data.limits) setQuota(data.limits);
  }

  async function restore() {
    // Bestehende Sitzung beim Neuladen wiederherstellen (falls vorhanden)
    const { res, data } = await api("/api/demo/result");
    if (res.ok && data && data.image) {
      els.resultImage.src = "/api/demo/image?t=" + Date.now();
      show(els.resultSection);
      hide(els.uploadSection);
      if (data.image.status === "done") {
        fillResult(data.image); // zeigt die Ergebnis-Inhalte, blendet den Generieren-Bereich aus
      } else {
        // Analyse laeuft noch (Seite waehrend der Verarbeitung neu geladen):
        // Status anzeigen und weiter pollen.
        show(els.generateArea);
        els.generateBtn.disabled = true;
        els.generateBtn.textContent = t("Generiere …");
        els.genStatus.textContent = t("Bild wird analysiert …");
        busy = true;
        poll();
      }
    }
    if (res.ok && data && data.limits) setQuota(data.limits);
  }

  // Generieren-Button startet die Analyse des gewaehlten Bildes
  els.generateBtn.addEventListener("click", startGeneration);

  // Drag & Drop + Auswahl
  els.fileInput.addEventListener("change", (e) => { if (e.target.files[0]) handleFile(e.target.files[0]); });
  ["dragover", "dragenter"].forEach((ev) =>
    els.dropzone.addEventListener(ev, (e) => { e.preventDefault(); els.dropzone.classList.add("dragover"); }));
  ["dragleave", "drop"].forEach((ev) =>
    els.dropzone.addEventListener(ev, (e) => { e.preventDefault(); els.dropzone.classList.remove("dragover"); }));
  els.dropzone.addEventListener("drop", (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
  });

  refreshQuota();
  restore();
})();
