/* HOCHLADEFELD (05.10.2026, Steve): „Alle neuen Upload-Felder bekommen GENAU die Hochlade-Komponente aus den Projekten —
 * gleiches Aussehen, gleiches Markup und Verhalten.“ Vorbild: app.html uploadBlockHtml() + setupProjectDropzone().
 *
 * Markup wie dort:
 *   section.proj-dropzone (aria-labelledby) > Überschrift.section-title
 *     div.dropzone-inner > input[type=file] (versteckt) + label.upload-btn („PDF-Datei auswählen“) + p.dropzone-or
 *     p.feld-hinweis (Hinweis, aria-describedby)
 *     p[role=status] (Statuszeile, aria-live=polite)
 * Verhalten wie dort: Das Hochladen startet mit der Auswahl (kein extra Knopf), Drag & Drop auf die ganze Fläche,
 * eine Datei. Statuszeile: „<Dateiname> wird hochgeladen“, danach Erfolg oder „Fehler: …“ — EINE Ansage je Schritt.
 * Barrierefreiheit (mitgeprüft): Fokus sichtbar (Ring am Etikett-Knopf, style.css), Dateiname angesagt (Statuszeile),
 * Fehler am Feld (aria-describedby zeigt auf Hinweis UND Statuszeile, aria-invalid bei Fehler).
 *
 * window.hochladefeld({
 *   id, titel, ebene (2–6, Standard 2), karte (true = .card wie in den Projekten), knopf, zusatz (versteckter Teil des
 *   Knopfnamens, z. B. „: Ergebnis für Jahresbericht.pdf“ — damit mehrere Felder unterscheidbar sind), hinweis, accept,
 *   hochladen: async (datei) => ({ ok: true, text: 'Erfolgsmeldung' }) | ({ ok: false, text: 'Fehlertext' }),
 * }) -> HTMLElement (section). Für Express: express.html (Kunde) und verwaltung_express_auftrag.html (Ergebnis,
 * Prüfbericht). Texte über t() aus window.I18N (scripts/check_i18n.py prüft sie mit).
 */
(function () {
  'use strict';

  function el(tag, klasse, text) {
    const e = document.createElement(tag);
    if (klasse) e.className = klasse;
    if (text !== undefined && text !== null) e.textContent = text;
    return e;
  }

  window.hochladefeld = function (o) {
    const zone = el('section', 'proj-dropzone hochladefeld' + (o.karte ? ' card' : ''));
    zone.id = o.id + 'Zone';
    // Bewusst OHNE Namen (kein aria-labelledby): sonst wäre jede Fläche ein eigener Bereich (Landmarke) — bei mehreren
    // Dokumenten in der Verwaltung zu viele (Nachprüfung Barrierefreiheit 05.10.2026, N2). Die Überschrift bleibt.
    const h = el('h' + Math.min(6, Math.max(2, o.ebene || 2)), 'section-title', o.titel);
    h.id = o.id + 'Titel';
    zone.appendChild(h);

    const innen = el('div', 'dropzone-inner');
    const feld = el('input');
    feld.type = 'file';
    feld.id = o.id;
    if (o.accept) feld.accept = o.accept;
    feld.setAttribute('aria-describedby', o.id + 'Hinweis ' + o.id + 'Status');
    const knopf = el('label', 'upload-btn', o.knopf);
    knopf.htmlFor = o.id;
    // Eindeutiger Name je Feld bei gleichem sichtbarem Text (N2): der sichtbare Text bleibt vorn (WCAG 2.5.3).
    if (o.zusatz) knopf.appendChild(el('span', 'visually-hidden', o.zusatz));
    innen.appendChild(feld);
    innen.appendChild(knopf);
    innen.appendChild(el('p', 'dropzone-or', t('oder Datei hierher ziehen')));
    zone.appendChild(innen);

    const hinweis = el('p', 'feld-hinweis', o.hinweis || '');
    hinweis.id = o.id + 'Hinweis';
    zone.appendChild(hinweis);
    const status = el('p');
    status.id = o.id + 'Status';
    status.setAttribute('role', 'status');
    status.setAttribute('aria-live', 'polite');
    status.style.marginTop = '0.5rem';
    status.style.fontWeight = '600';
    zone.appendChild(status);

    let laeuft = false;
    function fehler(text) {
      status.textContent = t('Fehler: {msg}', { msg: text || t('Upload fehlgeschlagen') });
      feld.setAttribute('aria-invalid', 'true');
      zone.classList.add('hochladefeld-fehler');
    }
    function fehlerWeg() {
      feld.removeAttribute('aria-invalid');
      zone.classList.remove('hochladefeld-fehler');
    }

    async function los(datei) {
      if (!datei || laeuft) return;
      laeuft = true;
      fehlerWeg();
      status.textContent = t('{name} wird hochgeladen', { name: datei.name });
      let r = null;
      try { r = await o.hochladen(datei); } catch (e) { r = null; }
      laeuft = false;
      // Wert leeren, damit dieselbe Datei noch einmal gewählt werden kann (sonst kein change-Ereignis).
      try { feld.value = ''; } catch (e) { /* alte Browser */ }
      if (r && r.ok) status.textContent = r.text || '';
      else fehler(r && r.text);
    }

    feld.addEventListener('change', () => los(feld.files && feld.files[0]));

    // Drag & Drop wie setupProjectDropzone: Ergänzung für Sehende, derselbe Weg wie der Knopf.
    const stop = (e) => { e.preventDefault(); e.stopPropagation(); };
    ['dragenter', 'dragover'].forEach((ev) => zone.addEventListener(ev, (e) => { stop(e); zone.classList.add('drag-over'); }));
    zone.addEventListener('dragleave', (e) => {
      stop(e);
      if (!zone.contains(e.relatedTarget)) zone.classList.remove('drag-over');
    });
    zone.addEventListener('drop', (e) => {
      stop(e);
      zone.classList.remove('drag-over');
      const dateien = e.dataTransfer && e.dataTransfer.files;
      if (dateien && dateien.length) los(dateien[0]);
    });

    zone.hochladefeld = { feld: feld, status: status, fehler: fehler, fehlerWeg: fehlerWeg };
    return zone;
  };
})();
