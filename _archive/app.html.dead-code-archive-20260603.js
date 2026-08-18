// Archiviertes totes Alt-Subsystem aus app.html, entfernt 03.06.2026.
// Alte In-App-Ansichten (Dashboard/Upload/Projektliste/Einstellungen+API-Keys/Admin)
// + Monkeypatch. Durch das Dashboard (eigene Seiten) ersetzt. Nur zur Referenz.

// ===== showDashboard (Z.71-105, original) =====
    // ─── Dashboard ───────────────────────────────────
    async function showDashboard() {
        if (pollInterval) { clearInterval(pollInterval); pollInterval = null; }
        highlightNav(null);
        const res = await fetch('/api/projects');
        if (res.status === 401) { window.location.href = '/'; return; }
        const data = await res.json();

        main.innerHTML = `
            <div class="upload-area" role="region" aria-label="Dateien hochladen oder Website scannen">
                <h2>Alt-Texte generieren</h2>
                <p>Laden Sie PDFs oder Bilder hoch, um Alt-Texte fuer alle enthaltenen Grafiken zu generieren.</p>
                <p style="font-size:0.9em;color:#555;margin-top:-0.5rem;">Hinweis: Damit die Grafiken in den PDF sauber erkannt werden koennen, sollte es sich um eine getaggte PDF handeln.</p>
                <label class="upload-btn" for="fileInput" role="button" tabindex="0">Dateien auswaehlen</label>
                <input type="file" id="fileInput" accept=".pdf,.jpg,.jpeg,.png,.gif,.svg,.webp,.heic,.heif,.bmp,.tiff,.tif" multiple aria-label="PDF oder Bilddateien auswaehlen">
                <div class="url-scan-area" style="margin-top:1.5rem;padding-top:1.5rem;border-top:1px solid var(--border,#ddd);">
                    <label for="urlInput" style="display:block;font-weight:600;margin-bottom:0.5rem;">Oder Website-URL eingeben</label>
                    <div style="display:flex;gap:0.5rem;flex-wrap:wrap;">
                        <input type="url" id="urlInput" placeholder="https://example.com" style="flex:1;min-width:200px;padding:0.6rem;border:1px solid var(--border,#ccc);border-radius:4px;font-size:1rem;" aria-describedby="urlHint">
                        <button class="btn btn-primary" id="scanBtn" onclick="scanUrl()" style="min-width:120px;">Website scannen</button>
                    </div>
                    <small id="urlHint" style="color:var(--text-muted,#666);display:block;margin-top:0.3rem;">Die Seite wird analysiert und alle Bilder ohne oder mit fehlenden Alt-Texten werden gefunden.</small>
                </div>
                <p id="uploadStatus" aria-live="polite" style="margin-top:1rem;padding:0.75rem;font-weight:600;text-align:center;"></p>
            </div>
            <h2 class="section-title">Meine Projekte</h2>
            <div id="projectList"></div>
        `;

        document.getElementById('fileInput').addEventListener('change', handleUpload);
        document.getElementById('urlInput').addEventListener('keydown', (e) => {
            if (e.key === 'Enter') { e.preventDefault(); scanUrl(); }
        });
        renderProjectList(data.projects);
    }

// ===== handleUpload (Z.107-140, original) =====
    async function handleUpload(e) {
        const files = e.target.files;
        if (!files || files.length === 0) return;
        const status = document.getElementById('uploadStatus');

        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            status.textContent = files.length > 1
                ? 'Datei ' + (i + 1) + ' von ' + files.length + ' wird hochgeladen...'
                : 'Wird hochgeladen...';
            announce(file.name + ' wird hochgeladen');

            const formData = new FormData();
            formData.append('file', file);

            try {
                const res = await fetch('/api/upload', { method: 'POST', body: formData });
                const data = await res.json();
                if (res.ok) {
                    if (i === files.length - 1) {
                        announce(data.total_images + ' Bilder gefunden.');
                        showProject(data.project_id);
                    }
                } else {
                    status.textContent = 'Fehler: ' + (data.detail || 'Upload fehlgeschlagen');
                    announce('Fehler beim Upload von ' + file.name);
                    return;
                }
            } catch (err) {
                status.textContent = 'Verbindungsfehler';
                return;
            }
        }
    }

// ===== scanUrl (Z.142-181, original) =====
    async function scanUrl() {
        const input = document.getElementById('urlInput');
        const btn = document.getElementById('scanBtn');
        const status = document.getElementById('uploadStatus');
        const url = input.value.trim();

        if (!url) { input.focus(); return; }
        if (!url.startsWith('http://') && !url.startsWith('https://')) {
            status.textContent = 'Bitte eine vollstaendige URL eingeben (mit https://)';
            announce('Bitte eine vollstaendige URL eingeben');
            input.focus();
            return;
        }

        btn.disabled = true;
        btn.textContent = 'Wird gescannt...';
        status.textContent = 'Website wird analysiert...';
        announce('Website wird gescannt: ' + url);

        try {
            const res = await fetch('/api/scan-url', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url }),
            });
            const data = await res.json();
            if (res.ok) {
                announce(data.total_images + ' Bilder auf der Seite gefunden.');
                showProject(data.project_id);
            } else {
                status.textContent = 'Fehler: ' + (data.detail || 'Scan fehlgeschlagen');
                announce('Fehler beim Scannen der Website');
            }
        } catch (err) {
            status.textContent = 'Verbindungsfehler';
        } finally {
            btn.disabled = false;
            btn.textContent = 'Website scannen';
        }
    }

// ===== renderProjectList (Z.183-215, original) =====
    function renderProjectList(projects) {
        const list = document.getElementById('projectList');
        if (!projects.length) {
            list.innerHTML = '<div class="empty-state"><p>Noch keine Projekte. Laden Sie Dateien hoch oder scannen Sie eine Website.</p></div>';
            return;
        }
        const statusLabels = { uploaded:'Hochgeladen', extracting:'Extrahiere...', extracted:'Bereit', processing:'Verarbeite...', done:'Fertig', error:'Fehler' };
        const statusClass = s => s === 'done' ? 'badge-done' : s === 'processing' ? 'badge-processing' : s === 'extracted' ? 'badge-ready' : 'badge-error';
        const typeLabels = { pdf:'PDF', images:'Bilder', url:'Website' };

        const extractionLabel = p => (p.project_type === 'pdf')
            ? (p.extraction_method === 'pdfix' ? ' | Extraktion: Strukturell (PDFix)' : ' | Extraktion: Heuristisch (PyMuPDF)')
            : '';

        list.innerHTML = projects.map(p => {
            const progress = p.total_images > 0 ? Math.round((p.processed_images / p.total_images) * 100) : 0;
            const typeLabel = typeLabels[p.project_type] || 'PDF';
            return `
            <div class="card" role="article" aria-label="Projekt ${p.filename}">
                <div class="card-header">
                    <span class="card-name">${p.filename}</span>
                    <span class="badge" style="background:var(--text-muted,#888);color:#fff;padding:0.15rem 0.5rem;border-radius:4px;font-size:0.8rem;">${typeLabel}</span>
                    <span class="badge ${statusClass(p.status)}">${statusLabels[p.status] || p.status}</span>
                </div>
                <div class="card-info">${p.total_images} Bilder | ${p.processed_images} verarbeitet${extractionLabel(p)} | ${new Date(p.created_at).toLocaleDateString('de-DE')}</div>
                ${p.status === 'processing' ? `<div class="progress-bar" role="progressbar" aria-valuenow="${progress}" aria-valuemin="0" aria-valuemax="100"><div class="progress-fill" style="width:${progress}%"></div></div>` : ''}
                <div class="card-actions">
                    <button class="btn btn-primary" onclick="showProject(${p.id})">Oeffnen</button>
                    <button class="btn btn-danger btn-small" onclick="deleteProject(${p.id}, '${p.filename}')">Loeschen</button>
                </div>
            </div>`;
        }).join('');
    }

// ===== deleteProject (Z.217-222, original) =====
    async function deleteProject(id, name) {
        if (!confirm('Projekt "' + name + '" und alle Daten wirklich loeschen?')) return;
        await fetch('/api/projects/' + id, { method: 'DELETE' });
        announce('Projekt geloescht');
        showDashboard();
    }

// ===== showSettings (Z.699-785, original) =====
    // ─── Settings ────────────────────────────────────
    function showSettings() {
        if (pollInterval) { clearInterval(pollInterval); pollInterval = null; }
        highlightNav('settingsBtn');
        main.innerHTML = `
            <button class="back-btn" onclick="showDashboard()">Zurueck</button>
            <h2 class="section-title">Einstellungen</h2>
            <div class="card">
                <h3 style="margin-bottom:1rem;">Passwort aendern</h3>
                <form id="changePwForm" novalidate>
                    <div class="form-group">
                        <label for="old_password">Aktuelles Passwort</label>
                        <input type="password" id="old_password" required autocomplete="current-password">
                    </div>
                    <div class="form-group">
                        <label for="new_password">Neues Passwort (mind. 8 Zeichen)</label>
                        <input type="password" id="new_password" required minlength="8" autocomplete="new-password">
                    </div>
                    <div class="form-group">
                        <label for="new_password2">Neues Passwort wiederholen</label>
                        <input type="password" id="new_password2" required autocomplete="new-password">
                    </div>
                    <button type="submit" class="btn btn-primary">Passwort aendern</button>
                    <span id="pwMsg" role="status" aria-live="polite" style="margin-left:1rem;"></span>
                </form>
            </div>
            <div class="card" style="margin-top:1rem;">
                <h3>Kontoinformationen</h3>
                <p style="margin-top:0.5rem;color:var(--text-muted);">Angemeldet als: ${currentUser.display_name} (${currentUser.email})</p>
            </div>

            <div class="card" style="margin-top:1rem;">
                <h3 id="apiKeysHeading">API-Schluessel</h3>
                <p style="margin-top:0.5rem;margin-bottom:1rem;color:var(--text-muted);">Mit API-Schluesseln koennen externe Anwendungen Alt-Texte generieren.</p>

                <div id="apiKeyCreateSection">
                    <h4>Neuen Schluessel erstellen</h4>
                    <div style="margin-top:0.5rem;display:flex;gap:0.5rem;align-items:center;flex-wrap:wrap;">
                        <label for="newKeyName" class="sr-only">Name fuer neuen API-Schluessel</label>
                        <input type="text" id="newKeyName" placeholder="Name (z.B. Meine App)" style="padding:0.5rem;border:1px solid var(--border);border-radius:4px;flex:1;min-width:150px;" aria-describedby="newKeyHint">
                        <button class="btn btn-primary" onclick="createApiKey()">Erstellen</button>
                    </div>
                    <p id="newKeyHint" class="sr-only">Geben Sie einen Namen ein um den Schluessel spaeter zuordnen zu koennen.</p>
                    <div id="newKeyResult" role="status" aria-live="polite" style="margin-top:0.5rem;"></div>
                </div>

                <div style="margin-top:1.5rem;">
                    <h4 id="myKeysHeading">Meine API-Schluessel <span id="apiKeyCount" style="color:var(--text-muted);font-weight:normal;"></span></h4>
                    <div id="apiKeyList" aria-live="polite" aria-labelledby="myKeysHeading" role="region" style="margin-top:0.5rem;"></div>
                </div>

                <div id="apiKeyEditOverlay" style="display:none;margin-top:1rem;" role="dialog" aria-label="API-Schluessel bearbeiten">
                </div>
            </div>
        `;

        document.getElementById('changePwForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const msg = document.getElementById('pwMsg');
            msg.textContent = '';
            msg.style.color = '';

            const old_password = document.getElementById('old_password').value;
            const new_password = document.getElementById('new_password').value;
            const new_password2 = document.getElementById('new_password2').value;

            if (new_password.length < 8) { msg.textContent = 'Mind. 8 Zeichen'; msg.style.color = 'var(--error)'; return; }
            if (new_password !== new_password2) { msg.textContent = 'Passwoerter stimmen nicht ueberein'; msg.style.color = 'var(--error)'; return; }

            const res = await fetch('/api/change-password', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ old_password, new_password }),
            });
            if (res.ok) {
                msg.textContent = 'Passwort geaendert!';
                msg.style.color = 'var(--success)';
                announce('Passwort wurde geaendert');
            } else {
                const d = await res.json();
                msg.textContent = d.detail || 'Fehler';
                msg.style.color = 'var(--error)';
            }
        });

        loadApiKeys();
    }

// ===== loadApiKeys (Z.787-812, original) =====
    async function loadApiKeys() {
        const list = document.getElementById('apiKeyList');
        const countEl = document.getElementById('apiKeyCount');
        if (!list) return;
        const res = await fetch('/api/api-keys');
        if (!res.ok) return;
        const data = await res.json();
        const keys = data.api_keys || [];

        if (countEl) countEl.textContent = '(' + keys.length + ')';

        if (keys.length === 0) {
            list.innerHTML = '<p style="color:var(--text-muted);">Noch keine API-Schluessel vorhanden.</p>';
            return;
        }
        list.innerHTML = '<ul role="list" style="list-style:none;padding:0;margin:0;">' + keys.map(k => `
            <li id="apikey-row-${k.id}" style="display:flex;justify-content:space-between;align-items:center;padding:0.6rem 0;border-bottom:1px solid var(--border,#eee);">
                <div>
                    <strong id="apikey-name-${k.id}">${k.name}</strong>
                    <span style="color:var(--text-muted);font-size:0.85rem;margin-left:0.5rem;">Erstellt: ${new Date(k.created_at).toLocaleDateString('de-DE')}</span>
                    ${k.last_used ? '<span style="color:var(--text-muted);font-size:0.85rem;margin-left:0.5rem;">Zuletzt verwendet: ' + new Date(k.last_used).toLocaleDateString('de-DE') + '</span>' : ''}
                </div>
                <button class="btn btn-small" onclick="showEditApiKey(${k.id}, '${k.name.replace(/'/g, "\\'")}')">Bearbeiten</button>
            </li>
        `).join('') + '</ul>';
    }

// ===== showEditApiKey (Z.814-832, original) =====
    function showEditApiKey(id, currentName) {
        const overlay = document.getElementById('apiKeyEditOverlay');
        overlay.style.display = 'block';
        overlay.innerHTML = '<div class="card" style="border:2px solid var(--primary);padding:1rem;">' +
            '<h4>Schluessel bearbeiten: ' + currentName + '</h4>' +
            '<div class="form-group" style="margin-top:0.5rem;">' +
            '<label for="editKeyName">Name</label>' +
            '<input type="text" id="editKeyName" value="' + currentName + '" style="padding:0.5rem;border:1px solid var(--border);border-radius:4px;width:100%;max-width:300px;">' +
            '</div>' +
            '<div style="display:flex;gap:0.5rem;margin-top:0.8rem;flex-wrap:wrap;">' +
            '<button class="btn btn-primary" onclick="saveApiKeyName(' + id + ')">Name speichern</button>' +
            '<button class="btn btn-danger" onclick="deleteApiKey(' + id + ', \'' + currentName.replace(/'/g, "\\'") + '\')">Schluessel loeschen</button>' +
            '<button class="btn" onclick="closeEditApiKey()">Abbrechen</button>' +
            '</div>' +
            '<div id="editKeyMsg" role="status" aria-live="polite" style="margin-top:0.5rem;"></div>' +
            '</div>';
        document.getElementById('editKeyName').focus();
        announce('Bearbeitung fuer ' + currentName + ' geoeffnet.');
    }

// ===== closeEditApiKey (Z.834-839, original) =====
    function closeEditApiKey() {
        const overlay = document.getElementById('apiKeyEditOverlay');
        overlay.style.display = 'none';
        overlay.innerHTML = '';
        announce('Bearbeitung geschlossen.');
    }

// ===== saveApiKeyName (Z.841-860, original) =====
    async function saveApiKeyName(id) {
        const nameInput = document.getElementById('editKeyName');
        const msg = document.getElementById('editKeyMsg');
        const newName = nameInput.value.trim();
        if (!newName) { nameInput.focus(); return; }

        const res = await fetch('/api/api-keys/' + id, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: newName }),
        });
        if (res.ok) {
            announce('Name geaendert zu: ' + newName);
            closeEditApiKey();
            loadApiKeys();
        } else {
            const d = await res.json();
            if (msg) { msg.textContent = d.detail || 'Fehler beim Speichern.'; msg.style.color = 'var(--error)'; }
        }
    }

// ===== createApiKey (Z.862-889, original) =====
    async function createApiKey() {
        const nameInput = document.getElementById('newKeyName');
        const result = document.getElementById('newKeyResult');
        const name = nameInput.value.trim();
        if (!name) { nameInput.focus(); return; }

        const res = await fetch('/api/api-keys', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name }),
        });
        if (res.ok) {
            const data = await res.json();
            result.innerHTML = '<div role="alert" style="padding:1rem;background:var(--bg-card,#f9f9f9);border:2px solid var(--primary);border-radius:4px;">' +
                '<h4>Neuer Schluessel erstellt</h4>' +
                '<p style="margin:0.5rem 0;color:var(--text-muted);">Dieser Schluessel wird nur einmal angezeigt. Bitte jetzt kopieren und sicher speichern.</p>' +
                '<code id="apiKeyValue" style="word-break:break-all;font-size:0.9rem;user-select:all;display:block;margin:0.5rem 0;padding:0.5rem;background:#fff;border:1px solid var(--border);border-radius:4px;" aria-label="API-Schluessel">' + data.api_key + '</code>' +
                '<div style="display:flex;gap:0.5rem;margin-top:0.8rem;">' +
                '<button class="btn btn-primary" onclick="copyApiKey()">Schluessel kopieren</button>' +
                '<button class="btn" onclick="closeNewKeyResult()">Zurueck zur Uebersicht</button>' +
                '</div></div>';
            nameInput.value = '';
            announce('Neuer API-Schluessel erstellt. Bitte jetzt kopieren.');
            loadApiKeys();
        } else {
            result.textContent = 'Fehler beim Erstellen.';
        }
    }

// ===== closeNewKeyResult (Z.891-895, original) =====
    function closeNewKeyResult() {
        document.getElementById('newKeyResult').innerHTML = '';
        announce('Schluessel-Anzeige geschlossen.');
        document.getElementById('newKeyName').focus();
    }

// ===== copyApiKey (Z.897-909, original) =====
    function copyApiKey() {
        var key = document.getElementById('apiKeyValue');
        if (!key) return;
        navigator.clipboard.writeText(key.textContent).then(function() {
            announce('API-Schluessel wurde in die Zwischenablage kopiert.');
        }).catch(function() {
            var range = document.createRange();
            range.selectNodeContents(key);
            window.getSelection().removeAllRanges();
            window.getSelection().addRange(range);
            announce('Schluessel markiert. Bitte mit Befehl-C kopieren.');
        });
    }

// ===== deleteApiKey (Z.911-922, original) =====
    async function deleteApiKey(id, keyName) {
        var msg = 'API-Schluessel "' + (keyName || 'Unbenannt') + '" wirklich loeschen? Anwendungen die diesen Schluessel verwenden verlieren sofort den Zugriff.';
        if (!confirm(msg)) return;
        const res = await fetch('/api/api-keys/' + id, { method: 'DELETE' });
        if (res.ok) {
            announce('API-Schluessel geloescht.');
            closeEditApiKey();
            loadApiKeys();
        } else {
            announce('Fehler beim Loeschen.');
        }
    }

// ===== showAdmin (Z.924-1016, original) =====
    // ─── Admin ───────────────────────────────────────
    async function showAdmin() {
        if (pollInterval) { clearInterval(pollInterval); pollInterval = null; }
        highlightNav('adminBtn');
        const res = await fetch('/api/admin/users');
        if (!res.ok) { announce('Kein Zugriff'); showDashboard(); return; }
        const data = await res.json();

        main.innerHTML = `
            <button class="back-btn" onclick="showDashboard()">Zurueck</button>
            <h2 class="section-title">Benutzerverwaltung (${data.users.length} Benutzer)</h2>
            <table class="admin-table" role="table" aria-label="Benutzerliste">
                <thead>
                    <tr><th>Name</th><th>E-Mail</th><th>Projekte</th><th>Status</th><th>Aktionen</th></tr>
                </thead>
                <tbody>
                    ${data.users.map(u => `
                    <tr>
                        <td>${u.display_name} ${u.is_admin ? '<span class="badge badge-admin">Admin</span>' : ''}</td>
                        <td>${u.email}</td>
                        <td>${u.project_count}</td>
                        <td><span class="badge ${u.is_active ? 'badge-done' : 'badge-inactive'}">${u.is_active ? 'Aktiv' : 'Gesperrt'}</span></td>
                        <td style="display:flex;gap:0.3rem;flex-wrap:wrap;">
                            ${u.id !== currentUser.id ? `
                                <button class="btn btn-secondary btn-small" onclick="toggleUser(${u.id})">${u.is_active ? 'Sperren' : 'Entsperren'}</button>
                                <button class="btn btn-secondary btn-small" onclick="resetUserPw(${u.id}, '${u.email}')">Passwort</button>
                                <button class="btn btn-danger btn-small" onclick="deleteUser(${u.id}, '${u.email}')">Loeschen</button>
                            ` : '<span style="color:var(--text-muted);font-size:0.85rem;">Sie selbst</span>'}
                        </td>
                    </tr>
                    `).join('')}
                </tbody>
            </table>
            <div class="card" style="margin-top:1.5rem;">
                <h3 style="margin-bottom:1rem;">Neuen Benutzer anlegen</h3>
                <form id="createUserForm" novalidate>
                    <div style="display:flex;gap:0.75rem;flex-wrap:wrap;align-items:flex-end;">
                        <div class="form-group" style="flex:1;min-width:150px;margin-bottom:0;">
                            <label for="newUserName">Name</label>
                            <input type="text" id="newUserName" required placeholder="Max Mustermann">
                        </div>
                        <div class="form-group" style="flex:1;min-width:200px;margin-bottom:0;">
                            <label for="newUserEmail">E-Mail</label>
                            <input type="email" id="newUserEmail" required placeholder="max@beispiel.de">
                        </div>
                        <div class="form-group" style="flex:1;min-width:150px;margin-bottom:0;">
                            <label for="newUserPw">Passwort (mind. 8)</label>
                            <input type="password" id="newUserPw" required minlength="8" placeholder="Passwort">
                        </div>
                        <div class="form-group" style="flex:1;min-width:150px;margin-bottom:0;">
                            <label for="newUserPw2">Passwort wiederholen</label>
                            <input type="password" id="newUserPw2" required minlength="8" placeholder="Wiederholen">
                        </div>
                        <button type="submit" class="btn btn-primary" style="margin-bottom:0;">Anlegen</button>
                    </div>
                    <div id="createUserMsg" role="status" aria-live="polite" style="margin-top:0.75rem;"></div>
                </form>
            </div>
        `;

        document.getElementById('createUserForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            var msg = document.getElementById('createUserMsg');
            msg.textContent = '';
            msg.style.color = '';
            var name = document.getElementById('newUserName').value.trim();
            var email = document.getElementById('newUserEmail').value.trim();
            var pw = document.getElementById('newUserPw').value;
            var pw2 = document.getElementById('newUserPw2').value;
            if (!name || !email || !pw) { msg.textContent = 'Bitte alle Felder ausfuellen.'; msg.style.color = 'var(--error)'; return; }
            if (pw.length < 8) { msg.textContent = 'Passwort muss mindestens 8 Zeichen haben.'; msg.style.color = 'var(--error)'; return; }
            if (pw !== pw2) { msg.textContent = 'Passwoerter stimmen nicht ueberein.'; msg.style.color = 'var(--error)'; return; }
            var res = await fetch('/api/admin/users/create', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ display_name: name, email: email, password: pw }),
            });
            var data = await res.json();
            if (res.ok) {
                msg.textContent = data.message;
                msg.style.color = 'var(--success)';
                announce('Benutzer erstellt: ' + name);
                document.getElementById('newUserName').value = '';
                document.getElementById('newUserEmail').value = '';
                document.getElementById('newUserPw').value = '';
                document.getElementById('newUserPw2').value = '';
                setTimeout(showAdmin, 1500);
            } else {
                msg.textContent = data.detail || 'Fehler beim Erstellen.';
                msg.style.color = 'var(--error)';
            }
        });
    }

// ===== toggleUser (Z.1018-1021, original) =====
    async function toggleUser(id) {
        await fetch('/api/admin/users/' + id + '/toggle-active', { method: 'POST' });
        showAdmin();
    }

// ===== resetUserPw (Z.1023-1035, original) =====
    async function resetUserPw(id, email) {
        const pw = prompt('Neues Passwort fuer ' + email + ' (mind. 8 Zeichen):');
        if (!pw || pw.length < 8) { alert('Passwort muss mindestens 8 Zeichen haben.'); return; }
        const pw2 = prompt('Passwort wiederholen:');
        if (pw !== pw2) { alert('Passwoerter stimmen nicht ueberein.'); return; }
        const res = await fetch('/api/admin/users/' + id + '/reset-password', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ new_password: pw }),
        });
        if (res.ok) { announce('Passwort zurueckgesetzt'); alert('Passwort wurde zurueckgesetzt.'); }
        else { alert('Fehler beim Zuruecksetzen.'); }
    }

// ===== deleteUser (Z.1037-1042, original) =====
    async function deleteUser(id, email) {
        if (!confirm('User "' + email + '" und ALLE Daten unwiderruflich loeschen? (DSGVO-Loeschung)')) return;
        const res = await fetch('/api/admin/users/' + id, { method: 'DELETE' });
        if (res.ok) { announce('User geloescht'); showAdmin(); }
        else { alert('Fehler beim Loeschen.'); }
    }

// ===== setupDragDrop (Z.1050-1078, original) =====
    // ─── Drag & Drop ────────────────────────────────
    function setupDragDrop() {
        const uploadArea = document.querySelector('.upload-area');
        if (!uploadArea) return;

        ['dragenter', 'dragover'].forEach(evt => {
            uploadArea.addEventListener(evt, (e) => {
                e.preventDefault();
                e.stopPropagation();
                uploadArea.classList.add('drag-over');
            });
        });

        ['dragleave', 'drop'].forEach(evt => {
            uploadArea.addEventListener(evt, (e) => {
                e.preventDefault();
                e.stopPropagation();
                uploadArea.classList.remove('drag-over');
            });
        });

        uploadArea.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (!files || files.length === 0) return;
            const fileInput = document.getElementById('fileInput');
            fileInput.files = files;
            fileInput.dispatchEvent(new Event('change'));
        });
    }

// ===== refreshDailyLimit (Z.1181-1190, original) =====
    async function refreshDailyLimit() {
        try {
            const r = await fetch('/api/me');
            const d = await r.json();
            if (d.ok && d.daily_limit) {
                window._dailyLimit = d.daily_limit;
                updateLimitDisplay();
            }
        } catch(e) {}
    }

// ===== addDonateBanner (Z.1192-1214, original) =====
    // ─── Donate banner on dashboard ─────────────────────
    function addDonateBanner() {
        if (document.getElementById('donateBanner')) return;
        const projectList = document.getElementById('projectList');
        if (!projectList) return;
        const banner = document.createElement('div');
        banner.id = 'donateBanner';
        banner.setAttribute('role', 'complementary');
        banner.setAttribute('aria-label', 'Unterstuetzung');
        banner.style.cssText = 'margin-top:2rem;padding:1.25rem 1.5rem;background:linear-gradient(135deg,#fff7ed,#fef3c7);border:1px solid #fed7aa;border-radius:8px;text-align:center;';
        const dl = window._dailyLimit;
        const limitHint = (dl && dl.remaining <= 0)
            ? '<p style="color:#dc2626;font-weight:600;margin-bottom:0.5rem;">Tageslimit erreicht – morgen geht es weiter.</p>'
            : '';
        banner.innerHTML = limitHint +
            '<h2 style="font-size:1.15rem;color:#1b2a4a;margin:0 0 0.5rem 0;">InkluDocs unterstuetzen</h2>' +
            '<p style="margin:0 0 0.75rem 0;color:#1e293b;font-size:1rem;">InkluDocs ist kostenlos und wird laufend weiterentwickelt. Wenn Sie das Projekt unterstuetzen moechten, freuen wir uns ueber einen freiwilligen Beitrag.</p>' +
            '<a href="https://www.paypal.com/donate?business=steve.weidel%40gmail.com&item_name=InkluDocs+-+Freiwilliger+Beitrag&currency_code=EUR" target="_blank" rel="noopener" ' +
            'style="display:inline-block;background:#0070ba;color:white;padding:0.6rem 1.5rem;border-radius:6px;text-decoration:none;font-weight:600;font-size:1rem;" ' +
            'aria-label="InkluDocs per PayPal unterstuetzen">InkluDocs unterstuetzen</a>' +
            '<p style="margin:0.5rem 0 0 0;color:#64748b;font-size:0.85rem;">Ihr Beitrag hilft, Barrierefreiheit im Web voranzubringen.</p>';
        projectList.parentNode.insertBefore(banner, projectList.nextSibling);
    }

// ===== Monkeypatch showDashboard (Z.1216-1223) =====
    // ─── Patch showDashboard to add drag & drop after render
    const _origShowDashboard = showDashboard;
    showDashboard = async function() {
        await _origShowDashboard();
        addDonateBanner();
        refreshDailyLimit();
        setupDragDrop();
    };
