// HTTP/2-Server (TLS) fuer den Netzwechsel-Nachweis: /langsam antwortet nach 25 s, alles andere sofort.
// Aufruf: node h2server.js <key.pem> <cert.pem> <adresse>
const http2 = require('http2'); const fs = require('fs');
const s = http2.createSecureServer({ key: fs.readFileSync(process.argv[2]), cert: fs.readFileSync(process.argv[3]), allowHTTP1: true });
s.on('request', (req, res) => {
  if (req.url.startsWith('/langsam')) { setTimeout(() => { res.writeHead(200, { 'content-type': 'application/json' }); res.end('{"ok":true}'); }, 25000); return; }
  res.writeHead(200, { 'content-type': 'text/html' }); res.end('<!doctype html><title>Nachweis</title><p>x</p>');
});
s.listen(8443, process.argv[4] || '127.0.0.1', () => console.log('h2-Server bereit'));
