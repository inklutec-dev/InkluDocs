#!/usr/bin/env python3
"""Unabhaengige Barrierefreiheits-Pruefung der STARTSEITE (03.09.2026, Steves Auftrag):
Tastaturreihenfolge, Fokus-Sichtbarkeit (pixelbasiert), Link-/Knopfnamen, verschachtelte
interaktive Elemente, Landmarken, Ueberschriftenfolge, Kontraste ALLER sichtbaren
Textknoten (WCAG 1.4.3/1.4.11), Zielgroessen (2.5.8), Reflow 320px, lang, Titel.
Aufruf: .venv-pw/bin/python a11y_start_audit.py [BASE]"""
import sys, re, json
from playwright.sync_api import sync_playwright
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8002"
ok = fail = 0
def check(name, cond, detail=""):
    global ok, fail
    if cond: ok += 1; print("  ok   ", name)
    else: fail += 1; print("  FEHL ", name, "->", detail)

JS_TEXTNODES = r"""
() => {
  function lum(c){const [r,g,b]=c.map(v=>{v/=255;return v<=0.03928?v/12.92:Math.pow((v+0.055)/1.055,2.4)});return 0.2126*r+0.7152*g+0.0722*b}
  function parse(s){const m=s.match(/rgba?\(([^)]+)\)/);if(!m)return null;const p=m[1].split(',').map(Number);return {rgb:p.slice(0,3),a:p.length>3?p[3]:1}}
  function bg(el){let e=el;while(e){const c=parse(getComputedStyle(e).backgroundColor);if(c&&c.a>0)return c.rgb;e=e.parentElement}return [255,255,255]}
  function ratio(a,b){const l1=lum(a),l2=lum(b);return (Math.max(l1,l2)+0.05)/(Math.min(l1,l2)+0.05)}
  const out=[];const w=document.createTreeWalker(document.body,NodeFilter.SHOW_TEXT);
  let n;while(n=w.nextNode()){const t=n.textContent.trim();if(!t)continue;const el=n.parentElement;const cs=getComputedStyle(el);
    if(cs.visibility==='hidden'||cs.display==='none')continue;const r=el.getBoundingClientRect();if(r.width===0||r.height===0)continue;
    if(el.closest('details:not([open]) > :not(summary)'))continue;
    const fg=parse(cs.color);if(!fg)continue;const b=bg(el);const size=parseFloat(cs.fontSize);const bold=parseInt(cs.fontWeight)>=700;
    const large=size>=24||(size>=18.66&&bold);const need=large?3:4.5;const rt=ratio(fg.rgb,b);
    out.push({text:t.slice(0,50),tag:el.tagName.toLowerCase()+(el.className?'.'+String(el.className).split(' ')[0]:''),ratio:+rt.toFixed(2),need,pass:rt>=need});}
  return out;}
"""
JS_FOCUS_ORDER = r"""
() => Array.from(document.querySelectorAll('a[href],button,input,select,textarea,summary,[tabindex]:not([tabindex="-1"])'))
  .filter(e=>{const cs=getComputedStyle(e);return cs.display!=='none'&&cs.visibility!=='hidden'})
  .map(e=>({tag:e.tagName.toLowerCase(),name:(e.getAttribute('aria-label')||e.innerText||e.value||'').trim().slice(0,60),href:e.getAttribute('href')||'',
    w:Math.round(e.getBoundingClientRect().width),h:Math.round(e.getBoundingClientRect().height),tabindex:e.getAttribute('tabindex'),
    nested:!!(e.parentElement&&e.parentElement.closest('a,button')) || !!e.querySelector('a,button,input,select,textarea,summary')}))
"""
with sync_playwright() as p:
    b = p.chromium.launch(); page = b.new_page(viewport={"width":1280,"height":900}, locale="de-DE")
    page.goto(BASE + "/", wait_until="networkidle")
    print("=== A. Grundgeruest")
    check("html lang=de", page.get_attribute("html","lang")=="de")
    check("Fenstertitel vorhanden, nennt InkluDocs", "InkluDocs" in page.title(), page.title())
    check("genau eine H1", page.locator("h1").count()==1)
    check("H1 = Versprechen mit Suchbegriffen", " ".join(page.locator("h1").inner_text().split()) == "Alt-Texte per KI für barrierefreie PDF-, Word- und Formulardokumente", page.locator("h1").inner_text())
    check("Markensatz als Absatz unter der H1", " ".join(page.locator("h1 + p.start-claim").inner_text().split()) == "Ein Inhalt. Viele Menschen. Gleiche Chancen.", page.locator("p.start-claim").count())
    for lm in ["header","nav","main","footer"]:
        check(f"Landmarke {lm} genau einmal", page.locator(lm).count()==1, str(page.locator(lm).count()))
    check("nur eine nav, daher ohne aria-label (ARIA nur wo noetig)", page.locator("nav").count()==1 and not page.get_attribute("nav","aria-label"))
    aria = page.evaluate("() => Array.from(document.querySelectorAll('*')).flatMap(e=>Array.from(e.attributes).filter(x=>x.name.startsWith('aria-')||x.name==='role').map(x=>x.name+'='+x.value))")
    check("ARIA auf der Seite = nur aria-current=page (Rest ist natives HTML)", set(aria) <= {"aria-current=page"}, str(sorted(set(aria))))
    # Ueberschriftenfolge
    hs = page.evaluate("() => Array.from(document.querySelectorAll('h1,h2,h3,h4,h5,h6')).map(h=>[+h.tagName[1],h.innerText.trim().slice(0,40)])")
    spruenge=[(hs[i-1],hs[i]) for i in range(1,len(hs)) if hs[i][0]>hs[i-1][0]+1]
    check(f"Ueberschriften ohne Sprung ({len(hs)} Stueck, {sum(1 for h in hs if h[0]==2)} H2, {sum(1 for h in hs if h[0]==3)} H3)", not spruenge, str(spruenge))
    check("leere Ueberschriften: keine", all(h[1] for h in hs))
    print("=== B. Verschachtelung / Semantik")
    nested = page.evaluate("() => document.querySelectorAll('a a, a button, button a, button button, summary a, summary button, h1 a, h2 a, h3 a').length")
    check("keine Links/Knoepfe ineinander, keine Links in Ueberschriften", nested==0, str(nested))
    check("keine div/span mit onclick oder role=button", page.evaluate("() => document.querySelectorAll('[onclick],div[role=button],span[role=button],div[role=link],span[role=link]').length")==0)
    check("keine positiven tabindex", page.evaluate("() => document.querySelectorAll('[tabindex]:not([tabindex=\"-1\"]):not([tabindex=\"0\"])').length")==0)
    check("Karten sind li in ul (kein article = kein Orientierungspunkt), ohne Links", page.evaluate("() => document.querySelectorAll('article, [role=article]').length")==0 and page.locator("ul.start-raster > li.start-karte").count()==9 and page.evaluate("() => document.querySelectorAll('li.start-karte a').length")==0)
    check("details/summary nativ (7 FAQ)", page.locator("details > summary").count()==7)
    check("Listen: ol/ul nur mit li-Kindern", page.evaluate("() => Array.from(document.querySelectorAll('ul,ol')).every(l=>Array.from(l.children).every(c=>c.tagName==='LI'))"))
    check("Bilder: alle mit alt (oder keine Bilder)", page.evaluate("() => Array.from(document.images).every(i=>i.hasAttribute('alt'))"))
    check("kein aria-hidden auf fokussierbaren Elementen", page.evaluate("() => document.querySelectorAll('[aria-hidden=true] a, [aria-hidden=true] button').length")==0)
    print("=== C. Linknamen")
    els = page.evaluate(JS_FOCUS_ORDER)
    links=[e for e in els if e["tag"]=="a"]
    check("alle Links haben einen Namen", all(e["name"] for e in links), str([e for e in links if not e["name"]]))
    vague=[e for e in links if e["name"].lower() in ("hier","mehr","klick","link","weiter","hier klicken")]
    check("keine nichtssagenden Linktexte (hier/mehr/weiter)", not vague, str(vague))
    # gleicher Name -> gleiches Ziel (WCAG 2.4.4 / 2.4.9)
    ziele={}
    for e in links: ziele.setdefault(e["name"],set()).add(e["href"])
    mehrdeutig={k:v for k,v in ziele.items() if len(v)>1}
    check("gleicher Linktext fuehrt immer zum gleichen Ziel", not mehrdeutig, str(mehrdeutig))
    extern=[e for e in links if e["href"].startswith("http") and "inkludocs.de" not in e["href"]]
    check("externe Links (ausserhalb inkludocs.de): keine oder gekennzeichnet", not extern, str(extern))
    print("=== D. Zielgroessen (WCAG 2.5.8, 24x24 min; Knoepfe 44)")
    inline = page.evaluate("() => Array.from(document.querySelectorAll('p a, .dash-legal-links a, li:not(.start-nav li) > a')).map(a=>a.getAttribute('href'))")
    klein=[e for e in els if (e["w"]<24 or e["h"]<24) and e["href"] not in inline]
    check("alle interaktiven Elemente >= 24x24 px (Links im Fliesstext ausgenommen, WCAG 2.5.8 Inline-Ausnahme)", not klein, str(klein))
    print("     Fliesstext-Links (Ausnahme greift):", len([e for e in els if e["href"] in inline]))
    knoepfe=[e for e in els if e["href"] in ("https://demo.inkludocs.de","/register","/login") and e["tag"]=="a"]
    check("Knoepfe (Demo/Register/Anmelden) >= 44 px hoch", all(e["h"]>=44 for e in knoepfe), str([(e["name"],e["h"]) for e in knoepfe]))
    print("=== E. Tastatur: Reihenfolge und sichtbarer Fokus (Pixelvergleich)")
    page.keyboard.press("Tab")
    first = page.evaluate("() => document.activeElement.className + '|' + document.activeElement.textContent.trim()")
    check("erste Tab-Station = Skip-Link", first.startswith("dash-skip"), first)
    # Skip-Link aktivieren: Fokus muss auf main landen
    page.keyboard.press("Enter"); page.wait_for_timeout(100)
    check("Skip-Link Enter -> Fokus auf main", page.evaluate("() => document.activeElement.id")=="main", page.evaluate("() => document.activeElement.tagName+'#'+document.activeElement.id"))
    # Fokusreihenfolge = DOM-Reihenfolge? Alle Stationen durchtabben
    page.goto(BASE + "/", wait_until="networkidle")
    reihenfolge=[]; unsichtbar=[]
    for i in range(80):
        page.keyboard.press("Tab")
        info = page.evaluate("""() => {const e=document.activeElement;if(!e||e===document.body)return null;const r=e.getBoundingClientRect();
           return {tag:e.tagName.toLowerCase(),name:(e.getAttribute('aria-label')||e.innerText||'').trim().slice(0,40),href:e.getAttribute('href')||'',y:Math.round(r.top+window.scrollY),x:Math.round(r.left)}}""")
        if info is None: break
        if info in reihenfolge: break
        # Fokus sichtbar? Screenshot des Elements mit Rand 12px, mit vs ohne Fokus
        el = page.evaluate_handle("() => document.activeElement")
        box = el.as_element().bounding_box()
        clip={"x":max(box["x"]-12,0),"y":max(box["y"]-12,0),"width":box["width"]+24,"height":box["height"]+24}
        mit = page.screenshot(clip=clip)
        page.evaluate("() => document.activeElement.blur()")
        ohne = page.screenshot(clip=clip)
        if mit == ohne: unsichtbar.append(info)
        # Fokus zurueck aufs Element setzen, damit Tab weitergeht
        el.as_element().focus()
        reihenfolge.append(info)
    check(f"Tab-Stationen erreicht: {len(reihenfolge)} (Kopfzeile 6 + Inhalt + FAQ 7 + Fusszeile 7)", len(reihenfolge)>=25, str(len(reihenfolge)))
    ys=[e["y"] for e in reihenfolge]
    # Reihenfolge visuell: y nicht-fallend ausser innerhalb einer Zeile (Toleranz 60px)
    rueck=[(reihenfolge[i-1]["name"],reihenfolge[i]["name"]) for i in range(1,len(ys)) if ys[i] < ys[i-1]-60]
    check("Fokusreihenfolge folgt der Leserichtung (kein Rueckwaertssprung)", not rueck, str(rueck))
    check("Fokus an JEDER Station sichtbar (Pixelvergleich mit/ohne Fokus)", not unsichtbar, str(unsichtbar))
    hero = page.evaluate("() => Array.from(document.querySelectorAll('section.start-hero a.btn-start')).map(a=>a.innerText.trim())")
    check("Hero-Knoepfe in DOM-/Tab-Reihenfolge Demo, Kostenlos starten, Anmelden", hero==["Ohne Anmeldung selbst erleben","Kostenlos starten","Anmelden"], str(hero))
    # FAQ per Tastatur bedienbar
    s = page.locator("details > summary").first; s.focus(); page.keyboard.press("Enter"); page.wait_for_timeout(100)
    check("FAQ: Enter auf summary oeffnet details", page.locator("details").first.get_attribute("open") is not None)
    page.keyboard.press("Space"); page.wait_for_timeout(100)
    check("FAQ: Leertaste schliesst wieder", page.locator("details").first.get_attribute("open") is None)
    print("=== F. Kontraste aller sichtbaren Textknoten (WCAG 1.4.3 AA)")
    page.goto(BASE + "/", wait_until="networkidle")
    page.evaluate("() => document.querySelectorAll('details').forEach(d=>d.open=true)")
    tn = page.evaluate(JS_TEXTNODES)
    schlecht=[t for t in tn if not t["pass"]]
    check(f"{len(tn)} Textknoten geprueft, alle >= 4,5:1 (gross 3:1)", not schlecht, json.dumps(schlecht, ensure_ascii=False)[:800])
    minr=min(tn,key=lambda t:t["ratio"]) if tn else None
    print("     schwaechster Kontrast:", minr)
    # Nicht-Text-Kontrast: Knopfrahmen/Fokusring gegen Hintergrund (1.4.11)
    nt = page.evaluate("""() => {
      function lum(c){const [r,g,b]=c.map(v=>{v/=255;return v<=0.03928?v/12.92:Math.pow((v+0.055)/1.055,2.4)});return 0.2126*r+0.7152*g+0.0722*b}
      function parse(s){const m=s.match(/rgba?\\(([^)]+)\\)/);return m?m[1].split(',').map(Number).slice(0,3):null}
      function ratio(a,b){const l1=lum(a),l2=lum(b);return (Math.max(l1,l2)+0.05)/(Math.min(l1,l2)+0.05)}
      const out=[];document.querySelectorAll('.btn-start').forEach(e=>{const cs=getComputedStyle(e);let p=e;let bgc=null;while(p&&!bgc){const c=parse(getComputedStyle(p).backgroundColor);if(c&&!getComputedStyle(p).backgroundColor.includes(', 0)'))bgc=c;p=p.parentElement}
        const bd=parse(cs.borderTopColor);const bgs=parse(cs.backgroundColor);const own=cs.backgroundColor.includes(', 0)')?null:bgs;
        out.push({name:e.innerText.trim(),rahmen:+ratio(bd,bgc||[255,255,255]).toFixed(2),flaeche:own?+ratio(own,bgc||[255,255,255]).toFixed(2):null})});return out}""")
    schwach=[x for x in nt if max(x["rahmen"], x["flaeche"] or 0) < 3]
    check("Knopf-Umrisse gegen Hintergrund >= 3:1 (1.4.11)", not schwach, str(schwach))
    print("     Knopf-Umrisse:", nt)
    print("=== G. Reflow 320 px / Zoom")
    m = b.new_page(viewport={"width":320,"height":800}, locale="de-DE"); m.goto(BASE + "/", wait_until="networkidle")
    check("320 px: kein horizontales Scrollen", m.evaluate("() => document.documentElement.scrollWidth <= 320"), str(m.evaluate("() => document.documentElement.scrollWidth")))
    check("320 px: Hero-Knoepfe sichtbar", m.locator("section.start-hero a.btn-start").first.is_visible())
    print("=== H. Weitere Sprachen: H1 + Knoepfe")
    for lang, h1start, claim in [("en","AI alt text","One document."),("fr","Textes alternatifs","Un contenu."),("es","Textos alternativos","Un contenido."),("da","AI-alternativtekster","Ét indhold."),("sv","AI-alternativtexter","Ett innehåll.")]:
        pg = b.new_page(extra_http_headers={"Accept-Language": f"{lang};q=1"}); pg.goto(BASE + "/", wait_until="networkidle")
        check(f"{lang}: html lang + H1 + Markensatz uebersetzt", pg.get_attribute("html","lang")==lang and pg.locator("h1").inner_text().startswith(h1start) and claim in pg.locator("p.start-claim").inner_text(), pg.locator("h1").inner_text()[:60]); pg.close()
    b.close()
print(f"\nERGEBNIS: {ok} ok, {fail} Fehler")
sys.exit(1 if fail else 0)
