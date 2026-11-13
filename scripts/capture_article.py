"""Screenshots and GIF frames of the demo app for the write-up.

  uv run python app/server.py &      # demo app on :8000
  uv run --with playwright --with pillow python scripts/capture_article.py

Writes article/img/*.png and article/img/*.gif. Uses the installed Chrome (channel="chrome").
"""
import os

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "article", "img")
os.makedirs(OUT, exist_ok=True)
URL = "http://127.0.0.1:8000"
OPENING = ["e2e4", "g1f3", "d2d4", "f3d4"]

SETUP = """
async ({moves, persona, style}) => {
  await new Promise(r => { const t = setInterval(() => { if (typeof META !== "undefined" && META && sid && state) { clearInterval(t); r(); } }, 100); });
  if (persona) { document.getElementById("persona").value = persona; document.getElementById("style").value = style || "unknown"; await newGame(); }
  window.ALL = [];
  for (const u of moves) {
    if (!state.legal.includes(u)) break;
    state = await api("/api/move", {sid, uci: u}); ALL.push(...state.new_plies);
  }
  clearInterval(playTimer); onState(); clearInterval(playTimer);
  document.getElementById("meta").textContent = "";
  return state.moves.join(" ");
}
"""

CAPTURE_LAYOUT = """
() => {
  const right = document.querySelector("main > section:nth-child(2)");
  [...right.querySelectorAll(":scope > .card, .row")].forEach(c => {
    const h = c.querySelector("h2"); if (c.classList.contains("row")) c.style.display = "none"; });
  const st = document.createElement("style");
  st.textContent = `#heat th.col{padding:0 !important;font-size:7px;width:10px}
    #heat td{width:10px !important;min-width:10px;max-width:10px;height:15px}
    .vstep{padding:6px 10px;margin:4px 0} main{grid-template-columns:520px 1fr !important}`;
  document.head.appendChild(st);
  window.frame = (pi, si) => {
    const ply = ALL[pi]; state = {...state, fen: ply.fen}; lastMove = ply.uci; renderBoard();
    document.getElementById("plylabel").innerHTML = `<span class="who">${ply.who === "model" ? "model" : "you"}</span><b>${ply.san}</b><span class="muted">= 1 token (#${ply.token})</span>`;
    document.getElementById("steps").innerHTML = ply.steps.map((s, j) => stepHTML(s, j === si ? "active" : j < si ? "done" : "")).join("");
    const st = ply.steps[Math.min(si, ply.steps.length - 1)];
    renderHeat(st.state, si < ply.steps.length ? st.cols : [], null);
    document.getElementById("moves").textContent = "";
    document.getElementById("status").textContent = (ply.who === "model" ? "Tal plays " : "You play ") + ply.san;
  };
  return ALL.length;
}
"""


def shot(page, sel, name, pad=0):
    el = page.locator(sel).first
    el.screenshot(path=os.path.join(OUT, name))


with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True)
    ctx = browser.new_context(viewport={"width": 1500, "height": 900}, device_scale_factor=2, color_scheme="dark")
    page = ctx.new_page()

    # 1. hero: full app after a few moves, animation finished
    page.goto(URL)
    print(page.evaluate(SETUP, {"moves": OPENING + ["f1e2", "e1g1"], "persona": None, "style": None}))
    page.evaluate("""() => { const ply = ALL[ALL.length-1]; renderHeat(state.memory);
      document.getElementById('plylabel').innerHTML = `<span class="who">${ply.who === "model" ? "model" : "you"}</span><b>${ply.san}</b><span class="muted">= 1 token (#${ply.token})</span>`;
      document.getElementById('steps').innerHTML = ply.steps.map(s => stepHTML(s, 'active')).join(''); window.scrollTo(0, 0); }""")
    page.screenshot(path=os.path.join(OUT, "01_app.png"))
    shot(page, "#memboard >> xpath=..", "05_memboard.png")
    page.evaluate("() => { const c = document.querySelector('.legend').closest('.card'); c.style.alignSelf = 'start'; c.parentElement.style.alignItems = 'start'; }")
    shot(page, ".legend >> xpath=..", "04_legend.png")

    # 2. GIF: four moves, every memory step
    page.evaluate(CAPTURE_LAYOUT)
    page.evaluate("() => renderHeat(state.memory)")
    shot(page, "#heat >> xpath=..", "06_heatmap.png")
    frames = []
    region = page.locator("main")
    n = page.evaluate("() => ALL.length")
    for pi in range(4, 8):
        steps = page.evaluate(f"() => ALL[{pi}].steps.length")
        for si in range(steps + 1):
            box = page.evaluate(f"""() => {{ frame({pi}, {si}); window.scrollTo(0, 0);
              const m = document.querySelector('main').getBoundingClientRect();
              const b = document.querySelector('#board').closest('.card').getBoundingClientRect();
              const h = document.querySelector('#heat').closest('.card').getBoundingClientRect();
              return {{x: m.x + 12, y: m.y + 10, width: m.width - 24, height: Math.max(b.bottom, h.bottom) - m.y - 2}}; }}""")
            path = os.path.join(OUT, f"_f{pi}_{si}.png")
            page.screenshot(path=path, clip=box)
            frames.append((path, 1300 if si == steps else 900))
    w0, h0 = Image.open(frames[0][0]).size
    imgs = [Image.open(f).convert("RGB").resize((w0 // 2, h0 // 2), Image.LANCZOS) for f, _ in frames]
    pal = [im.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for im in imgs]
    pal[0].save(os.path.join(OUT, "03_memory.gif"), save_all=True, append_images=pal[1:],
                duration=[d for _, d in frames], loop=0, optimize=True)
    for f, _ in frames:
        os.remove(f)

    # 3. personas: same position, Tal vs Kasparov candidates
    for persona in ("tal", "kasparov"):
        page = ctx.new_page()
        page.goto(URL)
        print(persona, page.evaluate(SETUP, {"moves": ["e2e4", "g1f3", "d2d4", "f3d4", "b1c3"], "persona": persona,
                                               "style": "unknown"}))
        shot(page, "#analysis >> xpath=..", f"07_{persona}.png")
    browser.close()
print("wrote", sorted(os.listdir(OUT)))
