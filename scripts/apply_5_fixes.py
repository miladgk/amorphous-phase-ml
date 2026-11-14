import re

with open('docs/index.html', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Global CSS for sub, sup, and font-variant-numeric: tabular-nums
css_insert = """
    sub, sup {
      font-size: 0.72em;
      line-height: 0;
      position: relative;
      vertical-align: baseline;
      font-weight: inherit;
      font-family: inherit;
    }
    sub { bottom: -0.22em; }
    sup { top: -0.38em; }

    .bento-table td,
    .kpi-pill-val,
    .probe-row,
    .cm-num,
    #scannerHUD {
      font-variant-numeric: tabular-nums;
    }
"""

anchor_css = '/* Zero vertical page scroll on desktop */\n    }'
assert anchor_css in text, "Could not find CSS anchor"
text = text.replace(anchor_css, anchor_css + css_insert)

# 2. Fix cm-bento column width and cm-row-lbl nowrap
assert 'grid-template-columns: 95px repeat(3, 1fr);' in text
text = text.replace('grid-template-columns: 95px repeat(3, 1fr);', 'grid-template-columns: 120px repeat(3, 1fr);')

assert '.cm-row-lbl {\n      display: flex;' in text
text = text.replace('.cm-row-lbl {\n      display: flex;', '.cm-row-lbl {\n      display: flex;\n      white-space: nowrap;')

# 3. Fix #cmHoverBox fixed height
old_cm_hover = 'id="cmHoverBox"\n          style="background:var(--bg-bento-inner); border:1px solid var(--border-dim); border-radius:10px; padding:11px 14px; font-family:\'JetBrains Mono\',monospace; font-size:0.76rem; color:var(--cyan);"'
new_cm_hover = 'id="cmHoverBox"\n          style="background:var(--bg-bento-inner); border:1px solid var(--border-dim); border-radius:10px; padding:0 14px; font-family:\'JetBrains Mono\',monospace; font-size:0.76rem; color:var(--cyan); height: 56px; display: flex; align-items: center; overflow: hidden; flex-shrink: 0;"'
assert old_cm_hover in text, "Could not find old_cm_hover"
text = text.replace(old_cm_hover, new_cm_hover)

# 4. Fix #featDetailBox fixed height
old_feat_detail = 'id="featDetailBox"\n          style="background:var(--bg-bento-inner); border:1px solid var(--border-dim); border-radius:10px; padding:10px 12px; font-size:0.77rem; color:#cbd5e1;"'
new_feat_detail = 'id="featDetailBox"\n          style="background:var(--bg-bento-inner); border:1px solid var(--border-dim); border-radius:10px; padding:0 12px; font-size:0.77rem; color:#cbd5e1; height: 62px; display: flex; align-items: center; overflow: hidden; flex-shrink: 0;"'
assert old_feat_detail in text, "Could not find old_feat_detail"
text = text.replace(old_feat_detail, new_feat_detail)

# 5. Fix HTML text with proper <sub> and <sup> in HTML portion (before <script>)
script_tag_idx = text.find('<script>')
assert script_tag_idx != -1
html_part = text[:script_tag_idx]
js_part = text[script_tag_idx:]

html_subs = [
    ('Cu₅₀Zr₅₀', 'Cu<sub>50</sub>Zr<sub>50</sub>'),
    ('Cu₄₆Zr₅₄', 'Cu<sub>46</sub>Zr<sub>54</sub>'),
    ('Cu₆₄Zr₃₆', 'Cu<sub>64</sub>Zr<sub>36</sub>'),
    ('R²', 'R<sup>2</sup>'),
    ('R₁', 'R<sub>1</sub>'),
    ('C₁', 'C<sub>1</sub>'),
    ('C₂', 'C<sub>2</sub>')
]
for old, new in html_subs:
    html_part = html_part.replace(old, new)

# In JS portion, replace Unicode subscripts/superscripts with clean ASCII
js_subs = [
    ('Cu₅₀Zr₅₀', 'Cu50Zr50'),
    ('Cu₄₆Zr₅₄', 'Cu46Zr54'),
    ('Cu₆₄Zr₃₆', 'Cu64Zr36'),
    ('R²', 'R2'),
    ('R₁', 'R1'),
    ('C₁', 'C1'),
    ('C₂', 'C2')
]
for old, new in js_subs:
    js_part = js_part.replace(old, new)

# 6. Add 3-column HTML readout row right above #holdoutSplitCanvas
old_holdout_slider = """        <input type="range" id="splitPlaneSlider" min="45" max="140" step="0.5" value="91.5"
          oninput="updateSplitSimulator(this.value)" style="margin-bottom:8px;">

        <div class="canvas-box" style="max-height:95px;">
          <canvas id="holdoutSplitCanvas"></canvas>
        </div>"""

new_holdout_slider = """        <input type="range" id="splitPlaneSlider" min="45" max="140" step="0.5" value="91.5"
          oninput="updateSplitSimulator(this.value)" style="margin-bottom:6px;">

        <div style="display:grid; grid-template-columns: 1fr auto 1fr; align-items:center; gap:8px; padding:6px 12px; background:var(--bg-bento-inner); border:1px solid var(--border-dim); border-radius:8px; margin-bottom:8px; font-family:'JetBrains Mono',monospace; font-size:0.73rem;">
          <div style="color:var(--cyan);">Train Domain: <strong id="trainCntLbl">18,049</strong> atoms</div>
          <div style="color:var(--emerald); text-align:center;">Quarantine Buffer: <strong>3.5 Å (704 atoms)</strong></div>
          <div style="color:var(--amber); text-align:right;">Test Domain: <strong id="testCntLbl">18,075</strong> atoms</div>
        </div>

        <div class="canvas-box" style="max-height:85px;">
          <canvas id="holdoutSplitCanvas"></canvas>
        </div>"""

assert old_holdout_slider in html_part, "Could not find old_holdout_slider"
html_part = html_part.replace(old_holdout_slider, new_holdout_slider)

# 7. Update updateSplitSimulator and drawHoldoutSimulator in JS
old_split_js = """    function updateSplitSimulator(val) {
      splitPlaneX = parseFloat(val);
      document.getElementById('splitPosLabel').textContent = `Split Plane: X = ${splitPlaneX.toFixed(1)} Å`;
      drawHoldoutSimulator();
    }

    function drawHoldoutSimulator() {
      const dpr = window.devicePixelRatio || 1;
      const rect = hsCanvas.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) return;
      hsCanvas.width = rect.width * dpr;
      hsCanvas.height = rect.height * dpr;
      hsCtx.scale(dpr, dpr);

      const W = rect.width, H = rect.height;
      hsCtx.clearRect(0, 0, W, H);

      const splitPx = (splitPlaneX / 183) * W;
      const bufPx = Math.max(18, (3.5 / 183) * W);

      // Left: Training Domain
      hsCtx.fillStyle = 'rgba(56, 189, 248, 0.16)';
      hsCtx.fillRect(0, 0, splitPx - bufPx / 2, H);

      // Center: 3.5 A Quarantine Buffer Zone
      hsCtx.fillStyle = 'rgba(52, 211, 153, 0.28)';
      hsCtx.fillRect(splitPx - bufPx / 2, 0, bufPx, H);
      hsCtx.strokeStyle = '#34d399';
      hsCtx.lineWidth = 1.5;
      hsCtx.strokeRect(splitPx - bufPx / 2, 0, bufPx, H);

      // Right: Spatial Holdout Test Domain
      hsCtx.fillStyle = 'rgba(251, 146, 60, 0.16)';
      hsCtx.fillRect(splitPx + bufPx / 2, 0, W - (splitPx + bufPx / 2), H);

      const trainAtoms = Math.round(((splitPlaneX - 1.75) / 183) * 36828);
      const bufAtoms = Math.round((3.5 / 183) * 36828);
      const testAtoms = 36828 - trainAtoms - bufAtoms;

      hsCtx.font = '600 11px "JetBrains Mono", monospace';
      hsCtx.fillStyle = '#38bdf8';
      hsCtx.fillText(`TRAIN DOMAIN (${trainAtoms.toLocaleString()} atoms)`, 12, H / 2 + 4);

      hsCtx.fillStyle = '#34d399';
      hsCtx.fillText('3.5Å BUFFER', Math.max(8, splitPx - 36), 18);

      hsCtx.fillStyle = '#fb923c';
      hsCtx.fillText(`TEST DOMAIN (${testAtoms.toLocaleString()} atoms)`, W - 210, H / 2 + 4);
    }"""

new_split_js = """    function updateSplitSimulator(val) {
      splitPlaneX = parseFloat(val);
      document.getElementById('splitPosLabel').textContent = `Split Plane: X = ${splitPlaneX.toFixed(1)} Å`;
      drawHoldoutSimulator();
    }

    function drawHoldoutSimulator() {
      const dpr = window.devicePixelRatio || 1;
      const rect = hsCanvas.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) return;
      hsCanvas.width = rect.width * dpr;
      hsCanvas.height = rect.height * dpr;
      hsCtx.scale(dpr, dpr);

      const W = rect.width, H = rect.height;
      hsCtx.clearRect(0, 0, W, H);

      const splitPx = (splitPlaneX / 183) * W;
      const bufPx = Math.max(22, (3.5 / 183) * W);

      // Left: Training Domain
      hsCtx.fillStyle = 'rgba(56, 189, 248, 0.16)';
      hsCtx.fillRect(0, 0, splitPx - bufPx / 2, H);

      // Center: 3.5 A Quarantine Buffer Zone
      hsCtx.fillStyle = 'rgba(52, 211, 153, 0.28)';
      hsCtx.fillRect(splitPx - bufPx / 2, 0, bufPx, H);
      hsCtx.strokeStyle = '#34d399';
      hsCtx.lineWidth = 1.5;
      hsCtx.strokeRect(splitPx - bufPx / 2, 0, bufPx, H);

      // Right: Spatial Holdout Test Domain
      hsCtx.fillStyle = 'rgba(251, 146, 60, 0.16)';
      hsCtx.fillRect(splitPx + bufPx / 2, 0, W - (splitPx + bufPx / 2), H);

      const trainAtoms = Math.round(((splitPlaneX - 1.75) / 183) * 36828);
      const bufAtoms = Math.round((3.5 / 183) * 36828);
      const testAtoms = 36828 - trainAtoms - bufAtoms;

      const trainLbl = document.getElementById('trainCntLbl');
      if (trainLbl) trainLbl.textContent = trainAtoms.toLocaleString();
      const testLbl = document.getElementById('testCntLbl');
      if (testLbl) testLbl.textContent = testAtoms.toLocaleString();

      hsCtx.font = '600 11px "JetBrains Mono", monospace';
      hsCtx.textAlign = 'center';

      // Dynamically center TRAIN label in left domain
      const trainCenter = Math.max(30, (splitPx - bufPx / 2) / 2);
      hsCtx.fillStyle = '#38bdf8';
      hsCtx.fillText('TRAIN', trainCenter, H / 2 + 4);

      // Center 3.5 A text inside a dark pill badge at splitPx
      const pillW = 44, pillH = 18;
      hsCtx.fillStyle = 'rgba(5, 8, 17, 0.9)';
      hsCtx.beginPath();
      if (hsCtx.roundRect) {
        hsCtx.roundRect(splitPx - pillW / 2, H / 2 - pillH / 2, pillW, pillH, 4);
      } else {
        hsCtx.rect(splitPx - pillW / 2, H / 2 - pillH / 2, pillW, pillH);
      }
      hsCtx.fill();
      hsCtx.strokeStyle = '#34d399';
      hsCtx.lineWidth = 1;
      hsCtx.stroke();
      hsCtx.fillStyle = '#34d399';
      hsCtx.fillText('3.5 Å', splitPx, H / 2 + 4);

      // Dynamically center TEST label in right domain
      const rightStart = splitPx + bufPx / 2;
      const testCenter = Math.min(W - 30, rightStart + (W - rightStart) / 2);
      hsCtx.fillStyle = '#fb923c';
      hsCtx.fillText('TEST', testCenter, H / 2 + 4);

      hsCtx.textAlign = 'left';
    }"""

assert old_split_js in js_part, "Could not find old_split_js"
js_part = js_part.replace(old_split_js, new_split_js)

# 8. Update drawRdfChart in JS
old_rdf_js = """      if (rdfState.shells) {
        [
          { r: 2.588, lbl: 'R1 (2.59Å)', c: '#34d399' },
          { r: 4.482, lbl: '√3·R1', c: '#fb923c' },
          { r: 5.176, lbl: '√4·R1', c: '#fb923c' },
          { r: 6.847, lbl: '√7·R1', c: '#fb923c' },
          { r: 8.965, lbl: '√12·R1', c: '#fb923c' }
        ].forEach(s => {
          const sx = padL + (s.r / 12) * pw;
          rCtx.beginPath();
          rCtx.setLineDash([4, 4]);
          rCtx.moveTo(sx, padT);
          rCtx.lineTo(sx, padT + ph);
          rCtx.strokeStyle = s.c;
          rCtx.lineWidth = 1.2;
          rCtx.stroke();
          rCtx.setLineDash([]);
          rCtx.fillStyle = s.c;
          rCtx.fillText(s.lbl, sx + 4, padT + 12);
        });
      }"""

new_rdf_js = """      if (rdfState.shells) {
        [
          { r: 2.588, lbl: 'R1 (2.59Å)', c: '#34d399', yOffset: 14 },
          { r: 4.482, lbl: '√3·R1', c: '#fb923c', yOffset: 14 },
          { r: 5.176, lbl: '√4·R1', c: '#fb923c', yOffset: 28 },
          { r: 6.847, lbl: '√7·R1', c: '#fb923c', yOffset: 14 },
          { r: 8.965, lbl: '√12·R1', c: '#fb923c', yOffset: 14 }
        ].forEach(s => {
          const sx = padL + (s.r / 12) * pw;
          rCtx.beginPath();
          rCtx.setLineDash([4, 4]);
          rCtx.moveTo(sx, padT);
          rCtx.lineTo(sx, padT + ph);
          rCtx.strokeStyle = s.c;
          rCtx.lineWidth = 1.2;
          rCtx.stroke();
          rCtx.setLineDash([]);

          // Dark pill background behind shell label text
          rCtx.font = '11px "JetBrains Mono", monospace';
          const txtW = rCtx.measureText(s.lbl).width;
          const labelY = padT + s.yOffset;
          rCtx.fillStyle = 'rgba(5, 8, 17, 0.88)';
          rCtx.fillRect(sx + 3, labelY - 11, txtW + 6, 15);

          rCtx.fillStyle = s.c;
          rCtx.fillText(s.lbl, sx + 6, labelY);
        });
      }"""

assert old_rdf_js in js_part, "Could not find old_rdf_js"
js_part = js_part.replace(old_rdf_js, new_rdf_js)

# Update hover badge in drawRdfChart to an opaque pill badge drawn ON TOP
old_hover_rdf = """      if (rdfHoverR !== null) {
        const hx = padL + (rdfHoverR / 12) * pw;
        const hg = sgModel(rdfHoverR);
        const hy = padT + ph - (hg / 4.6) * ph;

        rCtx.beginPath();
        rCtx.moveTo(hx, padT);
        rCtx.lineTo(hx, padT + ph);
        rCtx.strokeStyle = 'rgba(255,255,255,0.5)';
        rCtx.stroke();

        rCtx.beginPath();
        rCtx.arc(hx, hy, 4.5, 0, Math.PI * 2);
        rCtx.fillStyle = '#38bdf8';
        rCtx.fill();

        rCtx.fillStyle = '#f8fafc';
        rCtx.fillText(`r = ${rdfHoverR.toFixed(2)} Å │ g(r) = ${hg.toFixed(2)}`, W - 210, padT + 14);
      }"""

new_hover_rdf = """      if (rdfHoverR !== null) {
        const hx = padL + (rdfHoverR / 12) * pw;
        const hg = sgModel(rdfHoverR);
        const hy = padT + ph - (hg / 4.6) * ph;

        rCtx.beginPath();
        rCtx.moveTo(hx, padT);
        rCtx.lineTo(hx, padT + ph);
        rCtx.strokeStyle = 'rgba(255,255,255,0.5)';
        rCtx.stroke();

        rCtx.beginPath();
        rCtx.arc(hx, hy, 4.5, 0, Math.PI * 2);
        rCtx.fillStyle = '#38bdf8';
        rCtx.fill();

        // Opaque pill badge drawn on top of all curves and vertical lines
        const badgeX = W - 228, badgeY = padT + 4, badgeW = 212, badgeH = 24;
        rCtx.fillStyle = 'rgba(13, 19, 34, 0.95)';
        rCtx.beginPath();
        if (rCtx.roundRect) {
          rCtx.roundRect(badgeX, badgeY, badgeW, badgeH, 6);
        } else {
          rCtx.rect(badgeX, badgeY, badgeW, badgeH);
        }
        rCtx.fill();
        rCtx.strokeStyle = 'rgba(56, 189, 248, 0.45)';
        rCtx.lineWidth = 1;
        rCtx.stroke();

        rCtx.font = '11px "JetBrains Mono", monospace';
        rCtx.fillStyle = '#f8fafc';
        rCtx.fillText(`r = ${rdfHoverR.toFixed(2)} Å │ g(r) = ${hg.toFixed(2)}`, badgeX + 10, badgeY + 16);
      }"""

assert old_hover_rdf in js_part, "Could not find old_hover_rdf"
js_part = js_part.replace(old_hover_rdf, new_hover_rdf)

final_text = html_part + js_part
with open('docs/index.html', 'w', encoding='utf-8') as f:
    f.write(final_text)

with open('index.html', 'w', encoding='utf-8') as f:
    f.write(final_text)

print("Successfully applied all 5 fixes to docs/index.html and index.html!")
