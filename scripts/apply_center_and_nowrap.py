with open('docs/index.html', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Center-align each alloy item in its box:
old_legend = """          <div style="display:flex; gap:22px;">
            <div style="display:flex; flex-direction:column; line-height:1.2;">
              <span style="color:var(--cyan); font-weight:700;">● Cu<sub>50</sub>Zr<sub>50</sub></span>
              <span style="font-size:0.7rem; color:var(--text-muted); margin-top:2px;">32.0% (~40 Å gradient)</span>
            </div>
            <div style="display:flex; flex-direction:column; line-height:1.2;">
              <span style="color:var(--violet); font-weight:700;">● Cu<sub>46</sub>Zr<sub>54</sub></span>
              <span style="font-size:0.7rem; color:var(--text-muted); margin-top:2px;">24.7% (~40 Å gradient)</span>
            </div>
            <div style="display:flex; flex-direction:column; line-height:1.2;">
              <span style="color:var(--amber); font-weight:700;">● Cu<sub>64</sub>Zr<sub>36</sub></span>
              <span style="font-size:0.7rem; color:var(--text-muted); margin-top:2px;">37.1% (12 Å sharp boundary)</span>
            </div>
          </div>"""

new_legend = """          <div style="display:flex; gap:24px;">
            <div style="display:flex; flex-direction:column; align-items:center; text-align:center; line-height:1.25;">
              <span style="color:var(--cyan); font-weight:700;">● Cu<sub>50</sub>Zr<sub>50</sub></span>
              <span style="font-size:0.7rem; color:var(--text-muted); margin-top:2px;">32.0% (~40 Å gradient)</span>
            </div>
            <div style="display:flex; flex-direction:column; align-items:center; text-align:center; line-height:1.25;">
              <span style="color:var(--violet); font-weight:700;">● Cu<sub>46</sub>Zr<sub>54</sub></span>
              <span style="font-size:0.7rem; color:var(--text-muted); margin-top:2px;">24.7% (~40 Å gradient)</span>
            </div>
            <div style="display:flex; flex-direction:column; align-items:center; text-align:center; line-height:1.25;">
              <span style="color:var(--amber); font-weight:700;">● Cu<sub>64</sub>Zr<sub>36</sub></span>
              <span style="font-size:0.7rem; color:var(--text-muted); margin-top:2px;">37.1% (12 Å sharp boundary)</span>
            </div>
          </div>"""

assert old_legend in text, "old_legend not found"
text = text.replace(old_legend, new_legend)

# 2. Fix Holdout readout row line-wrapping when atoms > 10,000:
# Use white-space: nowrap on all 3 cells, add tabular-nums, adjust font-size and padding
old_holdout = """        <div style="display:grid; grid-template-columns: 1fr auto 1fr; align-items:center; gap:8px; padding:6px 12px; background:var(--bg-bento-inner); border:1px solid var(--border-dim); border-radius:8px; margin-bottom:8px; font-family:'JetBrains Mono',monospace; font-size:0.73rem;">
          <div style="color:var(--cyan);">Train Domain: <strong id="trainCntLbl">18,049</strong> atoms</div>
          <div style="color:var(--emerald); text-align:center;">Quarantine Buffer: <strong>3.5 Å (704 atoms)</strong></div>
          <div style="color:var(--amber); text-align:right;">Test Domain: <strong id="testCntLbl">18,075</strong> atoms</div>
        </div>"""

new_holdout = """        <div style="display:grid; grid-template-columns: 1fr auto 1fr; align-items:center; gap:8px; padding:5px 10px; background:var(--bg-bento-inner); border:1px solid var(--border-dim); border-radius:8px; margin-bottom:8px; font-family:'JetBrains Mono',monospace; font-size:0.71rem; font-variant-numeric:tabular-nums;">
          <div style="color:var(--cyan); white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">Train: <strong id="trainCntLbl">18,049</strong> atoms</div>
          <div style="color:var(--emerald); text-align:center; white-space:nowrap; padding:0 4px;">Buffer: <strong>3.5 Å (704)</strong></div>
          <div style="color:var(--amber); text-align:right; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">Test: <strong id="testCntLbl">18,075</strong> atoms</div>
        </div>"""

assert old_holdout in text, "old_holdout not found"
text = text.replace(old_holdout, new_holdout)

# 3. Fix probeSubtitle HTML formatting:
# In JS, change textContent to innerHTML so Cu50Zr50 / Cu46Zr54 have proper subscripts, and R2 has superscript R^2:
old_probe_sub_js = """      let regTitle = 'Diffuse ~40 Å Gradient';
      let regSub = 'Continuous compositional overlap between Cu50Zr50 and Cu46Zr54 (R2 = 0.938).';
      if (cursorXAng >= 116 && cursorXAng <= 128) {
        regTitle = '12 Å Sharp Transition';
        regSub = 'Abrupt structural boundary separating continuous gradient from Cu64Zr36.';
      } else if (cursorXAng > 128) {
        regTitle = 'Cu64Zr36 Domain';
        regSub = 'Structurally distinct polyamorphous phase classified with 0.9703 F1.';
      }

      document.getElementById('probeTitle').textContent = `X = ${cursorXAng.toFixed(1)} Å · ${regTitle}`;
      document.getElementById('probeSubtitle').textContent = regSub;"""

new_probe_sub_js = """      let regTitle = 'Diffuse ~40 Å Gradient';
      let regSub = 'Continuous compositional overlap between Cu<sub>50</sub>Zr<sub>50</sub> and Cu<sub>46</sub>Zr<sub>54</sub> (R<sup>2</sup> = 0.938).';
      if (cursorXAng >= 116 && cursorXAng <= 128) {
        regTitle = '12 Å Sharp Transition';
        regSub = 'Abrupt structural boundary separating continuous gradient from Cu<sub>64</sub>Zr<sub>36</sub>.';
      } else if (cursorXAng > 128) {
        regTitle = 'Cu<sub>64</sub>Zr<sub>36</sub> Domain';
        regSub = 'Structurally distinct polyamorphous phase classified with 0.9703 F1.';
      }

      document.getElementById('probeTitle').innerHTML = `X = ${cursorXAng.toFixed(1)} Å · ${regTitle}`;
      document.getElementById('probeSubtitle').innerHTML = regSub;"""

assert old_probe_sub_js in text, "old_probe_sub_js not found"
text = text.replace(old_probe_sub_js, new_probe_sub_js)

# Write to docs/index.html and root index.html
with open('docs/index.html', 'w', encoding='utf-8') as f:
    f.write(text)

with open('index.html', 'w', encoding='utf-8') as f:
    f.write(text)

print("Applied center-alignment, nowrap holdout labels, and proper subscripts in real-time inspector!")
