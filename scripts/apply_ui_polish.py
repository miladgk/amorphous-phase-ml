with open('docs/index.html', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Update alloy legend in View 01 (names on top, values underneath)
old_legend = """          <div style="display:flex; gap:16px;">
            <span><strong style="color:var(--cyan);">● Cu<sub>50</sub>Zr<sub>50</sub>:</strong> 32.0% (~40 Å gradient)</span>
            <span><strong style="color:var(--violet);">● Cu<sub>46</sub>Zr<sub>54</sub>:</strong> 24.7% (~40 Å gradient)</span>
            <span><strong style="color:var(--amber);">● Cu<sub>64</sub>Zr<sub>36</sub>:</strong> 37.1% (12 Å sharp boundary)</span>
          </div>"""

new_legend = """          <div style="display:flex; gap:22px;">
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

assert old_legend in text, "old_legend not found"
text = text.replace(old_legend, new_legend)

# 2. Fix Two Distinct Polyamorphous Regimes box overflow in View 01
old_v0_right = """      <!-- Right Top 4-Column Bento Tile: Live X-Slice Crosshair Probe -->
      <div class="bento-tile" style="grid-column: span 4; grid-row: span 7;">
        <div>
          <div class="tile-kicker">Real-Time Local Slice Inspector</div>
          <div class="tile-title" id="probeTitle">X = 122.0 Å · Sharp Transition</div>
          <p class="tile-desc" id="probeSubtitle" style="margin-bottom: 12px;">
            Narrow 12 Å structural interface separating diffuse Cu<sub>50</sub>Zr<sub>50</sub>/Cu<sub>46</sub>Zr<sub>54</sub> from Cu<sub>64</sub>Zr<sub>36</sub>.
          </p>

          <div class="probe-metric">
            <div class="probe-row">
              <span style="color:var(--cyan); font-weight:600;">Phase 0 (Cu<sub>50</sub>Zr<sub>50</sub>) Density</span>
              <span id="probeP0" style="font-family:'JetBrains Mono',monospace;">195 atoms/bin</span>
            </div>
            <div class="probe-bar">
              <div class="probe-fill" id="barProbe0" style="width:32%; background:var(--cyan);"></div>
            </div>
          </div>

          <div class="probe-metric">
            <div class="probe-row">
              <span style="color:var(--violet); font-weight:600;">Phase 1 (Cu<sub>46</sub>Zr<sub>54</sub>) Density</span>
              <span id="probeP1" style="font-family:'JetBrains Mono',monospace;">110 atoms/bin</span>
            </div>
            <div class="probe-bar">
              <div class="probe-fill" id="barProbe1" style="width:18%; background:var(--violet);"></div>
            </div>
          </div>

          <div class="probe-metric">
            <div class="probe-row">
              <span style="color:var(--amber); font-weight:600;">Phase 2 (Cu<sub>64</sub>Zr<sub>36</sub>) Density</span>
              <span id="probeP2" style="font-family:'JetBrains Mono',monospace;">340 atoms/bin</span>
            </div>
            <div class="probe-bar">
              <div class="probe-fill" id="barProbe2" style="width:56%; background:var(--amber);"></div>
            </div>
          </div>
        </div>

        <div
          style="display:grid; grid-template-columns:1fr 1fr; gap:8px; padding-top:8px; border-top:1px solid var(--border-dim);">
          <div style="background:rgba(255,255,255,0.02); padding:8px 10px; border-radius:8px;">
            <div style="font-size:0.68rem; color:var(--text-dim); font-family:'JetBrains Mono',monospace;">LOCAL VON
              MISES</div>
            <div id="probeVM"
              style="font-size:1.05rem; font-weight:700; font-family:'Space Grotesk',sans-serif; color:var(--emerald);">
              284 MPa</div>
          </div>
          <div style="background:rgba(255,255,255,0.02); padding:8px 10px; border-radius:8px;">
            <div style="font-size:0.68rem; color:var(--text-dim); font-family:'JetBrains Mono',monospace;">HYDROSTATIC
              PRESS.</div>
            <div id="probeHP"
              style="font-size:1.05rem; font-weight:700; font-family:'Space Grotesk',sans-serif; color:var(--cyan);">+18
              MPa</div>
          </div>
        </div>
      </div>

      <!-- Right Bottom 4-Column Bento Tile: Physical Regime Contrast -->
      <div class="bento-tile" style="grid-column: span 4; grid-row: span 5;">
        <div>
          <div class="tile-kicker">Two Distinct Polyamorphous Regimes</div>
          <div class="tile-title">Diffuse Gradient vs. Abrupt Boundary</div>
        </div>

        <div style="display:flex; flex-direction:column; gap:8px; margin:auto 0;">
          <div
            style="background:rgba(56,189,248,0.07); border-left:3px solid var(--cyan); padding:9px 12px; border-radius:0 10px 10px 0; font-size:0.79rem;">
            <strong style="color:var(--text-primary);">Left/Center (0 – 122 Å):</strong> Smooth ~40 Å compositional
            gradient between Cu<sub>50</sub>Zr<sub>50</sub> and Cu<sub>46</sub>Zr<sub>54</sub> (4 at.% difference, continuous regression <strong>R<sup>2</sup> =
              0.938</strong>).
          </div>
          <div
            style="background:rgba(251,146,60,0.08); border-left:3px solid var(--amber); padding:9px 12px; border-radius:0 10px 10px 0; font-size:0.79rem;">
            <strong style="color:var(--text-primary);">Right Domain (122 – 183 Å):</strong> Sharp 12 Å structural phase
            transition into Cu<sub>64</sub>Zr<sub>36</sub> with near-perfect separability (<strong>F1 = 0.9703</strong>).
          </div>
        </div>

        <div
          style="display:flex; justify-content:space-between; align-items:center; font-size:0.75rem; color:var(--text-muted);">
          <span>Confidence Filter: <strong id="miniConfVal" style="color:var(--cyan);">≥ 0.50</strong></span>
          <button class="seg-pill active" onclick="switchView(1)">Open Confidence Lab →</button>
        </div>
      </div>"""

new_v0_right = """      <!-- Right Top 4-Column Bento Tile: Live X-Slice Crosshair Probe -->
      <div class="bento-tile" style="grid-column: span 4; grid-row: span 6; padding: 14px 16px;">
        <div>
          <div class="tile-kicker">Real-Time Local Slice Inspector</div>
          <div class="tile-title" id="probeTitle" style="font-size:1.02rem;">X = 122.0 Å · Sharp Transition</div>
          <p class="tile-desc" id="probeSubtitle" style="margin-bottom: 7px; font-size:0.77rem;">
            Narrow 12 Å structural interface separating diffuse Cu<sub>50</sub>Zr<sub>50</sub>/Cu<sub>46</sub>Zr<sub>54</sub> from Cu<sub>64</sub>Zr<sub>36</sub>.
          </p>

          <div class="probe-metric" style="padding:5px 9px; margin-bottom:4px;">
            <div class="probe-row" style="font-size:0.73rem; margin-bottom:2px;">
              <span style="color:var(--cyan); font-weight:600;">Phase 0 (Cu<sub>50</sub>Zr<sub>50</sub>) Density</span>
              <span id="probeP0" style="font-family:'JetBrains Mono',monospace;">195 atoms/bin</span>
            </div>
            <div class="probe-bar">
              <div class="probe-fill" id="barProbe0" style="width:32%; background:var(--cyan);"></div>
            </div>
          </div>

          <div class="probe-metric" style="padding:5px 9px; margin-bottom:4px;">
            <div class="probe-row" style="font-size:0.73rem; margin-bottom:2px;">
              <span style="color:var(--violet); font-weight:600;">Phase 1 (Cu<sub>46</sub>Zr<sub>54</sub>) Density</span>
              <span id="probeP1" style="font-family:'JetBrains Mono',monospace;">110 atoms/bin</span>
            </div>
            <div class="probe-bar">
              <div class="probe-fill" id="barProbe1" style="width:18%; background:var(--violet);"></div>
            </div>
          </div>

          <div class="probe-metric" style="padding:5px 9px; margin-bottom:4px;">
            <div class="probe-row" style="font-size:0.73rem; margin-bottom:2px;">
              <span style="color:var(--amber); font-weight:600;">Phase 2 (Cu<sub>64</sub>Zr<sub>36</sub>) Density</span>
              <span id="probeP2" style="font-family:'JetBrains Mono',monospace;">340 atoms/bin</span>
            </div>
            <div class="probe-bar">
              <div class="probe-fill" id="barProbe2" style="width:56%; background:var(--amber);"></div>
            </div>
          </div>
        </div>

        <div
          style="display:grid; grid-template-columns:1fr 1fr; gap:6px; padding-top:6px; border-top:1px solid var(--border-dim);">
          <div style="background:rgba(255,255,255,0.02); padding:5px 8px; border-radius:8px;">
            <div style="font-size:0.62rem; color:var(--text-dim); font-family:'JetBrains Mono',monospace;">LOCAL VON MISES</div>
            <div id="probeVM"
              style="font-size:0.95rem; font-weight:700; font-family:'Space Grotesk',sans-serif; color:var(--emerald);">
              284 MPa</div>
          </div>
          <div style="background:rgba(255,255,255,0.02); padding:5px 8px; border-radius:8px;">
            <div style="font-size:0.62rem; color:var(--text-dim); font-family:'JetBrains Mono',monospace;">HYDROSTATIC PRESS.</div>
            <div id="probeHP"
              style="font-size:0.95rem; font-weight:700; font-family:'Space Grotesk',sans-serif; color:var(--cyan);">+18 MPa</div>
          </div>
        </div>
      </div>

      <!-- Right Bottom 4-Column Bento Tile: Physical Regime Contrast -->
      <div class="bento-tile" style="grid-column: span 4; grid-row: span 6; padding: 14px 16px;">
        <div>
          <div class="tile-kicker">Two Distinct Polyamorphous Regimes</div>
          <div class="tile-title" style="font-size:1.02rem;">Diffuse Gradient vs. Abrupt Boundary</div>
        </div>

        <div style="display:flex; flex-direction:column; gap:6px; margin:auto 0;">
          <div
            style="background:rgba(56,189,248,0.07); border-left:3px solid var(--cyan); padding:7px 10px; border-radius:0 8px 8px 0; font-size:0.75rem; line-height:1.4;">
            <strong style="color:var(--text-primary);">Left/Center (0 – 122 Å):</strong> Smooth ~40 Å compositional
            gradient between Cu<sub>50</sub>Zr<sub>50</sub> and Cu<sub>46</sub>Zr<sub>54</sub> (4 at.% difference, continuous regression <strong>R<sup>2</sup> =
              0.938</strong>).
          </div>
          <div
            style="background:rgba(251,146,60,0.08); border-left:3px solid var(--amber); padding:7px 10px; border-radius:0 8px 8px 0; font-size:0.75rem; line-height:1.4;">
            <strong style="color:var(--text-primary);">Right Domain (122 – 183 Å):</strong> Sharp 12 Å structural phase
            transition into Cu<sub>64</sub>Zr<sub>36</sub> with near-perfect separability (<strong>F1 = 0.9703</strong>).
          </div>
        </div>

        <div
          style="display:flex; justify-content:space-between; align-items:center; font-size:0.74rem; color:var(--text-muted); padding-top:6px; border-top:1px solid var(--border-dim); flex-shrink:0;">
          <span>Confidence Filter: <strong id="miniConfVal" style="color:var(--cyan);">≥ 0.50</strong></span>
          <button class="seg-pill active" onclick="switchView(1)">Open Confidence Lab →</button>
        </div>
      </div>"""

assert old_v0_right in text, "old_v0_right not found"
text = text.replace(old_v0_right, new_v0_right)

# 3. Move RDF hover readout to the bottom of the chart
old_rdf_hover = """      if (rdfHoverR !== null) {
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

new_rdf_hover = """      if (rdfHoverR !== null) {
        const hx = padL + (rdfHoverR / 12) * pw;
        const hg = sgModel(rdfHoverR);
        const hy = padT + ph - (hg / 4.6) * ph;

        rCtx.beginPath();
        rCtx.moveTo(hx, padT);
        rCtx.lineTo(hx, padT + ph);
        rCtx.strokeStyle = 'rgba(255,255,255,0.45)';
        rCtx.stroke();

        rCtx.beginPath();
        rCtx.arc(hx, hy, 4.5, 0, Math.PI * 2);
        rCtx.fillStyle = '#38bdf8';
        rCtx.fill();

        // Opaque pill badge placed cleanly at the bottom right of the plot (avoids top vertical lines)
        const badgeW = 220, badgeH = 25;
        const badgeX = W - padR - badgeW - 8;
        const badgeY = H - padB - badgeH - 6;

        rCtx.fillStyle = 'rgba(6, 10, 20, 0.94)';
        rCtx.beginPath();
        if (rCtx.roundRect) {
          rCtx.roundRect(badgeX, badgeY, badgeW, badgeH, 6);
        } else {
          rCtx.rect(badgeX, badgeY, badgeW, badgeH);
        }
        rCtx.fill();
        rCtx.strokeStyle = 'rgba(56, 189, 248, 0.55)';
        rCtx.lineWidth = 1.2;
        rCtx.stroke();

        rCtx.font = '600 11px "JetBrains Mono", monospace';
        rCtx.fillStyle = '#38bdf8';
        rCtx.fillText(`r = ${rdfHoverR.toFixed(2)} Å`, badgeX + 12, badgeY + 16);
        rCtx.fillStyle = '#f8fafc';
        rCtx.fillText(`│ g(r) = ${hg.toFixed(2)}`, badgeX + 110, badgeY + 16);
      }"""

assert old_rdf_hover in text, "old_rdf_hover not found"
text = text.replace(old_rdf_hover, new_rdf_hover)

# Write to docs/index.html and root index.html
with open('docs/index.html', 'w', encoding='utf-8') as f:
    f.write(text)

with open('index.html', 'w', encoding='utf-8') as f:
    f.write(text)

print("All refinements applied cleanly!")
