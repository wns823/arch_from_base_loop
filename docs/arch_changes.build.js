const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.theme = { headFontFace: "Calibri", bodyFontFace: "Calibri" };
const s = pres.addSlide();
s.background = { color: "FFFFFF" };
const INK = "1F2933", GRAY = "7B8794", LINE = "9AA5B1";
const OLD = { fill: "F3F5F7", line: "CBD2D9", text: INK };
const NEW = { fill: "FBF0E6", line: "C8691C", text: "8A4510" };
const KEEP = { fill: "EAF1F7", line: "2F6690", text: INK };

function box(x, y, w, h, t, st, fs = 13) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.06, fill: { color: st.fill }, line: { color: st.line, width: st === NEW ? 2 : 1.25 } });
  s.addText(t, { x, y, w, h, fontSize: fs, color: st.text, align: "center", valign: "middle", margin: 2, bold: st === NEW, isTextBox: true });
}
function arr(x1, y1, x2, y2) {
  s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.max(Math.abs(x2 - x1), 0.001), h: Math.max(Math.abs(y2 - y1), 0.001),
    flipV: y2 < y1, flipH: x2 < x1, line: { color: LINE, width: 1.5, endArrowType: "triangle" } });
}
function label(x, y, w, t, opt = {}) { s.addText(t, Object.assign({ x, y, w, h: 0.35, fontSize: 12, color: GRAY, margin: 0, isTextBox: true }, opt)); }

// headers
s.addText("Before (base model)", { x: 1.75, y: 0.15, w: 5.0, h: 0.4, fontSize: 18, bold: true, color: GRAY, align: "center", margin: 0, isTextBox: true });
s.addText("After (found by the search)", { x: 7.45, y: 0.15, w: 5.6, h: 0.4, fontSize: 18, bold: true, color: "C8691C", align: "center", margin: 0, isTextBox: true });
const rows = [0.7, 2.95, 5.2];
const names = [["XChrom", "unseen cells"], ["ZEN", "unseen mice"], ["scooby", "cell types"]];
rows.forEach((y, i) => {
  s.addText([{ text: names[i][0], options: { bold: true, fontSize: 17, color: INK, breakLine: true } }, { text: names[i][1], options: { fontSize: 12, color: GRAY } }],
    { x: 0.2, y: y + 0.45, w: 1.4, h: 0.9, margin: 0, valign: "middle", isTextBox: true });
  s.addShape(pres.shapes.LINE, { x: 6.95, y: y + 0.85, w: 0.4, h: 0, line: { color: "C8691C", width: 3, endArrowType: "triangle" } });
  if (i > 0) s.addShape(pres.shapes.LINE, { x: 0.2, y: y - 0.12, w: 12.9, h: 0, line: { color: "E4E7EB", width: 1 } });
});

// ---- XChrom
let y = rows[0];
box(1.75, y + 0.3, 1.0, 0.55, "DNA seq", OLD); arr(2.75, y + 0.575, 3.0, y + 0.575);
box(3.0, y + 0.3, 0.8, 0.55, "CNN", OLD); arr(3.8, y + 0.575, 4.25, y + 0.575);
box(4.25, y + 0.3, 0.6, 0.55, "dot", OLD); arr(4.85, y + 0.575, 5.15, y + 0.575);
box(5.15, y + 0.3, 1.55, 0.55, "accessibility", OLD);
box(3.55, y + 1.15, 2.0, 0.6, "lookup table:\none vector per training cell", OLD, 11); arr(4.55, y + 1.15, 4.55, y + 0.85);
label(1.75, y + 1.85, 5.0, "a cell not seen in training has no vector", { italic: true });
box(7.45, y + 0.3, 1.0, 0.55, "DNA seq", OLD); arr(8.45, y + 0.575, 8.7, y + 0.575);
box(8.7, y + 0.3, 0.8, 0.55, "CNN", OLD); arr(9.5, y + 0.575, 9.85, y + 0.575);
box(9.85, y + 0.3, 1.0, 0.55, "FiLM", NEW); arr(10.85, y + 0.575, 11.45, y + 0.575);
box(11.45, y + 0.3, 1.55, 0.55, "accessibility", OLD);
box(7.45, y + 1.15, 1.0, 0.55, "cell RNA", OLD); arr(8.45, y + 1.425, 8.7, y + 1.425);
box(8.7, y + 1.15, 1.35, 0.55, "MLP encoder", NEW); arr(10.05, y + 1.425, 10.35, y + 1.425); arr(10.35, y + 1.425, 10.35, y + 0.85);
box(10.6, y + 1.15, 1.25, 0.55, "seq. depth", NEW); arr(11.85, y + 1.425, 12.0, y + 1.425); arr(12.0, y + 1.425, 12.0, y + 0.85);
label(7.45, y + 1.85, 5.6, "any cell with an RNA profile can be predicted", { italic: true, color: "8A4510" });

// ---- ZEN
y = rows[1];
box(1.75, y + 0.3, 0.85, 0.55, "spikes", OLD, 12); arr(2.6, y + 0.575, 2.75, y + 0.575);
box(2.75, y + 0.3, 1.45, 0.55, "per-session unit vectors", OLD, 11); arr(4.2, y + 0.575, 4.35, y + 0.575);
box(4.35, y + 0.3, 0.6, 0.55, "sum", OLD, 12); arr(4.95, y + 0.575, 5.1, y + 0.575);
box(5.1, y + 0.3, 0.9, 0.55, "trans-\nformer", OLD, 11); arr(6.0, y + 0.575, 6.1, y + 0.575);
box(6.1, y + 0.3, 0.75, 0.55, "wheel,\nreward", OLD, 11);
label(1.75, y + 1.0, 5.0, "every mouse/session needs its own trained\nunit vectors → cannot decode a new mouse", { italic: true, h: 0.6 });
box(7.45, y + 0.3, 0.85, 0.55, "spikes", OLD, 12); arr(8.3, y + 0.575, 8.5, y + 0.575);
box(8.5, y + 0.3, 1.0, 0.55, "unit\ntokens", NEW, 12); arr(9.5, y + 0.575, 9.7, y + 0.575);
box(9.7, y + 0.3, 1.25, 0.55, "set trans-\nformer", NEW, 12); arr(10.95, y + 0.575, 11.15, y + 0.575);
box(11.15, y + 0.3, 0.95, 0.55, "trans-\nformer", OLD, 11); arr(12.1, y + 0.575, 12.25, y + 0.575);
box(12.25, y + 0.3, 0.8, 0.55, "wheel,\nreward", OLD, 11);
box(7.45, y + 1.15, 1.25, 0.55, "ISI features\nof each unit", OLD, 11); arr(8.7, y + 1.425, 8.85, y + 1.425);
box(8.85, y + 1.15, 0.7, 0.55, "MLP", NEW, 12); arr(9.2, y + 1.15, 9.2, y + 0.85);
label(9.75, y + 1.2, 3.3, "unit vectors computed from activity:\nno per-mouse parameters", { italic: true, color: "8A4510", h: 0.6 });

// ---- scooby
y = rows[2];
box(1.75, y + 0.3, 1.6, 0.55, "frozen DNA trunk", KEEP, 12); arr(3.35, y + 0.575, 3.6, y + 0.575);
box(3.6, y + 0.3, 1.7, 0.55, "per-cell-type\n1×1 head", OLD, 11); arr(5.3, y + 0.575, 5.55, y + 0.575);
box(5.55, y + 0.3, 1.15, 0.55, "coverage", OLD, 12);
box(7.45, y + 0.3, 1.6, 0.55, "frozen DNA trunk", KEEP, 12); arr(9.05, y + 0.575, 9.3, y + 0.575);
box(9.3, y + 0.3, 1.7, 0.55, "per-cell-type\n1×1 head", OLD, 11); arr(11.0, y + 0.575, 11.25, y + 0.575);
box(11.25, y + 0.3, 1.15, 0.55, "coverage", OLD, 12);
label(7.45, y + 1.0, 5.6, "same modules; only hyperparameters tuned (explorer was starved in this run)", { italic: true });

// legend
box(9.3, 6.85, 0.45, 0.3, "", NEW); label(9.85, 6.83, 1.6, "new module");
box(11.0, 6.85, 0.45, 0.3, "", OLD); label(11.55, 6.83, 1.6, "kept from base");

pres.writeFile({ fileName: process.argv[2] }).then(() => console.log("written"));
