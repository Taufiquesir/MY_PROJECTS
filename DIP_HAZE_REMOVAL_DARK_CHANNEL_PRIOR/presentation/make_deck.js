// Build presentation/DIP_Project_Presentation.pptx
// Run: NODE_PATH=<dir with pptxgenjs, react-icons, react, react-dom, sharp>/node_modules node presentation/make_deck.js
// All numbers come from results/*.csv (see IMPLEMENTATION_LOG.md); figures from figures/ and presentation/assets/.
const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");
// Optional helper that writes the theme colours into the .pptx. Point APPLY_THEME at a module
// exporting applyTheme(file, theme); without it the deck still builds, but the colours are not embedded.
let applyTheme = null;
if (process.env.APPLY_THEME) {
  applyTheme = require(process.env.APPLY_THEME).applyTheme;
} else {
  console.warn("APPLY_THEME not set: theme colours will not be embedded in the .pptx.");
}

const ROOT = path.join(__dirname, "..");
const OUT = path.join(__dirname, "DIP_Project_Presentation.pptx");

// ---- theme: "fog and low sun" ------------------------------------------------
const THEME = {
  name: "Fog and Low Sun",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "17283A", lt1: "FFFFFF", dk2: "2F4858", lt2: "EDF1F4",
    accent1: "E8A33D", accent2: "4F7CAC", accent3: "8FA3B3",
    accent4: "C0503A", accent5: "4E8F73", accent6: "B9C6D1",
    hlink: "4F7CAC", folHlink: "8FA3B3",
  },
};
const HEX = THEME.colors; // charts need hex

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Single-Image Haze Removal via Dark Channel Prior, with a Temporally Consistent Video Extension";
pres.author = "Sohail Memon, Kathan Patel, Md Taufique Hussain";
const C = pres.SchemeColor;

// ---- layouts -----------------------------------------------------------------
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: HEX.dk1 },
  objects: [],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: HEX.lt1 },
  slideNumber: { x: 9.2, y: 5.2, w: 0.5, h: 0.3, fontFace: "Calibri", fontSize: 10, color: HEX.accent3, align: "right" },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 9.0, h: 0.95,
        fontFace: "Cambria", fontSize: 28, bold: true, color: HEX.dk1, align: "left", valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CLOSE_DARK",
  background: { color: HEX.dk1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.35, w: 9.0, h: 0.8,
        fontFace: "Cambria", fontSize: 30, bold: true, color: HEX.lt1, align: "left", valign: "top", margin: 0 }, text: "" } },
  ],
});

// ---- helpers -----------------------------------------------------------------
async function icon(Comp, colorHex, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + colorHex, size: String(size) }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}
const shadow = () => ({ type: "outer", color: "000000", opacity: 0.12, blur: 8, offset: 2, angle: 90 });

function badge(slide, data, x, y, d, fillColor, name) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fillColor }, line: { color: fillColor }, objectName: name + " badge" });
  const p = d * 0.26;
  slide.addImage({ data, x: x + p, y: y + p, w: d - 2 * p, h: d - 2 * p, objectName: name + " icon", altText: name });
}
function card(slide, x, y, w, h, name, fill = C.background2) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill },
    shadow: shadow(), objectName: name });
}
function stat(slide, x, y, w, big, label, color, bigSize = 40) {
  slide.addText(big, { x, y, w, h: 0.75, fontFace: "Cambria", fontSize: bigSize, bold: true, color, margin: 0, valign: "bottom", isTextBox: true, objectName: "stat " + label });
  slide.addText(label, { x, y: y + 0.78, w, h: 0.5, fontFace: "Calibri", fontSize: 14, color: C.text2, margin: 0, valign: "top", isTextBox: true, objectName: "stat label " + label });
}
function chartBase(extra) {
  return Object.assign({
    catAxisLabelColor: HEX.dk2, valAxisLabelColor: HEX.dk2, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
    catAxisLabelFontSize: 12, valAxisLabelFontSize: 11, dataLabelFontFace: "+mn-lt", dataLabelColor: HEX.dk1, dataLabelFontSize: 12,
    legendFontFace: "+mn-lt", legendFontSize: 12, legendColor: HEX.dk2,
    valGridLine: { color: "D5DDE3", size: 0.75 }, catGridLine: { style: "none" },
    titleFontFace: "+mn-lt", titleFontSize: 13, titleColor: HEX.dk2,
  }, extra);
}
function notes(slide, text) { slide.addNotes(text); }

async function main() {
  ["Title", "Problem", "Single image", "Video", "Failure case", "Conclusions"].forEach((t) => pres.addSection({ title: t }));
  const iCloud = await icon(fa.FaCloudSun, HEX.lt1);
  const iQ1 = await icon(fa.FaBullseye, HEX.lt1);
  const iQ2 = await icon(fa.FaVideo, HEX.lt1);
  const iQ3 = await icon(fa.FaSlidersH, HEX.lt1);
  const iCheck = await icon(fa.FaCheckCircle, HEX.lt1);
  const iBan = await icon(fa.FaTimesCircle, HEX.lt1);
  const iWarn = await icon(fa.FaExclamationTriangle, HEX.lt1);
  const iCog = await icon(fa.FaCog, HEX.lt1);

  // ===== 1. Title ==============================================================
  let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Title" });
  s.addImage({ path: path.join(__dirname, "assets/title_split.png"), x: 5.6, y: 0, w: 4.4, h: 5.625,
    sizing: { type: "cover", w: 4.4, h: 5.625 }, altText: "Outdoor scene: hazy on the left half, dehazed on the right half", objectName: "title image" });
  s.addText("Single-Image Haze Removal via Dark Channel Prior", { x: 0.5, y: 1.0, w: 4.8, h: 1.7, fontFace: "Cambria", fontSize: 32, bold: true,
    color: C.background1, margin: 0, valign: "top", isTextBox: true, objectName: "deck title" });
  s.addText("with a temporally consistent video extension", { x: 0.5, y: 2.8, w: 4.8, h: 0.6, fontFace: "Calibri", fontSize: 18, italic: true,
    color: HEX.accent1, margin: 0, valign: "top", isTextBox: true, objectName: "deck subtitle" });
  s.addText([
    { text: "Sohail Memon · Kathan Patel · Md Taufique Hussain", options: { breakLine: true } },
    { text: "CS7.404 Digital Image Processing" },
  ], { x: 0.5, y: 4.35, w: 4.8, h: 0.8, fontFace: "Calibri", fontSize: 14, color: HEX.accent6, margin: 0, valign: "top", isTextBox: true, objectName: "authors" });
  notes(s, "Left half of the picture is the hazy input, right half our dehazed output (SOTS outdoor, classical DCP, no learning). Three students, CS7.404. Everything in this talk is classical image processing: no learned component in the dehazing method itself.");

  // ===== 2. Problem ============================================================
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Problem" });
  s.addText("One hazy image has too many unknowns; a statistical prior closes the gap", { placeholder: "title" });
  card(s, 0.5, 1.45, 3.9, 3.45, "model card", C.text1);
  badge(s, iCloud, 0.75, 1.65, 0.62, HEX.accent1, "haze model");
  s.addText("The scattering model", { x: 1.5, y: 1.7, w: 2.7, h: 0.5, fontFace: "Cambria", fontSize: 18, bold: true, color: C.background1, margin: 0, valign: "middle", isTextBox: true, objectName: "model heading" });
  s.addText("I = J·t + A·(1 − t)", { x: 0.75, y: 2.5, w: 3.4, h: 0.6, fontFace: "Cambria", fontSize: 26, bold: true, color: HEX.accent1, margin: 0, isTextBox: true, objectName: "model formula" });
  s.addText("Seen: I. Wanted: J. Unknown per pixel: A and t. Under-constrained from one image.", {
    x: 0.75, y: 3.15, w: 3.4, h: 0.75, fontFace: "Calibri", fontSize: 14, color: HEX.lt2, margin: 0, valign: "top", isTextBox: true, objectName: "model text" });
  s.addText("Dark Channel Prior (He, Sun, Tang, CVPR 2009): in haze-free outdoor patches, one colour channel is almost always dark.", {
    x: 0.75, y: 4.0, w: 3.4, h: 0.8, fontFace: "Calibri", fontSize: 12, italic: true, color: HEX.accent6, margin: 0, valign: "top", isTextBox: true, objectName: "prior text" });

  const qs = [
    [iQ1, "How good is our implementation?", "Full DCP pipeline, validated on synthetic haze and benchmarked on RESIDE SOTS."],
    [iQ2, "What makes video flicker?", "Per-frame DCP jitters because A and t come from extremal statistics."],
    [iQ3, "Can classical smoothing fix it?", "Four variants V0–V3, scored against clean ground-truth video (REVIDE)."],
  ];
  qs.forEach((q, i) => {
    const y = 1.45 + i * 1.2;
    card(s, 4.7, y, 4.8, 1.05, "question card " + (i + 1));
    badge(s, q[0], 4.9, y + 0.2, 0.65, HEX.accent2, "question " + (i + 1));
    s.addText(q[1], { x: 5.75, y: y + 0.12, w: 3.6, h: 0.35, fontFace: "Calibri", fontSize: 16, bold: true, color: C.text1, margin: 0, valign: "top", isTextBox: true, objectName: "question title " + (i + 1) });
    s.addText(q[2], { x: 5.75, y: y + 0.48, w: 3.6, h: 0.5, fontFace: "Calibri", fontSize: 13, color: C.text2, margin: 0, valign: "top", isTextBox: true, objectName: "question text " + (i + 1) });
  });
  notes(s, "Haze model I = J t + A(1-t). DCP supplies the extra constraint. Three questions drive the talk: how good is our implementation, what makes video flicker, can classical temporal smoothing fix it. No learned component anywhere in the dehazing method.");

  // ===== 3. Pipeline ===========================================================
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Single image" });
  s.addText("Five classical stages, every intermediate output exposed", { placeholder: "title" });
  const steps = ["Dark channel", "Atmospheric light A", "Raw transmission t", "Refine t", "Recover J"];
  const sw = 1.66, sg = 0.175;
  steps.forEach((t, i) => {
    const x = 0.5 + i * (sw + sg);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.4, w: sw, h: 0.55, rectRadius: 0.08, fill: { color: i === 3 ? HEX.accent1 : HEX.dk2 }, line: { color: i === 3 ? HEX.accent1 : HEX.dk2 }, objectName: "step " + (i + 1) });
    s.addText(`${i + 1}  ${t}`, { x, y: 1.4, w: sw, h: 0.55, fontFace: "Calibri", fontSize: 12, bold: true, color: i === 3 ? HEX.dk1 : HEX.lt1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "step label " + (i + 1) });
  });
  s.addImage({ path: path.join(ROOT, "figures/single_image_stages.png"), x: 0.5, y: 2.15, w: 5.5, h: 2.96,
    altText: "Six panels for one image: hazy input, dark channel, raw transmission, refined transmission, recovered image, ground truth", objectName: "stages figure" });
  card(s, 6.3, 2.15, 3.2, 2.96, "pipeline notes");
  s.addText([
    { text: "Refiners compared", options: { bold: true, color: C.text1, breakLine: true } },
    { text: "Soft matting · guided filter · fast guided filter", options: { color: C.text2, breakLine: true, paraSpaceAfter: 10 } },
    { text: "Synthetic round trip", options: { bold: true, color: C.text1, breakLine: true } },
    { text: "Known A and t recovered; A within 0.044 per channel", options: { color: C.text2, breakLine: true, paraSpaceAfter: 10 } },
    { text: "Watch the sky", options: { bold: true, color: HEX.accent4, breakLine: true } },
    { text: "Recovered J over-saturates it: foreshadows the failure case", options: { color: C.text2 } },
  ], { x: 6.5, y: 2.3, w: 2.85, h: 2.7, fontFace: "Calibri", fontSize: 13, margin: 0, valign: "top", isTextBox: true, objectName: "pipeline notes text" });
  notes(s, "Walk the figure left to right: the dark channel shows the patch-min block structure; the raw t inherits it; guided refinement snaps t to image edges; J is recovered with t floored at 0.1. Note the over-saturated sky in J: that is the failure case later. The synthetic round-trip test recovers A within 0.044 per channel.");

  // ===== 4. Refinement results ==================================================
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Single image" });
  s.addText("Refinement adds 1.3–1.9 dB; soft matting costs ~100× more for no gain", { placeholder: "title" });
  stat(s, 0.5, 1.5, 2.1, "19.82", "dB, indoor (SOTS, 500 images)", HEX.accent2);
  stat(s, 0.5, 2.95, 2.1, "17.03", "dB, outdoor (SOTS, 500 images)", HEX.accent2);
  s.addText("Defaults: patch 15, ω 0.95, t₀ 0.1, guided filter", { x: 0.5, y: 4.4, w: 2.2, h: 0.6, fontFace: "Calibri", fontSize: 12, color: C.text2, margin: 0, valign: "top", isTextBox: true, objectName: "defaults caption" });
  s.addChart(pres.charts.BAR, [
    { name: "Indoor", labels: ["none", "guided", "fast guided", "matting"], values: [18.43, 20.11, 20.10, 20.01] },
    { name: "Outdoor", labels: ["none", "guided", "fast guided", "matting"], values: [16.23, 17.59, 17.56, 17.48] },
  ], chartBase({ x: 2.9, y: 1.4, w: 6.6, h: 3.3, barDir: "col", barGrouping: "clustered", chartColors: [HEX.accent2, HEX.accent1],
    showTitle: true, title: "PSNR (dB), fixed 25-image subset per split — axis starts at 14", showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0",
    valAxisMinVal: 14, valAxisMaxVal: 22, valAxisMajorUnit: 2, showLegend: true, legendPos: "b", barGapWidthPct: 60 }));
  s.addText("Seconds per image:   none 0.2   ·   guided 0.3   ·   fast guided 0.2   ·   matting ≈ 38", { x: 2.9, y: 4.75, w: 6.6, h: 0.35, fontFace: "Calibri", fontSize: 13, color: C.text2, margin: 0, valign: "middle", isTextBox: true, objectName: "runtime caption" });
  notes(s, "Defaults on all 500 images per split: indoor 19.82 dB / 0.859, outdoor 17.03 / 0.859. The chart is a fixed 25-image subset because matting is ~35-40 s per image; the axis starts at 14 dB so differences are visible, which exaggerates them. Matting is about 0.1 dB below guided and ~100x slower. Fast guided matches guided within 0.04 dB at ~60% of the time. Timings were taken on a shared CPU, so only the relative ordering is reliable. Published DCP numbers vary 16.95-20.15 dB indoor across papers, so we only claim ours is inside that spread.");

  // ===== 5. Sensitivity ========================================================
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Single image" });
  s.addText("The paper's defaults are not PSNR-optimal on SOTS, and we did not tune on test", { placeholder: "title" });
  s.addChart(pres.charts.BAR, [
    { name: "Indoor", labels: ["0.80", "0.90", "0.95", "1.00"], values: [20.49, 20.73, 19.82, 18.06] },
    { name: "Outdoor", labels: ["0.80", "0.90", "0.95", "1.00"], values: [21.26, 18.78, 17.03, 15.13] },
  ], chartBase({ x: 0.5, y: 1.4, w: 4.9, h: 3.6, barDir: "col", barGrouping: "clustered", chartColors: [HEX.accent2, HEX.accent1],
    showTitle: true, title: "PSNR (dB) vs ω, full 500 images per split — axis starts at 12", showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0",
    dataLabelFontSize: 11, valAxisMinVal: 12, valAxisMaxVal: 23, valAxisMajorUnit: 2, showLegend: true, legendPos: "b", barGapWidthPct: 50,
    showCatAxisTitle: true, catAxisTitle: "ω (default 0.95)", catAxisTitleColor: HEX.dk2, catAxisTitleFontSize: 12 }));
  const hdr = (t) => ({ text: t, options: { bold: true, color: HEX.lt1, fill: { color: HEX.dk2 }, fontFace: "Calibri", fontSize: 13, align: "center" } });
  const cell = (t, o = {}) => ({ text: t, options: Object.assign({ fontFace: "Calibri", fontSize: 13, color: HEX.dk1, align: "center" }, o) });
  s.addTable([
    [hdr("Factor"), hdr("Indoor"), hdr("Outdoor")],
    [cell("patch 7", { align: "left" }), cell("17.5"), cell("15.0")],
    [cell("patch 15 (default)", { align: "left", bold: true }), cell("19.8", { bold: true }), cell("17.0", { bold: true })],
    [cell("patch 31", { align: "left" }), cell("21.4"), cell("19.2")],
    [cell("t₀ 0.1 (default)", { align: "left", bold: true }), cell("19.8", { bold: true }), cell("17.0", { bold: true })],
    [cell("t₀ 0.3", { align: "left" }), cell("20.3"), cell("18.4")],
    [cell("A: brightest (default)", { align: "left", bold: true }), cell("19.8", { bold: true }), cell("17.0", { bold: true })],
    [cell("A: mean of candidates", { align: "left" }), cell("19.9"), cell("17.4")],
  ], { x: 5.7, y: 1.45, w: 3.8, colW: [1.9, 0.95, 0.95], rowH: 0.36, border: { type: "solid", pt: 0.5, color: "D5DDE3" }, fill: { color: HEX.lt1 }, objectName: "sensitivity table" });
  s.addText("PSNR in dB, one factor changed at a time. Plausible reason (not verified): SOTS haze is synthetic and uniform, and PSNR rewards conservative dehazing. We keep the paper defaults as the baseline.", {
    x: 5.7, y: 4.4, w: 3.8, h: 0.7, fontFace: "Calibri", fontSize: 12, italic: true, color: C.text2, margin: 0, valign: "top", isTextBox: true, objectName: "sensitivity caption" });
  notes(s, "One-factor-at-a-time on the full 500 images per split. Larger patches, smaller omega and a larger t0 all score higher than the paper defaults on SOTS. Picking a best setting from these numbers would be tuning on the test set, so we report the defaults as the baseline and show this as a sensitivity study. It says nothing about visual quality or real haze. The mean-of-candidates A estimator is slightly better than the brightest pixel.");

  // ===== 6. Variants ===========================================================
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Video" });
  s.addText("Four variants isolate what each kind of temporal smoothing buys", { placeholder: "title" });
  const vs = [
    ["V0", "Per-frame DCP", "Baseline: nothing shared between frames", HEX.accent3, HEX.dk1],
    ["V1", "EMA on A", "Smooth the atmospheric light; re-estimate t under it", HEX.accent2, HEX.lt1],
    ["V2", "V1 + EMA on t", "Naive: blend t with the previous frame's t", HEX.dk2, HEX.lt1],
    ["V3", "V1 + warped EMA on t", "Flow-warp previous t; forward-backward occlusion mask", HEX.accent1, HEX.dk1],
  ];
  vs.forEach((v, i) => {
    const x = 0.5 + i * 2.28;
    card(s, x, 1.45, 2.1, 2.35, "variant card " + v[0]);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x + 0.2, y: 1.65, w: 0.8, h: 0.5, rectRadius: 0.08, fill: { color: v[3] }, line: { color: v[3] }, objectName: v[0] + " tag" });
    s.addText(v[0], { x: x + 0.2, y: 1.65, w: 0.8, h: 0.5, fontFace: "Cambria", fontSize: 18, bold: true, color: v[4], align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: v[0] + " tag text" });
    s.addText(v[1], { x: x + 0.2, y: 2.25, w: 1.75, h: 0.55, fontFace: "Calibri", fontSize: 15, bold: true, color: C.text1, margin: 0, valign: "top", isTextBox: true, objectName: v[0] + " name" });
    s.addText(v[2], { x: x + 0.2, y: 2.85, w: 1.75, h: 0.85, fontFace: "Calibri", fontSize: 13, color: C.text2, margin: 0, valign: "top", isTextBox: true, objectName: v[0] + " description" });
  });
  card(s, 0.5, 4.0, 9.0, 1.1, "metric card", C.text1);
  s.addText([
    { text: "Flicker index", options: { bold: true, color: HEX.accent1 } },
    { text: " = flow-warped frame difference ÷ the clean video's own   (1.0 = natural change; higher = spurious flicker)", options: { color: HEX.lt1, breakLine: true } },
    { text: "REVIDE indoor test split: 6 clips, width 640. Camera moves 5–28 px/frame, so the proposal's static-ROI variance became a grid-block error variance.", options: { color: HEX.accent6, fontSize: 12 } },
  ], { x: 0.75, y: 4.05, w: 8.5, h: 1.0, fontFace: "Calibri", fontSize: 14, margin: 0, valign: "middle", isTextBox: true, objectName: "metric text" });
  notes(s, "V0 baseline, V1 smooth A, V2 add naive t smoothing, V3 add flow-warped t smoothing with a forward-backward occlusion mask; alpha is the weight on history. Flicker index is measured with flow computed on the clean ground-truth clip so it does not depend on dehazing artefacts. Deviation to mention: the camera moves, so the proposal's static-ROI variance cannot be computed; we use a grid-block error variance. Indoor only: REVIDE outdoor is unreleased.");

  // ===== 7. Video results ======================================================
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Video" });
  s.addText("Smoothing A cuts flicker; smoothing t does not", { placeholder: "title" });
  s.addChart(pres.charts.BAR, [
    { name: "α = 0.5", labels: ["V0", "V1", "V2", "V3"], values: [2.04, 1.77, 2.21, 2.07] },
    { name: "α = 0.7", labels: ["V0", "V1", "V2", "V3"], values: [2.04, 1.69, 2.25, 2.12] },
    { name: "α = 0.9", labels: ["V0", "V1", "V2", "V3"], values: [2.04, 1.58, 2.19, 2.17] },
  ], chartBase({ x: 0.5, y: 1.35, w: 5.9, h: 3.75, barDir: "col", barGrouping: "clustered", chartColors: [HEX.accent6, HEX.accent3, HEX.accent2],
    showTitle: true, title: "Flicker index (lower is better; clean video = 1.0), mean of 6 clips", showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00",
    dataLabelFontSize: 10, valAxisMinVal: 0, valAxisMaxVal: 2.6, valAxisMajorUnit: 0.5, showLegend: true, legendPos: "b", barGapWidthPct: 50 }));
  stat(s, 6.75, 1.45, 2.75, "2.04 → 1.58", "V1 vs V0 at α = 0.9", HEX.accent5, 30);
  stat(s, 6.75, 2.85, 2.75, "≈ 2× slower", "V3 runs at 5.5 FPS vs 10 for V0 (CPU, width 640)", HEX.accent4, 30);
  s.addText("V3 with ground-truth flow is also worse than V1, so flow quality is not the cause. We found no verified mechanism.", {
    x: 6.75, y: 4.3, w: 2.75, h: 0.85, fontFace: "Calibri", fontSize: 13, italic: true, color: C.text2, margin: 0, valign: "top", isTextBox: true, objectName: "v3 caveat" });
  notes(s, "Only V1 clearly helps: 2.04 to 1.58 at alpha 0.9. V2 and V3 do not reduce flicker; V3 is about twice as slow. V3 with ground-truth flow is also worse than V1, so it is not a flow-quality problem; we tested and rejected exposure jitter in the hazy input, and we do not have a verified mechanism. Be upfront that this is a negative result. PSNR is low (~12.7 dB for every variant) because DCP handles dense indoor haze poorly. Six clips, no significance tests.");

  // ===== 8. Causes of flicker ==================================================
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Video" });
  s.addText("Atmospheric-light jitter is the largest removable cause; 1/t noise gain also shows", { placeholder: "title" });
  s.addChart(pres.charts.BAR, [
    { name: "Flicker index", labels: ["V0 baseline", "A frozen", "A frozen + t smoothed", "t₀ = 0.3", "t₀ = 0.05"], values: [2.04, 1.58, 1.94, 1.99, 2.04] },
  ], chartBase({ x: 0.5, y: 1.4, w: 5.3, h: 3.65, barDir: "bar", chartColors: [HEX.accent2], showTitle: true, title: "Intervention on V0: flicker index (lower is better)",
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", valAxisMinVal: 0, valAxisMaxVal: 2.5, valAxisMajorUnit: 0.5, showLegend: false,
    catAxisOrientation: "maxMin", barGapWidthPct: 45 }));
  card(s, 6.1, 1.45, 3.4, 3.7, "quartile card");
  s.addText("Where the flicker lives", { x: 6.3, y: 1.6, w: 3.0, h: 0.4, fontFace: "Cambria", fontSize: 16, bold: true, color: C.text1, margin: 0, valign: "top", isTextBox: true, objectName: "quartile heading" });
  stat(s, 6.3, 1.95, 3.0, "≈ 3×", "natural flicker in the haziest quartile of t", HEX.accent4, 36);
  stat(s, 6.3, 3.15, 3.0, "≈ 1.3×", "in the clearest quartile", HEX.accent5, 36);
  s.addText("Consistent with 1/t noise amplification, a fourth cause beyond the proposal's three; not isolated by a direct intervention.", {
    x: 6.3, y: 4.3, w: 3.0, h: 0.75, fontFace: "Calibri", fontSize: 12, italic: true, color: C.text2, margin: 0, valign: "top", isTextBox: true, objectName: "quartile caveat" });
  notes(s, "Interventions on V0, mean over 6 clips. Freezing A to the clip median drops the flicker index from 2.04 to 1.58; the effect is strong on some clips (J004, W002, L006) and small on others (C005, E005). Raising t0 barely helps, so boundary clipping is a minor cause; smoothing t on top of frozen A adds flicker. Splitting V0's flicker by quartile of t: 0.0289, 0.0208, 0.0150, 0.0124 against 0.0094 for the clean video, i.e. about 3x in the haziest quartile and 1.3x in the clearest. That fits noise amplification by 1/t, but we did not isolate it directly.");

  // ===== 9. Before / after =====================================================
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Video" });
  s.addText("Before and after on clip J004: smoother t, but no less flicker", { placeholder: "title" });
  s.addImage({ path: path.join(ROOT, "figures/transmission_J004.png"), x: 0.9, y: 1.3, w: 8.2, h: 3.2,
    altText: "Top row: hazy frame, V0 output, V3 output, ground truth. Bottom row: transmission maps for V0, V1, V3", objectName: "before after figure" });
  s.addText("Full side-by-side clip (hazy | V0 | V1 | V3 | GT): figures/demo_J004.mp4. V3's transmission map is visibly smoother, but that does not lower flicker in the recovered video.", {
    x: 0.9, y: 4.6, w: 8.2, h: 0.55, fontFace: "Calibri", fontSize: 13, color: C.text2, margin: 0, valign: "top", isTextBox: true, objectName: "before after caption" });
  notes(s, "Play figures/demo_J004.mp4 if time allows: hazy, V0, V1, V3, GT side by side. The maps show transmission for V0, V1, V3. V3 is visibly smoother but that does not translate into lower flicker in J. Note the darkening typical of DCP on indoor scenes.");

  // ===== 10. Sky failure =======================================================
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Failure case" });
  s.addText("Sky breaks the dark-channel prior; a transmission floor wins back 2.4 dB", { placeholder: "title" });
  stat(s, 0.5, 1.5, 2.4, "r = −0.50", "PSNR vs sky fraction, 200 outdoor images", HEX.accent4, 32);
  s.addText("Sky has no dark channel, so t is underestimated and the region is over-amplified.", { x: 0.5, y: 2.95, w: 2.4, h: 1.0, fontFace: "Calibri", fontSize: 13, color: C.text2, margin: 0, valign: "top", isTextBox: true, objectName: "sky explanation" });
  s.addText("Mitigation: raise t to 0.6 inside the detected sky region (classical detector, no learning).", { x: 0.5, y: 3.95, w: 2.4, h: 1.1, fontFace: "Calibri", fontSize: 13, bold: true, color: C.text1, margin: 0, valign: "top", isTextBox: true, objectName: "mitigation text" });
  s.addChart(pres.charts.BAR, [
    { name: "Baseline DCP", labels: ["< 5% sky (54)", "5–25% (82)", "≥ 25% (64)"], values: [18.26, 17.78, 14.47] },
    { name: "With sky floor", labels: ["< 5% sky (54)", "5–25% (82)", "≥ 25% (64)"], values: [18.35, 18.76, 16.87] },
  ], chartBase({ x: 3.2, y: 1.4, w: 6.3, h: 3.3, barDir: "col", barGrouping: "clustered", chartColors: [HEX.accent3, HEX.accent1],
    showTitle: true, title: "PSNR (dB) by sky fraction (n images) — axis starts at 12", showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0",
    valAxisMinVal: 12, valAxisMaxVal: 20, valAxisMajorUnit: 2, showLegend: true, legendPos: "b", barGapWidthPct: 60 }));
  s.addText("Caveat: the floor value and thresholds were chosen once, not tuned, but scored on the same images used for the analysis, so the gain is somewhat optimistic.", {
    x: 3.2, y: 4.75, w: 6.3, h: 0.45, fontFace: "Calibri", fontSize: 12, italic: true, color: C.text2, margin: 0, valign: "top", isTextBox: true, objectName: "sky caveat" });
  notes(s, "DCP assumes some channel is dark in every patch; sky violates that. PSNR correlates with sky fraction at r = -0.50 over 200 fixed SOTS-outdoor images. A sky-aware transmission floor of 0.6 lifts images with at least 25% sky from 14.47 to 16.87 dB and barely changes images without sky. The floor and thresholds were chosen once, not tuned, but evaluated on the same images used for the analysis; a held-out check was not done. The y axis starts at 12 dB.");

  // ===== 11. Conclusions =======================================================
  s = pres.addSlide({ masterName: "CLOSE_DARK", sectionTitle: "Conclusions" });
  s.addText("What we learned, and what we cannot claim", { placeholder: "title" });
  const cc = [
    [iCheck, HEX.accent5, "What worked", "Refinement adds 1.3–1.9 dB. Smoothing A cuts flicker (2.04 → 1.58). A sky floor lifts sky-heavy images by 2.4 dB."],
    [iBan, HEX.accent4, "What did not", "Flow-warped smoothing of t never beat smoothing A alone, even with ground-truth flow. No verified mechanism."],
    [iWarn, HEX.accent1, "Limits", "Indoor-only video, 6 clips, no significance tests, mitigation scored on its analysis set, CPU timings under load."],
  ];
  cc.forEach((c, i) => {
    const x = 0.5 + i * 3.05;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.5, w: 2.9, h: 2.75, rectRadius: 0.08, fill: { color: HEX.dk2 }, line: { color: HEX.dk2 }, objectName: "conclusion card " + (i + 1) });
    badge(s, c[0], x + 0.25, 1.7, 0.6, c[1], "conclusion " + (i + 1));
    s.addText(c[2], { x: x + 0.25, y: 2.45, w: 2.4, h: 0.4, fontFace: "Cambria", fontSize: 18, bold: true, color: HEX.lt1, margin: 0, valign: "top", isTextBox: true, objectName: "conclusion title " + (i + 1) });
    s.addText(c[3], { x: x + 0.25, y: 2.9, w: 2.4, h: 1.4, fontFace: "Calibri", fontSize: 13, color: HEX.lt2, margin: 0, valign: "top", isTextBox: true, objectName: "conclusion text " + (i + 1) });
  });
  badge(s, iCog, 0.5, 4.6, 0.5, HEX.accent2, "reproducibility");
  s.addText("Everything regenerates with scripts/run_all.sh (verified from empty results/); design decisions and deviations are in IMPLEMENTATION_LOG.md.", {
    x: 1.25, y: 4.55, w: 8.25, h: 0.6, fontFace: "Calibri", fontSize: 13, color: HEX.accent6, margin: 0, valign: "middle", isTextBox: true, objectName: "reproducibility text" });
  notes(s, "Take-aways: refinement adds 1.3-1.9 dB; stabilise A for video; sky is a quantifiable failure mode a simple floor partly repairs. Limits: indoor-only video, six clips, no significance tests, mitigation scored on the same images used to analyse it, CPU timings under load. Everything regenerates with scripts/run_all.sh; details and deviations are in IMPLEMENTATION_LOG.md. Invite questions.");

  await pres.writeFile({ fileName: OUT });
  if (applyTheme) await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
}
main().catch((e) => { console.error(e); process.exit(1); });
