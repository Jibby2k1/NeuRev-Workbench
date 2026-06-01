// Focused review dashboard for calcium_video_2 parameter tuning.
//
// Generates an HTML dashboard plus frame montages from the current high-pass,
// candidate, and temporal scoring outputs. This is a reporting layer only; it
// does not modify the processing outputs.

import ij.IJ
import ij.ImagePlus
import java.awt.BasicStroke
import java.awt.Color
import java.awt.Font
import java.awt.RenderingHints
import java.awt.image.BufferedImage
import javax.imageio.ImageIO
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths

final Path projectRoot = Paths.get("/home/jibby2k1/CNEL/State Analysis (Fish)/Separable-Gamma-CFAR")
final Path rawPath = projectRoot.resolve("Inputs/050126/050126/calcium video 2.tif")
final Path highPassDir = projectRoot.resolve("Outputs/HighPass/calcium_video_2")
final Path candidateDir = projectRoot.resolve("Outputs/CandidateEventPipeline/calcium_video_2")
final Path temporalDir = projectRoot.resolve("Outputs/TemporalCandidateScoring/calcium_video_2")
final Path outputDir = projectRoot.resolve("Outputs/ReviewReports/calcium_video_2")
final Path montageDir = outputDir.resolve("montages")

final int maxTopFrames = 8
final int maxQuietFrames = 3
final int labelHeight = 24
final int leftLabelWidth = 118
final int gap = 6

final List<Map<String, Object>> variants = [
    [
        id: "sigma06_strict_score_ge_075",
        label: "sigma06 strict >= .75",
        sigma: "sigma06",
        preset: "strict",
        presetTag: "strict_seed020_grow011_min4",
        cutoff: "0.75",
        cutoffTag: "075",
        note: "Best first-look option: lowest clutter among the current focused outputs."
    ],
    [
        id: "sigma08_strict_score_ge_075",
        label: "sigma08 strict >= .75",
        sigma: "sigma08",
        preset: "strict",
        presetTag: "strict_seed020_grow011_min4",
        cutoff: "0.75",
        cutoffTag: "075",
        note: "Similar strict option with slower temporal background removed."
    ],
    [
        id: "sigma06_strict_score_ge_050",
        label: "sigma06 strict >= .50",
        sigma: "sigma06",
        preset: "strict",
        presetTag: "strict_seed020_grow011_min4",
        cutoff: "0.50",
        cutoffTag: "050",
        note: "More permissive strict setting to check whether .75 is missing sparse events."
    ],
    [
        id: "sigma08_balanced_score_ge_075",
        label: "sigma08 balanced >= .75",
        sigma: "sigma08",
        preset: "balanced",
        presetTag: "balanced_seed017_grow009_min3",
        cutoff: "0.75",
        cutoffTag: "075",
        note: "Tests whether balanced spatial candidates recover events without too much extra noise."
    ],
]

Files.createDirectories(montageDir)

static String htmlEscape(Object value) {
    String s = value == null ? "" : value.toString()
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\"", "&quot;")
}

static float[] pixelsAsFloat(ImagePlus imp, int frame) {
    return (float[]) imp.getStack().getProcessor(frame).convertToFloatProcessor().getPixels()
}

static byte[] pixelsAsByte(ImagePlus imp, int frame) {
    Object pixels = imp.getStack().getProcessor(frame).getPixels()
    if (pixels instanceof byte[]) {
        return (byte[]) pixels
    }
    byte[] out = new byte[imp.getWidth() * imp.getHeight()]
    float[] f = pixelsAsFloat(imp, frame)
    for (int i = 0; i < out.length; i++) {
        out[i] = (byte) ((f[i] as double) > 0.0d ? 255 : 0)
    }
    return out
}

static double percentile(float[] pixels, double fraction) {
    float[] copy = pixels.clone()
    Arrays.sort(copy)
    int idx = Math.max(0, Math.min(copy.length - 1, Math.round((copy.length - 1) * fraction) as int))
    return copy[idx] as double
}

static int clampByte(double value) {
    if (value < 0.0d) return 0
    if (value > 255.0d) return 255
    return Math.round(value) as int
}

static BufferedImage grayImage(float[] pixels, int width, int height, double lo, double hi) {
    BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
    double range = Math.max(1.0d, hi - lo)
    for (int y = 0; y < height; y++) {
        int row = y * width
        for (int x = 0; x < width; x++) {
            int v = clampByte(((pixels[row + x] as double) - lo) * 255.0d / range)
            int rgb = (v << 16) | (v << 8) | v
            img.setRGB(x, y, rgb)
        }
    }
    return img
}

static BufferedImage maskOverlay(float[] raw, byte[] mask, int width, int height, Color color) {
    double lo = percentile(raw, 0.01d)
    double hi = percentile(raw, 0.995d)
    BufferedImage img = grayImage(raw, width, height, lo, hi)
    double alpha = 0.64d
    for (int y = 0; y < height; y++) {
        int row = y * width
        for (int x = 0; x < width; x++) {
            int idx = row + x
            if (((mask[idx] as int) & 0xff) == 0) continue
            int base = img.getRGB(x, y)
            int br = (base >> 16) & 0xff
            int bg = (base >> 8) & 0xff
            int bb = base & 0xff
            int r = clampByte(br * (1.0d - alpha) + color.getRed() * alpha)
            int g = clampByte(bg * (1.0d - alpha) + color.getGreen() * alpha)
            int b = clampByte(bb * (1.0d - alpha) + color.getBlue() * alpha)
            img.setRGB(x, y, (r << 16) | (g << 8) | b)
        }
    }
    return img
}

static int countMaskPixels(ImagePlus imp, int frame) {
    byte[] mask = pixelsAsByte(imp, frame)
    int count = 0
    for (int i = 0; i < mask.length; i++) {
        if (((mask[i] as int) & 0xff) > 0) count++
    }
    return count
}

static List<Map<String, String>> readTsv(Path path) {
    if (!Files.exists(path)) return []
    List<String> lines = Files.readAllLines(path)
    if (lines.isEmpty()) return []
    String[] header = lines[0].split("\t", -1)
    List<Map<String, String>> rows = []
    for (int i = 1; i < lines.size(); i++) {
        if (lines[i].trim().isEmpty()) continue
        String[] cells = lines[i].split("\t", -1)
        Map<String, String> row = new LinkedHashMap<>()
        for (int c = 0; c < header.length; c++) {
            row[header[c]] = c < cells.length ? cells[c] : ""
        }
        rows.add(row)
    }
    return rows
}

static String maskFileName(Map<String, Object> variant) {
    return "calcium_video_2_${variant.sigma}_${variant.presetTag}_score_ge_${variant.cutoffTag}_mask.tif"
}

static String candidateMaskFileName(Map<String, Object> variant) {
    return "calcium_video_2_${variant.sigma}_${variant.presetTag}_mask.tif"
}

static String highPassFileName(String sigma) {
    return "calcium_video_2_hp_gaussian_${sigma.replace('sigma', 'sigma')}f_float32.tif"
}

static String robustZFileName(String sigma) {
    return "calcium_video_2_${sigma}_robust_positive_z_float32.tif"
}

static void drawPanelLabel(g, String label, int x, int y, int width, int height) {
    g.setColor(new Color(248, 250, 252))
    g.fillRect(x, y, width, height)
    g.setColor(new Color(51, 65, 85))
    g.setFont(new Font("SansSerif", Font.BOLD, 13))
    g.drawString(label, x + 7, y + 16)
}

static void drawImagePanel(g, BufferedImage image, String label, int x, int y, int labelHeight) {
    drawPanelLabel(g, label, x, y, image.getWidth(), labelHeight)
    g.drawImage(image, x, y + labelHeight, null)
    g.setColor(new Color(203, 213, 225))
    g.setStroke(new BasicStroke(1.0f))
    g.drawRect(x, y, image.getWidth(), image.getHeight() + labelHeight)
}

static void writeText(Path path, String value) {
    Files.write(path, value.getBytes("UTF-8"))
}

println("Loading raw video: ${rawPath}")
ImagePlus rawImp = IJ.openImage(rawPath.toString())
if (rawImp == null) throw new IllegalStateException("Could not open raw video: ${rawPath}")
final int width = rawImp.getWidth()
final int height = rawImp.getHeight()
final int frames = rawImp.getStackSize()

Map<String, ImagePlus> hpBySigma = new LinkedHashMap<>()
Map<String, ImagePlus> zBySigma = new LinkedHashMap<>()
Map<String, ImagePlus> candidateMaskByVariant = new LinkedHashMap<>()
Map<String, ImagePlus> temporalMaskByVariant = new LinkedHashMap<>()

variants.collect { it.sigma }.unique().each { sigma ->
    Path hpPath = highPassDir.resolve(highPassFileName(sigma as String))
    Path zPath = candidateDir.resolve(robustZFileName(sigma as String))
    println("Loading ${sigma} high-pass and robust z stacks")
    hpBySigma[sigma as String] = IJ.openImage(hpPath.toString())
    zBySigma[sigma as String] = IJ.openImage(zPath.toString())
    if (hpBySigma[sigma as String] == null) throw new IllegalStateException("Could not open ${hpPath}")
    if (zBySigma[sigma as String] == null) throw new IllegalStateException("Could not open ${zPath}")
}

variants.each { variant ->
    Path candidateMaskPath = candidateDir.resolve(candidateMaskFileName(variant))
    Path temporalMaskPath = temporalDir.resolve(maskFileName(variant))
    println("Loading masks for ${variant.id}")
    candidateMaskByVariant[variant.id as String] = IJ.openImage(candidateMaskPath.toString())
    temporalMaskByVariant[variant.id as String] = IJ.openImage(temporalMaskPath.toString())
    if (candidateMaskByVariant[variant.id as String] == null) {
        throw new IllegalStateException("Could not open ${candidateMaskPath}")
    }
    if (temporalMaskByVariant[variant.id as String] == null) {
        throw new IllegalStateException("Could not open ${temporalMaskPath}")
    }
}

Map<Integer, Integer> activityByFrame = new LinkedHashMap<>()
for (int frame = 1; frame <= frames; frame++) {
    int total = 0
    variants.each { variant ->
        total += countMaskPixels(temporalMaskByVariant[variant.id as String], frame)
    }
    activityByFrame[frame] = total
}

Set<Integer> selectedFrames = new TreeSet<>()
[1, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, frames].each { f ->
    selectedFrames.add(Math.max(1, Math.min(frames, f as int)))
}
activityByFrame.entrySet().sort { a, b -> b.value <=> a.value }.take(maxTopFrames).each { selectedFrames.add(it.key) }
activityByFrame.entrySet().sort { a, b -> a.value <=> b.value }.take(maxQuietFrames).each { selectedFrames.add(it.key) }

List<Map<String, String>> summaryRows = readTsv(temporalDir.resolve("temporal_summary.tsv"))
Map<String, Map<String, String>> summaryByKey = new LinkedHashMap<>()
summaryRows.each { row ->
    summaryByKey["${row.variant}|${row.preset}|${row.cutoff}"] = row
}

Path manifestPath = outputDir.resolve("review_manifest.tsv")
Path reviewSheetPath = outputDir.resolve("manual_review.tsv")
Path reviewTemplatePath = outputDir.resolve("manual_review_template.tsv")
Files.newBufferedWriter(manifestPath).withCloseable { writer ->
    writer.write("frame\tactivity_pixels\tmontage_file\n")
}

Closure writeReviewSheet = { Path path ->
    Files.newBufferedWriter(path).withCloseable { writer ->
    writer.write("frame\tvariant_id\tvariant_label\tnoise_rating_0_clean_3_bad\tmisses_obvious_events_yes_no\tovermerged_clusters_0_none_3_bad\tkeep_reject_unsure\tnotes\n")
    selectedFrames.each { frame ->
        variants.each { variant ->
            writer.write("${frame}\t${variant.id}\t${variant.label}\t\t\t\t\t\n")
        }
    }
    }
}

if (!Files.exists(reviewSheetPath)) {
    writeReviewSheet(reviewSheetPath)
}
writeReviewSheet(reviewTemplatePath)

List<Map<String, Object>> montageRows = []
selectedFrames.each { frame ->
    println("Rendering montage for frame ${frame}")
    float[] raw = pixelsAsFloat(rawImp, frame)
    double rawLo = percentile(raw, 0.01d)
    double rawHi = percentile(raw, 0.995d)

    int columns = 5
    int rows = variants.size()
    int outW = leftLabelWidth + columns * width + (columns + 1) * gap
    int outH = gap + rows * (height + labelHeight + gap)
    BufferedImage montage = new BufferedImage(outW, outH, BufferedImage.TYPE_INT_RGB)
    def g = montage.createGraphics()
    g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON)
    g.setColor(new Color(241, 245, 249))
    g.fillRect(0, 0, outW, outH)

    variants.eachWithIndex { variant, rowIndex ->
        String sigma = variant.sigma as String
        int y = gap + rowIndex * (height + labelHeight + gap)
        int x = leftLabelWidth + gap

        g.setColor(new Color(15, 23, 42))
        g.setFont(new Font("SansSerif", Font.BOLD, 13))
        List<String> rowLabelParts = (variant.label as String).split(" ") as List<String>
        int labelY = y + 18
        rowLabelParts.each { part ->
            g.drawString(part, 8, labelY)
            labelY += 16
        }

        float[] hp = pixelsAsFloat(hpBySigma[sigma], frame)
        float[] z = pixelsAsFloat(zBySigma[sigma], frame)
        byte[] candidateMask = pixelsAsByte(candidateMaskByVariant[variant.id as String], frame)
        byte[] temporalMask = pixelsAsByte(temporalMaskByVariant[variant.id as String], frame)

        double hpAbs = Math.max(Math.abs(percentile(hp, 0.005d)), Math.abs(percentile(hp, 0.995d)))
        hpAbs = Math.max(1.0d, hpAbs)
        BufferedImage rawImg = grayImage(raw, width, height, rawLo, rawHi)
        BufferedImage hpImg = grayImage(hp, width, height, -hpAbs, hpAbs)
        BufferedImage zImg = grayImage(z, width, height, 0.0d, 5.0d)
        BufferedImage candidateOverlay = maskOverlay(raw, candidateMask, width, height, new Color(249, 115, 22))
        BufferedImage temporalOverlay = maskOverlay(raw, temporalMask, width, height, new Color(20, 184, 166))

        drawImagePanel(g, rawImg, "Raw", x, y, labelHeight)
        x += width + gap
        drawImagePanel(g, hpImg, "High-pass", x, y, labelHeight)
        x += width + gap
        drawImagePanel(g, zImg, "Robust z", x, y, labelHeight)
        x += width + gap
        drawImagePanel(g, candidateOverlay, "Candidate mask", x, y, labelHeight)
        x += width + gap
        drawImagePanel(g, temporalOverlay, "Temporal mask", x, y, labelHeight)
    }
    g.dispose()

    String fileName = String.format("frame_%03d_review_montage.png", frame)
    Path montagePath = montageDir.resolve(fileName)
    ImageIO.write(montage, "png", montagePath.toFile())
    Files.newBufferedWriter(manifestPath, java.nio.file.StandardOpenOption.APPEND).withCloseable { writer ->
        writer.write("${frame}\t${activityByFrame[frame]}\tmontages/${fileName}\n")
    }
    montageRows.add([frame: frame, activity: activityByFrame[frame], file: "montages/${fileName}"])
}

StringBuilder html = new StringBuilder()
html << "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
html << "<title>calcium_video_2 review dashboard</title>\n"
html << "<style>\n"
html << "body{font-family:Arial,Helvetica,sans-serif;margin:24px;background:#f8fafc;color:#0f172a;line-height:1.42}h1{font-size:24px;margin:0 0 6px}h2{font-size:18px;margin:28px 0 10px}.meta{color:#475569;margin-bottom:18px}table{border-collapse:collapse;width:100%;background:white;margin:10px 0 20px}th,td{border:1px solid #cbd5e1;padding:7px 8px;text-align:left;font-size:13px;vertical-align:top}th{background:#e2e8f0}.tag{display:inline-block;padding:2px 7px;border-radius:5px;background:#dbeafe;color:#1e3a8a;font-size:12px}.note{color:#475569}.montage{margin:20px 0 34px}.montage img{max-width:100%;height:auto;border:1px solid #cbd5e1;background:white}.bar{display:inline-block;height:10px;background:#14b8a6;vertical-align:middle;margin-right:6px}.small{font-size:12px;color:#64748b}.files code{background:#e2e8f0;padding:2px 4px;border-radius:4px}\n"
html << "</style>\n</head>\n<body>\n"
html << "<h1>calcium_video_2 Review Dashboard</h1>\n"
html << "<div class=\"meta\">Generated from current high-pass, candidate, and temporal scoring outputs. Review sheet: <code>manual_review.tsv</code></div>\n"

html << "<h2>Focused Variants</h2>\n<table><thead><tr><th>Variant</th><th>Mean components/frame</th><th>Mean pixels/frame</th><th>Why included</th></tr></thead><tbody>\n"
variants.each { variant ->
    Map<String, String> row = summaryByKey["${variant.sigma}|${variant.preset}|${variant.cutoff}"]
    html << "<tr><td><span class=\"tag\">${htmlEscape(variant.id)}</span><br>${htmlEscape(variant.label)}</td>"
    html << "<td>${htmlEscape(row?.mean_components_per_frame ?: '')}</td>"
    html << "<td>${htmlEscape(row?.mean_pixels_per_frame ?: '')}</td>"
    html << "<td class=\"note\">${htmlEscape(variant.note)}</td></tr>\n"
}
html << "</tbody></table>\n"

html << "<h2>All Temporal Summary Rows</h2>\n<table><thead><tr><th>Variant</th><th>Preset</th><th>Cutoff</th><th>Components/frame</th><th>Pixels/frame</th><th>Relative pixels</th></tr></thead><tbody>\n"
double maxMeanPixels = summaryRows.collect { (it.mean_pixels_per_frame ?: "0") as double }.max() ?: 1.0d
summaryRows.each { row ->
    double pixels = (row.mean_pixels_per_frame ?: "0") as double
    int barW = Math.max(1, Math.round(180.0d * pixels / Math.max(1.0d, maxMeanPixels)) as int)
    boolean focused = variants.any { it.sigma == row.variant && it.preset == row.preset && it.cutoff == row.cutoff }
    html << "<tr><td>${focused ? '<span class=\"tag\">focused</span> ' : ''}${htmlEscape(row.variant)}</td>"
    html << "<td>${htmlEscape(row.preset)}</td><td>${htmlEscape(row.cutoff)}</td>"
    html << "<td>${htmlEscape(row.mean_components_per_frame)}</td><td>${htmlEscape(row.mean_pixels_per_frame)}</td>"
    html << "<td><span class=\"bar\" style=\"width:${barW}px\"></span>${String.format('%.1f', pixels)}</td></tr>\n"
}
html << "</tbody></table>\n"

html << "<h2>Review Frames</h2>\n<table><thead><tr><th>Frame</th><th>Combined focused mask pixels</th><th>Montage</th></tr></thead><tbody>\n"
montageRows.each { row ->
    html << "<tr><td>${row.frame}</td><td>${row.activity}</td><td><a href=\"${htmlEscape(row.file)}\">${htmlEscape(row.file)}</a></td></tr>\n"
}
html << "</tbody></table>\n"

html << "<h2>Montages</h2>\n"
montageRows.each { row ->
    html << "<div class=\"montage\"><h3>Frame ${row.frame} <span class=\"small\">combined focused pixels: ${row.activity}</span></h3>"
    html << "<a href=\"${htmlEscape(row.file)}\"><img src=\"${htmlEscape(row.file)}\" alt=\"Frame ${row.frame} montage\"></a></div>\n"
}

html << "<h2>Files</h2><p class=\"files\">Use <code>manual_review.tsv</code> for notes; reruns preserve it once it exists. Use <code>manual_review_template.tsv</code> for a fresh blank sheet and <code>review_manifest.tsv</code> to trace generated montages.</p>\n"
html << "</body>\n</html>\n"

writeText(outputDir.resolve("review_dashboard.html"), html.toString())
writeText(outputDir.resolve("parameters.txt"), [
    "raw_video=${rawPath}",
    "high_pass_dir=${highPassDir}",
    "candidate_dir=${candidateDir}",
    "temporal_dir=${temporalDir}",
    "output_dir=${outputDir}",
    "focused_variants=${variants.collect { it.id }.join(',')}",
    "selected_frames=${selectedFrames.join(',')}",
    "manual_review_preserved_if_present=true",
    "activity_metric=sum of focused temporal mask pixels per frame",
    "montage_columns=raw,high-pass,robust_z,candidate_mask_overlay,temporal_mask_overlay"
].join("\n") + "\n")

rawImp.close()
hpBySigma.values().each { it.close() }
zBySigma.values().each { it.close() }
candidateMaskByVariant.values().each { it.close() }
temporalMaskByVariant.values().each { it.close() }

println("Done. Wrote review dashboard to ${outputDir.resolve('review_dashboard.html')}")
System.exit(0)
