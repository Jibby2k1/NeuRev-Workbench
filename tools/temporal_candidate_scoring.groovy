// Temporal scoring for candidate event components.
//
// Reads the permissive spatial candidates and assigns each component a soft
// temporal score using raw fluorescence and high-pass residual traces. The
// score is not a hard duration gate: one-frame events can survive if they have
// enough raw/high-pass support.

import ij.IJ
import ij.ImagePlus
import ij.ImageStack
import ij.io.FileSaver
import ij.process.ByteProcessor
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.util.Locale

static String setting(String key, String fallback) {
    String envKey = "NEUROBENCH_" + key.toUpperCase(Locale.ROOT).replaceAll("[^A-Z0-9]+", "_")
    String envValue = System.getenv(envKey)
    if (envValue != null && !envValue.isBlank()) return envValue
    String propValue = System.getProperty("neurobench." + key)
    if (propValue != null && !propValue.isBlank()) return propValue
    return fallback
}

static Path resolvePath(Path projectRoot, String value) {
    Path path = Paths.get(value)
    return path.isAbsolute() ? path : projectRoot.resolve(path)
}

final Path projectRoot = Paths.get(setting("project_root", System.getProperty("user.dir")))
final String datasetId = setting("dataset_id", "calcium_video_2")
final Path rawPath = resolvePath(projectRoot, setting("raw_video", "Inputs/050126/050126/calcium video 2.tif"))
final Path outputRoot = resolvePath(projectRoot, setting("output_root", "Outputs"))
final Path highPassDir = outputRoot.resolve("HighPass").resolve(datasetId)
final Path candidateDir = outputRoot.resolve("CandidateEventPipeline").resolve(datasetId)
final Path outputDir = outputRoot.resolve("TemporalCandidateScoring").resolve(datasetId)

final int preFrames = Integer.parseInt(setting("trace_pre_frames", "4"))
final int rawPostFrames = Integer.parseInt(setting("trace_raw_post_frames", "8"))
final int hpSideFrames = Integer.parseInt(setting("trace_hp_side_frames", "4"))
final int roiExpansion = Integer.parseInt(setting("trace_roi_expansion_px", "1"))
final double[] scoreCutoffs = [0.25d, 0.50d, 0.75d] as double[]
final String requestedSigmaLabel = setting("sigma_label", "")
final String requestedPresetName = setting("component_preset_name", "")

final List<Map<String, String>> variants = [
    [label: "sigma04", hpFile: "${datasetId}_hp_gaussian_sigma04f_float32.tif"],
    [label: "sigma06", hpFile: "${datasetId}_hp_gaussian_sigma06f_float32.tif"],
    [label: "sigma08", hpFile: "${datasetId}_hp_gaussian_sigma08f_float32.tif"],
]
final List<Map<String, String>> activeVariants = requestedSigmaLabel.isBlank()
    ? variants
    : [[label: "sigma${requestedSigmaLabel}", hpFile: "${datasetId}_hp_gaussian_sigma${requestedSigmaLabel}f_float32.tif"]]

final List<Map<String, String>> presets = [
    [name: "permissive", tag: "permissive_seed014_grow007_min3"],
    [name: "balanced", tag: "balanced_seed017_grow009_min3"],
    [name: "strict", tag: "strict_seed020_grow011_min4"],
]
final List<Map<String, String>> activePresets = requestedPresetName.isBlank()
    ? presets
    : [[
        name: requestedPresetName,
        tag: "${requestedPresetName}_seed${thresholdTag(Double.parseDouble(setting("component_seed_z", "2.0")))}_grow${thresholdTag(Double.parseDouble(setting("component_grow_z", "1.1")))}_min${Integer.parseInt(setting("component_min_area_px", "4"))}"
    ]]

Files.createDirectories(outputDir)

static String scoreTag(double score) {
    return String.format("%03d", Math.round(score * 100.0d) as int)
}

static String thresholdTag(double value) {
    return String.format("%03d", Math.round(value * 10.0d) as int)
}

static double clamp01(double x) {
    if (x < 0.0d) return 0.0d
    if (x > 1.0d) return 1.0d
    return x
}

static double median(List<Double> values, double fallback) {
    if (values == null || values.isEmpty()) return fallback
    List<Double> copy = new ArrayList<>(values)
    Collections.sort(copy)
    int n = copy.size()
    if ((n & 1) == 1) return copy[n.intdiv(2)]
    return 0.5d * (copy[n.intdiv(2) - 1] + copy[n.intdiv(2)])
}

static double madSigma(List<Double> values, double center, double fallback) {
    if (values == null || values.isEmpty()) return fallback
    List<Double> dev = new ArrayList<>(values.size())
    values.each { v -> dev.add(Math.abs(v - center)) }
    return Math.max(1.0d, 1.4826d * median(dev, fallback))
}

static double[] integralForProcessor(ip, int width, int height) {
    float[] pix = (float[]) ip.convertToFloatProcessor().getPixels()
    int stride = width + 1
    double[] integral = new double[(height + 1) * stride]
    for (int y = 1; y <= height; y++) {
        double rowSum = 0.0d
        int srcBase = (y - 1) * width
        int dstBase = y * stride
        int prevBase = (y - 1) * stride
        for (int x = 1; x <= width; x++) {
            rowSum += pix[srcBase + x - 1] as double
            integral[dstBase + x] = integral[prevBase + x] + rowSum
        }
    }
    return integral
}

static double[][] integralStack(ImagePlus imp) {
    int width = imp.getWidth()
    int height = imp.getHeight()
    int frames = imp.getStackSize()
    double[][] integrals = new double[frames][]
    for (int s = 1; s <= frames; s++) {
        integrals[s - 1] = integralForProcessor(imp.getStack().getProcessor(s), width, height)
    }
    return integrals
}

static double rectMean(double[] integral, int width, int x0, int y0, int x1, int y1) {
    int stride = width + 1
    int xa = x0
    int xb = x1 + 1
    int ya = y0
    int yb = y1 + 1
    int area = Math.max(1, (x1 - x0 + 1) * (y1 - y0 + 1))
    double sum = integral[yb * stride + xb] - integral[ya * stride + xb] -
        integral[yb * stride + xa] + integral[ya * stride + xa]
    return sum / area
}

static int adjacentLabelPixels(ImagePlus labelsImp, int frame, int width, int height, int x0, int y0, int x1, int y1) {
    if (frame < 1 || frame > labelsImp.getStackSize()) return 0
    Object raw = labelsImp.getStack().getProcessor(frame).getPixels()
    int count = 0
    for (int y = y0; y <= y1; y++) {
        int row = y * width
        for (int x = x0; x <= x1; x++) {
            int idx = row + x
            int value
            if (raw instanceof short[]) {
                value = ((short[]) raw)[idx] & 0xffff
            } else if (raw instanceof byte[]) {
                value = ((byte[]) raw)[idx] & 0xff
            } else {
                value = (int) ((float[]) raw)[idx]
            }
            if (value > 0) count++
        }
    }
    return count
}

static void paintComponent(ImagePlus labelsImp, int frame, int componentId, byte[] out, int width, int height, int x0, int y0, int x1, int y1) {
    Object raw = labelsImp.getStack().getProcessor(frame).getPixels()
    for (int y = y0; y <= y1; y++) {
        int row = y * width
        for (int x = x0; x <= x1; x++) {
            int idx = row + x
            int value
            if (raw instanceof short[]) {
                value = ((short[]) raw)[idx] & 0xffff
            } else if (raw instanceof byte[]) {
                value = ((byte[]) raw)[idx] & 0xff
            } else {
                value = (int) ((float[]) raw)[idx]
            }
            if (value == componentId) {
                out[idx] = (byte) 255
            }
        }
    }
}

println("Loading raw video and precomputing raw integral images: ${rawPath}")
ImagePlus rawImp = IJ.openImage(rawPath.toString())
if (rawImp == null) throw new IllegalStateException("Could not open raw video: ${rawPath}")
final int width = rawImp.getWidth()
final int height = rawImp.getHeight()
final int frames = rawImp.getStackSize()
double[][] rawIntegrals = integralStack(rawImp)
rawImp.close()

StringBuilder params = new StringBuilder()
params << "dataset_id=${datasetId}\n"
params << "raw_video=${rawPath}\n"
params << "high_pass_dir=${highPassDir}\n"
params << "candidate_dir=${candidateDir}\n"
params << "output_dir=${outputDir}\n"
params << "roi=component_bbox_expanded_by_${roiExpansion}_pixel\n"
params << "raw_trace_window=t-${preFrames}..t+${rawPostFrames}\n"
params << "high_pass_trace_window=t-${hpSideFrames}..t+${hpSideFrames}\n"
params << "score_cutoffs=${scoreCutoffs.join(',')}\n"
params << "score=soft combination of candidate z, raw rise, high-pass peak, nearby-frame label support, minus collapse penalty\n"
params << "hard_duration_gate=false\n"
Files.writeString(outputDir.resolve("parameters.txt"), params.toString())

Path eventInput = candidateDir.resolve("candidate_events.tsv")
Path temporalEvents = outputDir.resolve("temporal_candidates.tsv")
Path summaryPath = outputDir.resolve("temporal_summary.tsv")

Files.newBufferedWriter(temporalEvents).withCloseable { eventWriter ->
    Files.newBufferedWriter(summaryPath).withCloseable { summaryWriter ->
        eventWriter.write("variant\tpreset\tframe\tcomponent_id\tarea\tcentroid_x\tcentroid_y\tpeak_z\tmean_z\tmin_x\tmin_y\tmax_x\tmax_y\tbbox_width\tbbox_height\tfill_fraction\taspect_ratio\traw_baseline\traw_at_t\traw_post_max\traw_rise\traw_z\thp_at_t\thp_peak_abs\thp_z\tcoincidence_score\tcollapse_penalty\ttemporal_score\n")
        summaryWriter.write("variant\tpreset\tcutoff\tkept_components\tkept_pixels\tmean_components_per_frame\tmean_pixels_per_frame\n")

        activeVariants.each { variant ->
            String variantLabel = variant.label
            Path hpPath = highPassDir.resolve(variant.hpFile)
            println("Loading high-pass stack and integrals for ${variantLabel}: ${hpPath}")
            ImagePlus hpImp = IJ.openImage(hpPath.toString())
            if (hpImp == null) throw new IllegalStateException("Could not open high-pass stack: ${hpPath}")
            double[][] hpIntegrals = integralStack(hpImp)
            hpImp.close()

            activePresets.each { preset ->
                String presetName = preset.name
                String presetTag = preset.tag
                Path labelsPath = candidateDir.resolve("${datasetId}_${variantLabel}_${presetTag}_labels.tif")
                println("Scoring ${variantLabel}/${presetName}: ${labelsPath}")
                ImagePlus labelsImp = IJ.openImage(labelsPath.toString())
                if (labelsImp == null) throw new IllegalStateException("Could not open label stack: ${labelsPath}")

                Map<Double, ImageStack> cutoffStacks = new LinkedHashMap<>()
                Map<Double, byte[]> currentMasks = new LinkedHashMap<>()
                Map<Double, Integer> keptComponents = new LinkedHashMap<>()
                Map<Double, Integer> keptPixels = new LinkedHashMap<>()
                scoreCutoffs.each { cutoff ->
                    cutoffStacks[cutoff] = new ImageStack(width, height)
                    currentMasks[cutoff] = new byte[width * height]
                    keptComponents[cutoff] = 0
                    keptPixels[cutoff] = 0
                }

                int currentFrame = 1

                Closure flushUntil = { int targetFrame ->
                    while (currentFrame < targetFrame) {
                        scoreCutoffs.each { cutoff ->
                            cutoffStacks[cutoff].addSlice("frame_${currentFrame}", new ByteProcessor(width, height, currentMasks[cutoff]))
                            currentMasks[cutoff] = new byte[width * height]
                        }
                        currentFrame++
                    }
                }

                eventInput.toFile().withReader { reader ->
                    String header = reader.readLine()
                    String line
                    while ((line = reader.readLine()) != null) {
                        String[] c = line.split("\t", -1)
                        if (c[0] != variantLabel || c[1] != presetName) continue

                        int frame = c[2] as int
                        int componentId = c[3] as int
                        int area = c[4] as int
                        double centroidX = c[5] as double
                        double centroidY = c[6] as double
                        double peakZ = c[7] as double
                        double meanZ = c[8] as double
                        int minX0 = c[9] as int
                        int minY0 = c[10] as int
                        int maxX0 = c[11] as int
                        int maxY0 = c[12] as int
                        int bboxW = c[13] as int
                        int bboxH = c[14] as int
                        double fillFraction = c[15] as double
                        double aspectRatio = c[16] as double

                        flushUntil(frame)

                        int x0 = Math.max(0, minX0 - roiExpansion)
                        int y0 = Math.max(0, minY0 - roiExpansion)
                        int x1 = Math.min(width - 1, maxX0 + roiExpansion)
                        int y1 = Math.min(height - 1, maxY0 + roiExpansion)

                        List<Double> rawPre = []
                        List<Double> rawPost = []
                        List<Double> hpWindow = []
                        for (int f = Math.max(1, frame - preFrames); f <= frame - 1; f++) {
                            rawPre.add(rectMean(rawIntegrals[f - 1], width, x0, y0, x1, y1))
                        }
                        for (int f = frame + 1; f <= Math.min(frames, frame + rawPostFrames); f++) {
                            rawPost.add(rectMean(rawIntegrals[f - 1], width, x0, y0, x1, y1))
                        }
                        for (int f = Math.max(1, frame - hpSideFrames); f <= Math.min(frames, frame + hpSideFrames); f++) {
                            hpWindow.add(rectMean(hpIntegrals[f - 1], width, x0, y0, x1, y1))
                        }

                        double rawAtT = rectMean(rawIntegrals[frame - 1], width, x0, y0, x1, y1)
                        double hpAtT = rectMean(hpIntegrals[frame - 1], width, x0, y0, x1, y1)
                        double rawBaseline = median(rawPre, rawAtT)
                        double rawSigma = madSigma(rawPre, rawBaseline, 1.0d)
                        double rawRise = rawAtT - rawBaseline
                        double rawZ = rawRise / rawSigma
                        double rawPostMax = rawPost.isEmpty() ? rawAtT : rawPost.max()
                        double hpAbsMedian = median(hpWindow.collect { v -> Math.abs(v) }, 1.0d)
                        double hpPeakAbs = hpWindow.collect { v -> Math.abs(v) }.max()
                        double hpZ = hpAtT / Math.max(1.0d, hpAbsMedian)

                        int adjacentPixels = 0
                        adjacentPixels += adjacentLabelPixels(labelsImp, frame - 1, width, height, x0, y0, x1, y1)
                        adjacentPixels += adjacentLabelPixels(labelsImp, frame + 1, width, height, x0, y0, x1, y1)
                        adjacentPixels += adjacentLabelPixels(labelsImp, frame + 2, width, height, x0, y0, x1, y1)
                        double coincidenceScore = clamp01(adjacentPixels / Math.max(2.0d, area * 0.5d))

                        double candidateScore = clamp01((peakZ - 1.0d) / 2.5d)
                        double rawRiseScore = clamp01(rawZ / 3.0d)
                        double hpPeakScore = clamp01(hpZ / 3.0d)
                        double postSupport = rawRise > 0.0d ? clamp01((rawPostMax - rawBaseline) / Math.max(rawRise, 1.0d)) : 0.0d
                        double collapsePenalty = (coincidenceScore < 0.05d && postSupport < 0.25d && hpAtT > 0.0d && hpPeakAbs <= Math.max(1.0d, Math.abs(hpAtT) * 1.2d)) ? 0.35d : 0.0d
                        double temporalScore = clamp01(0.25d * candidateScore + 0.35d * rawRiseScore + 0.25d * hpPeakScore + 0.25d * coincidenceScore - collapsePenalty)

                        eventWriter.write([
                            variantLabel, presetName, frame, componentId, area, centroidX, centroidY, peakZ, meanZ,
                            minX0, minY0, maxX0, maxY0, bboxW, bboxH, fillFraction, aspectRatio,
                            rawBaseline, rawAtT, rawPostMax, rawRise, rawZ, hpAtT, hpPeakAbs, hpZ,
                            coincidenceScore, collapsePenalty, temporalScore
                        ].join("\t"))
                        eventWriter.write("\n")

                        scoreCutoffs.each { cutoff ->
                            if (temporalScore >= cutoff) {
                                int before = keptPixels[cutoff]
                                paintComponent(labelsImp, frame, componentId, currentMasks[cutoff], width, height, minX0, minY0, maxX0, maxY0)
                                keptComponents[cutoff] = keptComponents[cutoff] + 1
                                keptPixels[cutoff] = before + area
                            }
                        }
                    }
                }

                flushUntil(frames + 1)

                scoreCutoffs.each { cutoff ->
                    String tag = scoreTag(cutoff)
                    ImagePlus out = new ImagePlus("${datasetId}_${variantLabel}_${presetTag}_score_ge_${tag}_mask", cutoffStacks[cutoff])
                    Path maskPath = outputDir.resolve("${datasetId}_${variantLabel}_${presetTag}_score_ge_${tag}_mask.tif")
                    new FileSaver(out).saveAsTiffStack(maskPath.toString())
                    out.close()
                    summaryWriter.write([
                        variantLabel, presetName, cutoff, keptComponents[cutoff], keptPixels[cutoff],
                        keptComponents[cutoff] / frames, keptPixels[cutoff] / frames
                    ].join("\t"))
                    summaryWriter.write("\n")
                }

                labelsImp.close()
                println("  Done ${variantLabel}/${presetName}")
            }
        }
    }
}

println("Done. Wrote outputs to ${outputDir}")
System.exit(0)
