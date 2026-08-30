// Candidate-first event pipeline for high-pass calcium residuals.
//
// This pipeline is deliberately permissive at candidate generation, then rejects
// random impulse noise with per-frame spatial component filters. It does not
// smooth over time and does not require multi-frame event duration.

import ij.IJ
import ij.ImagePlus
import ij.ImageStack
import ij.io.FileSaver
import ij.plugin.filter.RankFilters
import ij.process.ByteProcessor
import ij.process.FloatProcessor
import ij.process.ShortProcessor
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.util.Arrays
import java.util.ArrayDeque
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
final Path outputRoot = resolvePath(projectRoot, setting("output_root", "Outputs"))
final Path highPassDir = outputRoot.resolve("HighPass").resolve(datasetId)
final Path outputDir = outputRoot.resolve("CandidateEventPipeline").resolve(datasetId)

final double localRadius = Double.parseDouble(setting("local_radius_px", "11.0"))
final double madScale = 1.4826d
final double epsilon = Double.parseDouble(setting("epsilon", "1.0"))
final int maxArea = Integer.parseInt(setting("component_max_area_px", "300"))
final double minFillFraction = Double.parseDouble(setting("component_min_fill_fraction", "0.10"))
final double maxAspectRatio = Double.parseDouble(setting("component_max_aspect_ratio", "6.0"))
final String requestedSigmaLabel = setting("sigma_label", "")
final String requestedPresetName = setting("component_preset_name", "")

final List<Map<String, String>> variants = [
    [label: "sigma04", file: "${datasetId}_hp_gaussian_sigma04f_float32.tif"],
    [label: "sigma06", file: "${datasetId}_hp_gaussian_sigma06f_float32.tif"],
    [label: "sigma08", file: "${datasetId}_hp_gaussian_sigma08f_float32.tif"],
]
final List<Map<String, String>> activeVariants = requestedSigmaLabel.isBlank()
    ? variants
    : [[label: "sigma${requestedSigmaLabel}", file: "${datasetId}_hp_gaussian_sigma${requestedSigmaLabel}f_float32.tif"]]

final List<Map<String, Object>> presets = [
    [name: "permissive", seed: 1.4d, grow: 0.7d, minArea: 3],
    [name: "balanced", seed: 1.7d, grow: 0.9d, minArea: 3],
    [name: "strict", seed: 2.0d, grow: 1.1d, minArea: 4],
]
final List<Map<String, Object>> activePresets = requestedPresetName.isBlank()
    ? presets
    : [[
        name: requestedPresetName,
        seed: Double.parseDouble(setting("component_seed_z", "2.0")),
        grow: Double.parseDouble(setting("component_grow_z", "1.1")),
        minArea: Integer.parseInt(setting("component_min_area_px", "4"))
    ]]

Files.createDirectories(outputDir)

static double median(float[] values) {
    float[] copy = values.clone()
    Arrays.sort(copy)
    int n = copy.length
    if ((n & 1) == 1) {
        return copy[n.intdiv(2)] as double
    }
    return 0.5d * ((copy[n.intdiv(2) - 1] as double) + (copy[n.intdiv(2)] as double))
}

static float[] duplicatePixels(float[] pixels) {
    float[] copy = new float[pixels.length]
    System.arraycopy(pixels, 0, copy, 0, pixels.length)
    return copy
}

static FloatProcessor medianFiltered(float[] pixels, int width, int height, double radius, RankFilters rankFilters) {
    FloatProcessor fp = new FloatProcessor(width, height, duplicatePixels(pixels))
    rankFilters.rank(fp, radius, RankFilters.MEDIAN)
    return fp
}

static String thresholdTag(double value) {
    return String.format("%03d", Math.round(value * 10.0d) as int)
}

static String presetTag(Map<String, Object> preset) {
    return "${preset.name}_seed${thresholdTag(preset.seed as double)}_grow${thresholdTag(preset.grow as double)}_min${preset.minArea}"
}

static Map<String, Object> componentFilter(
    float[] z,
    int width,
    int height,
    double seedThreshold,
    double growThreshold,
    int minArea,
    int maxArea,
    double minFillFraction,
    double maxAspectRatio,
    String variant,
    String preset,
    int frame
) {
    int n = z.length
    boolean[] visited = new boolean[n]
    byte[] mask = new byte[n]
    short[] labels = new short[n]
    List<String> rows = []
    int componentId = 0
    int accepted = 0
    int rejectedNoSeed = 0
    int rejectedArea = 0
    int rejectedShape = 0

    int[] queue = new int[n]
    int[] members = new int[n]
    int[] dx = [-1, 0, 1, -1, 1, -1, 0, 1] as int[]
    int[] dy = [-1, -1, -1, 0, 0, 1, 1, 1] as int[]

    for (int start = 0; start < n; start++) {
        if (visited[start] || (z[start] as double) < growThreshold) {
            continue
        }

        int head = 0
        int tail = 0
        int count = 0
        boolean hasSeed = false
        double peakZ = Double.NEGATIVE_INFINITY
        double sumZ = 0.0d
        double sumX = 0.0d
        double sumY = 0.0d
        int minX = width
        int maxX = -1
        int minY = height
        int maxY = -1

        visited[start] = true
        queue[tail++] = start

        while (head < tail) {
            int idx = queue[head++]
            members[count++] = idx
            double value = z[idx] as double
            int y = idx.intdiv(width)
            int x = idx - y * width

            if (value >= seedThreshold) {
                hasSeed = true
            }
            if (value > peakZ) {
                peakZ = value
            }
            sumZ += value
            sumX += x
            sumY += y
            if (x < minX) minX = x
            if (x > maxX) maxX = x
            if (y < minY) minY = y
            if (y > maxY) maxY = y

            for (int k = 0; k < 8; k++) {
                int nx = x + dx[k]
                int ny = y + dy[k]
                if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                    continue
                }
                int nidx = ny * width + nx
                if (!visited[nidx] && (z[nidx] as double) >= growThreshold) {
                    visited[nidx] = true
                    queue[tail++] = nidx
                }
            }
        }

        if (!hasSeed) {
            rejectedNoSeed++
            continue
        }
        if (count < minArea || count > maxArea) {
            rejectedArea++
            continue
        }

        int bboxW = maxX - minX + 1
        int bboxH = maxY - minY + 1
        double fill = count / (bboxW * bboxH) as double
        double aspect = Math.max(bboxW, bboxH) / Math.max(1.0d, Math.min(bboxW, bboxH) as double)
        if (fill < minFillFraction || aspect > maxAspectRatio) {
            rejectedShape++
            continue
        }

        accepted++
        componentId = Math.min(accepted, 65535)
        for (int i = 0; i < count; i++) {
            int idx = members[i]
            mask[idx] = (byte) 255
            labels[idx] = (short) componentId
        }

        rows << [
            variant,
            preset,
            frame,
            componentId,
            count,
            sumX / count,
            sumY / count,
            peakZ,
            sumZ / count,
            minX,
            minY,
            maxX,
            maxY,
            bboxW,
            bboxH,
            fill,
            aspect
        ].join("\t")
    }

    return [
        mask: mask,
        labels: labels,
        rows: rows,
        accepted: accepted,
        rejectedNoSeed: rejectedNoSeed,
        rejectedArea: rejectedArea,
        rejectedShape: rejectedShape
    ]
}

StringBuilder params = new StringBuilder()
params << "dataset_id=${datasetId}\n"
params << "input_dir=${highPassDir}\n"
params << "output_dir=${outputDir}\n"
params << "local_robust_z=(frame_median_corrected - local_median_radius_${localRadius}) / (1.4826 * local_median_abs_deviation + ${epsilon})\n"
params << "positive_only=true\n"
params << "temporal_smoothing=none\n"
params << "component_connectivity=8\n"
params << "component_max_area=${maxArea}\n"
params << "component_min_fill_fraction=${minFillFraction}\n"
params << "component_max_aspect_ratio=${maxAspectRatio}\n"
activePresets.each { p ->
    params << "preset=${p.name},seed_z=${p.seed},grow_z=${p.grow},min_area=${p.minArea}\n"
}

StringBuilder events = new StringBuilder()
events << "variant\tpreset\tframe\tcomponent_id\tarea\tcentroid_x\tcentroid_y\tpeak_z\tmean_z\tmin_x\tmin_y\tmax_x\tmax_y\tbbox_width\tbbox_height\tfill_fraction\taspect_ratio\n"

StringBuilder frameStats = new StringBuilder()
frameStats << "variant\tpreset\tframe\taccepted_components\taccepted_pixels\trejected_no_seed\trejected_area\trejected_shape\n"

RankFilters rankFilters = new RankFilters()

activeVariants.each { variant ->
    String variantLabel = variant.label
    Path inPath = highPassDir.resolve(variant.file)
    if (!Files.exists(inPath)) {
        throw new FileNotFoundException("Missing high-pass input: ${inPath}")
    }

    println("Processing ${variantLabel}: ${inPath}")
    ImagePlus imp = IJ.openImage(inPath.toString())
    if (imp == null) {
        throw new IllegalStateException("Could not open ${inPath}")
    }

    int width = imp.getWidth()
    int height = imp.getHeight()
    int slices = imp.getStackSize()
    ImageStack zStack = new ImageStack(width, height)
    Map<String, ImageStack> maskStacks = new LinkedHashMap<>()
    Map<String, ImageStack> labelStacks = new LinkedHashMap<>()
    activePresets.each { p ->
        String tag = presetTag(p)
        maskStacks[tag] = new ImageStack(width, height)
        labelStacks[tag] = new ImageStack(width, height)
    }

    for (int s = 1; s <= slices; s++) {
        float[] raw = (float[]) imp.getStack().getProcessor(s).convertToFloatProcessor().getPixels()
        double frameMedian = median(raw)
        float[] corrected = new float[raw.length]
        for (int i = 0; i < raw.length; i++) {
            corrected[i] = ((raw[i] as double) - frameMedian) as float
        }

        FloatProcessor localMedianFp = medianFiltered(corrected, width, height, localRadius, rankFilters)
        float[] localMedian = (float[]) localMedianFp.getPixels()
        float[] absDev = new float[corrected.length]
        for (int i = 0; i < corrected.length; i++) {
            absDev[i] = Math.abs((corrected[i] as double) - (localMedian[i] as double)) as float
        }
        FloatProcessor localMadFp = medianFiltered(absDev, width, height, localRadius, rankFilters)
        float[] localMad = (float[]) localMadFp.getPixels()

        float[] z = new float[corrected.length]
        for (int i = 0; i < corrected.length; i++) {
            double signal = (corrected[i] as double) - (localMedian[i] as double)
            if (signal <= 0.0d) {
                z[i] = 0.0f
            } else {
                double sigma = madScale * (localMad[i] as double)
                z[i] = (signal / (sigma + epsilon)) as float
            }
        }
        zStack.addSlice("frame_${s}", new FloatProcessor(width, height, z))

        activePresets.each { preset ->
            String tag = presetTag(preset)
            Map<String, Object> result = componentFilter(
                z,
                width,
                height,
                preset.seed as double,
                preset.grow as double,
                preset.minArea as int,
                maxArea,
                minFillFraction,
                maxAspectRatio,
                variantLabel,
                preset.name as String,
                s
            )

            byte[] mask = (byte[]) result.mask
            short[] labels = (short[]) result.labels
            maskStacks[tag].addSlice("frame_${s}", new ByteProcessor(width, height, mask))
            labelStacks[tag].addSlice("frame_${s}", new ShortProcessor(width, height, labels, null))

            int acceptedPixels = 0
            for (byte b : mask) {
                if ((b & 0xff) > 0) acceptedPixels++
            }
            ((List<String>) result.rows).each { row -> events << row << "\n" }
            frameStats << [
                variantLabel,
                preset.name,
                s,
                result.accepted,
                acceptedPixels,
                result.rejectedNoSeed,
                result.rejectedArea,
                result.rejectedShape
            ].join("\t") << "\n"
        }

        if (s % 25 == 0 || s == slices) {
            println("  ${variantLabel}: ${s}/${slices} frames")
        }
    }
    imp.close()

    ImagePlus zImp = new ImagePlus("${datasetId}_${variantLabel}_robust_positive_z_float32", zStack)
    new FileSaver(zImp).saveAsTiffStack(outputDir.resolve("${datasetId}_${variantLabel}_robust_positive_z_float32.tif").toString())
    zImp.close()

    activePresets.each { preset ->
        String tag = presetTag(preset)
        ImagePlus maskImp = new ImagePlus("${datasetId}_${variantLabel}_${tag}_mask", maskStacks[tag])
        new FileSaver(maskImp).saveAsTiffStack(outputDir.resolve("${datasetId}_${variantLabel}_${tag}_mask.tif").toString())
        maskImp.close()

        ImagePlus labelImp = new ImagePlus("${datasetId}_${variantLabel}_${tag}_labels", labelStacks[tag])
        new FileSaver(labelImp).saveAsTiffStack(outputDir.resolve("${datasetId}_${variantLabel}_${tag}_labels.tif").toString())
        labelImp.close()
    }
}

Files.writeString(outputDir.resolve("parameters.txt"), params.toString())
Files.writeString(outputDir.resolve("candidate_events.tsv"), events.toString())
Files.writeString(outputDir.resolve("frame_stats.tsv"), frameStats.toString())

println("Done. Wrote outputs to ${outputDir}")
System.exit(0)
