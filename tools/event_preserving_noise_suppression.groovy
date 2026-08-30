// Event-preserving noise suppression for temporal high-pass calcium residuals.
//
// For each high-pass stack:
//   1. subtract each frame's spatial median to remove common-mode flicker,
//   2. compute positive-only local z-scores using a 23x23 spatial window,
//   3. write 3x3-neighborhood-supported binary masks for z >= 3, 4, and 5.
//
// This deliberately does not smooth across time.

import ij.IJ
import ij.ImagePlus
import ij.ImageStack
import ij.io.FileSaver
import ij.process.ByteProcessor
import ij.process.FloatProcessor
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.util.Arrays
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
final Path outputDir = outputRoot.resolve("EventPreservingNoiseSuppression").resolve(datasetId)
final int localWindow = Integer.parseInt(setting("local_window_px", "23"))
final int supportWindow = 3
final int supportMinPixels = 2
final double epsilon = Double.parseDouble(setting("epsilon", "1.0"))
final double noiseFloorFractionOfMadSigma = 0.25d
final double[] thresholds = [2.0d, 2.5d, 3.0d, 3.5d, 4.0d] as double[]
final String requestedSigmaLabel = setting("sigma_label", "")

final List<Map<String, String>> variants = [
    [label: "sigma04", file: "${datasetId}_hp_gaussian_sigma04f_float32.tif"],
    [label: "sigma06", file: "${datasetId}_hp_gaussian_sigma06f_float32.tif"],
    [label: "sigma08", file: "${datasetId}_hp_gaussian_sigma08f_float32.tif"],
]
final List<Map<String, String>> activeVariants = requestedSigmaLabel.isBlank()
    ? variants
    : [[label: "sigma${requestedSigmaLabel}", file: "${datasetId}_hp_gaussian_sigma${requestedSigmaLabel}f_float32.tif"]]

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

static double medianAbs(float[] values) {
    float[] copy = new float[values.length]
    for (int i = 0; i < values.length; i++) {
        copy[i] = Math.abs(values[i])
    }
    Arrays.sort(copy)
    int n = copy.length
    if ((n & 1) == 1) {
        return copy[n.intdiv(2)] as double
    }
    return 0.5d * ((copy[n.intdiv(2) - 1] as double) + (copy[n.intdiv(2)] as double))
}

static double[] integralImage(float[] values, int width, int height, boolean squared) {
    int stride = width + 1
    double[] integral = new double[(height + 1) * stride]
    for (int y = 1; y <= height; y++) {
        double rowSum = 0.0d
        int srcBase = (y - 1) * width
        int dstBase = y * stride
        int prevBase = (y - 1) * stride
        for (int x = 1; x <= width; x++) {
            double v = values[srcBase + x - 1] as double
            if (squared) {
                v *= v
            }
            rowSum += v
            integral[dstBase + x] = integral[prevBase + x] + rowSum
        }
    }
    return integral
}

static double rectSum(double[] integral, int stride, int x0, int y0, int x1, int y1) {
    int xa = x0
    int xb = x1 + 1
    int ya = y0
    int yb = y1 + 1
    return integral[yb * stride + xb] - integral[ya * stride + xb] -
        integral[yb * stride + xa] + integral[ya * stride + xa]
}

static float[] positiveLocalZ(float[] corrected, int width, int height, int windowSize, double epsilon, double noiseFloor) {
    int radius = windowSize.intdiv(2)
    int stride = width + 1
    double[] sums = integralImage(corrected, width, height, false)
    double[] sumsSq = integralImage(corrected, width, height, true)
    float[] z = new float[corrected.length]

    for (int y = 0; y < height; y++) {
        int y0 = Math.max(0, y - radius)
        int y1 = Math.min(height - 1, y + radius)
        int row = y * width
        for (int x = 0; x < width; x++) {
            int idx = row + x
            double signal = corrected[idx] as double
            if (signal <= 0.0d) {
                z[idx] = 0.0f
                continue
            }
            int x0 = Math.max(0, x - radius)
            int x1 = Math.min(width - 1, x + radius)
            int area = (x1 - x0 + 1) * (y1 - y0 + 1)
            double mean = rectSum(sums, stride, x0, y0, x1, y1) / area
            double meanSq = rectSum(sumsSq, stride, x0, y0, x1, y1) / area
            double variance = Math.max(0.0d, meanSq - mean * mean)
            double sigma = Math.max(Math.sqrt(variance), noiseFloor)
            z[idx] = ((signal - mean) / (sigma + epsilon)) as float
        }
    }
    return z
}

static byte[] supportedMask(float[] z, int width, int height, double threshold, int supportWindow, int supportMinPixels) {
    int radius = supportWindow.intdiv(2)
    byte[] raw = new byte[z.length]
    byte[] supported = new byte[z.length]
    for (int i = 0; i < z.length; i++) {
        raw[i] = z[i] >= threshold ? (byte) 1 : (byte) 0
    }

    for (int y = 0; y < height; y++) {
        int row = y * width
        for (int x = 0; x < width; x++) {
            int idx = row + x
            if (raw[idx] == 0) {
                continue
            }
            int count = 0
            for (int yy = Math.max(0, y - radius); yy <= Math.min(height - 1, y + radius); yy++) {
                int nrow = yy * width
                for (int xx = Math.max(0, x - radius); xx <= Math.min(width - 1, x + radius); xx++) {
                    if (raw[nrow + xx] != 0) {
                        count++
                    }
                }
            }
            supported[idx] = count >= supportMinPixels ? (byte) 255 : (byte) 0
        }
    }
    return supported
}

static int countMask(byte[] mask) {
    int count = 0
    for (byte b : mask) {
        if ((b & 0xff) > 0) {
            count++
        }
    }
    return count
}

static String thresholdLabel(double threshold) {
    return String.format("%03d", Math.round(threshold * 10.0d) as int)
}

StringBuilder params = new StringBuilder()
params << "dataset_id=${datasetId}\n"
params << "input_dir=${highPassDir}\n"
params << "output_dir=${outputDir}\n"
params << "frame_correction=per-frame spatial median subtraction\n"
params << "zscore=positive-only local z-score\n"
params << "local_window=${localWindow}x${localWindow}\n"
params << "epsilon=${epsilon}\n"
params << "noise_floor=max(${epsilon}, ${noiseFloorFractionOfMadSigma} * 1.4826 * frame_mad)\n"
params << "support_filter=${supportWindow}x${supportWindow}, min_active_pixels=${supportMinPixels}\n"
params << "thresholds=${thresholds.join(',')}\n"
params << "temporal_smoothing=none\n"

StringBuilder stats = new StringBuilder()
stats << "variant\tframe\tmedian\tmad_sigma\tnoise_floor\tz_min\tz_max\tz_mean"
thresholds.each { t -> stats << "\tmask_pixels_z${thresholdLabel(t)}" }
stats << "\n"

activeVariants.each { variant ->
    String label = variant.label
    Path inPath = highPassDir.resolve(variant.file)
    if (!Files.exists(inPath)) {
        throw new FileNotFoundException("Missing high-pass input: ${inPath}")
    }

    println("Processing ${label}: ${inPath}")
    ImagePlus imp = IJ.openImage(inPath.toString())
    if (imp == null) {
        throw new IllegalStateException("Could not open ${inPath}")
    }
    int width = imp.getWidth()
    int height = imp.getHeight()
    int slices = imp.getStackSize()

    ImageStack correctedStack = new ImageStack(width, height)
    ImageStack zStack = new ImageStack(width, height)
    Map<Double, ImageStack> maskStacks = new LinkedHashMap<>()
    thresholds.each { t -> maskStacks[t] = new ImageStack(width, height) }

    for (int s = 1; s <= slices; s++) {
        float[] raw = (float[]) imp.getStack().getProcessor(s).convertToFloatProcessor().getPixels()
        double med = median(raw)
        float[] corrected = new float[raw.length]
        for (int i = 0; i < raw.length; i++) {
            corrected[i] = ((raw[i] as double) - med) as float
        }

        double madSigma = 1.4826d * medianAbs(corrected)
        double noiseFloor = Math.max(epsilon, noiseFloorFractionOfMadSigma * madSigma)
        float[] z = positiveLocalZ(corrected, width, height, localWindow, epsilon, noiseFloor)

        correctedStack.addSlice("frame_${s}", new FloatProcessor(width, height, corrected))
        zStack.addSlice("frame_${s}", new FloatProcessor(width, height, z))

        double zMin = Double.POSITIVE_INFINITY
        double zMax = Double.NEGATIVE_INFINITY
        double zSum = 0.0d
        for (float value : z) {
            double v = value as double
            if (v < zMin) zMin = v
            if (v > zMax) zMax = v
            zSum += v
        }
        stats << "${label}\t${s}\t${med}\t${madSigma}\t${noiseFloor}\t${zMin}\t${zMax}\t${zSum / z.length}"

        thresholds.each { t ->
            byte[] mask = supportedMask(z, width, height, t, supportWindow, supportMinPixels)
            maskStacks[t].addSlice("frame_${s}", new ByteProcessor(width, height, mask))
            stats << "\t${countMask(mask)}"
        }
        stats << "\n"

        if (s % 25 == 0 || s == slices) {
            println("  ${label}: ${s}/${slices} frames")
        }
    }

    imp.close()

    ImagePlus correctedImp = new ImagePlus("${datasetId}_${label}_median_corrected_float32", correctedStack)
    Path correctedPath = outputDir.resolve("${datasetId}_${label}_median_corrected_float32.tif")
    new FileSaver(correctedImp).saveAsTiffStack(correctedPath.toString())
    correctedImp.close()

    ImagePlus zImp = new ImagePlus("${datasetId}_${label}_positive_local_z_float32", zStack)
    Path zPath = outputDir.resolve("${datasetId}_${label}_positive_local_z_float32.tif")
    new FileSaver(zImp).saveAsTiffStack(zPath.toString())
    zImp.close()

    thresholds.each { t ->
        ImagePlus maskImp = new ImagePlus("${datasetId}_${label}_mask_z${t}", maskStacks[t])
        String thresholdTag = thresholdLabel(t)
        Path maskPath = outputDir.resolve("${datasetId}_${label}_mask_z${thresholdTag}_support3x3_min2.tif")
        new FileSaver(maskImp).saveAsTiffStack(maskPath.toString())
        maskImp.close()
    }
}

Files.writeString(outputDir.resolve("parameters.txt"), params.toString())
Files.writeString(outputDir.resolve("frame_stats.tsv"), stats.toString())

println("Done. Wrote outputs to ${outputDir}")
System.exit(0)
