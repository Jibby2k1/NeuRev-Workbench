// Create signed 32-bit temporal high-pass TIFF stacks for a calcium video.
// Formula: high_pass = raw_32bit - GaussianBlur3D(raw_32bit, x=0, y=0, z=sigma_frames)

requires("1.54p");

function argValue(args, key, defaultValue) {
    parts = split(args, ",");
    prefix = key + "=";
    for (j = 0; j < parts.length; j++) {
        item = trim(parts[j]);
        if (startsWith(item, prefix)) {
            return substring(item, lengthOf(prefix));
        }
    }
    return defaultValue;
}

args = getArgument();
projectRoot = argValue(args, "project_root", getDirectory("current"));
datasetId = argValue(args, "dataset_id", "calcium_video_2");
inputPath = argValue(args, "input_path", projectRoot + "/Inputs/050126/050126/calcium video 2.tif");
outputRoot = argValue(args, "output_root", projectRoot + "/Outputs");
outputHighPass = outputRoot + "/HighPass";
outputDir = outputHighPass + "/" + datasetId;
sigmas = newArray(4, 6, 8);
labels = newArray("04", "06", "08");
requestedSigma = argValue(args, "sigma_frames", "");
requestedLabel = argValue(args, "sigma_label", "");
if (requestedSigma != "") {
    if (requestedLabel == "") requestedLabel = "custom";
    sigmas = newArray(parseFloat(requestedSigma));
    labels = newArray(requestedLabel);
}

File.makeDirectory(outputRoot);
File.makeDirectory(outputHighPass);
File.makeDirectory(outputDir);
setBatchMode(true);

open(inputPath);
run("32-bit");
rename("raw32_" + datasetId);
rawTitle = getTitle();
getDimensions(width, height, channels, slices, frames);

summary = "";
summary += "dataset_id=" + datasetId + "\n";
summary += "input_path=" + inputPath + "\n";
summary += "output_dir=" + outputDir + "\n";
summary += "formula=high_pass = raw_32bit - GaussianBlur3D(raw_32bit, x=0, y=0, z=sigma_frames)\n";
summary += "sigma_frames=";
for (i = 0; i < sigmas.length; i++) {
    if (i > 0) summary += ",";
    summary += sigmas[i];
}
summary += "\n";
summary += "output_dtype=32-bit float signed residual TIFF stack\n";
summary += "width=" + width + "\n";
summary += "height=" + height + "\n";
summary += "channels=" + channels + "\n";
summary += "slices=" + slices + "\n";
summary += "frames=" + frames + "\n";

for (i = 0; i < sigmas.length; i++) {
    sigma = sigmas[i];
    label = labels[i];

    selectWindow(rawTitle);
    run("Duplicate...", "title=baseline_sigma" + label + " duplicate");
    baselineTitle = getTitle();

    run("Gaussian Blur 3D...", "x=0 y=0 z=" + sigma);

    imageCalculator("Subtract create 32-bit stack", rawTitle, baselineTitle);
    resultTitle = getTitle();
    rename(datasetId + "_hp_gaussian_sigma" + label + "f_float32");

    savePath = outputDir + "/" + datasetId + "_hp_gaussian_sigma" + label + "f_float32.tif";
    saveAs("Tiff", savePath);
    summary += "wrote=" + savePath + "\n";

    close();
    selectWindow(baselineTitle);
    close();
}

selectWindow(rawTitle);
close();

File.saveString(summary, outputDir + "/parameters.txt");
setBatchMode(false);
print(summary);
call("java.lang.System.exit", "0");
