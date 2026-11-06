"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.detectionArgs = exports.sharpenArgs = exports.collateArgs = exports.countArgs = exports.intensityArgs = exports.alignArgs = exports.adjustArgs = exports.maxArgs = exports.isTrue = exports.naturalCompare = exports.resourcesToDownload = exports.RESOURCES = exports.hasInstalledRequirements = exports.requirementsNeedInstall = exports.hashRequirements = exports.parseJsonObject = exports.isJsonObject = void 0;
const crypto_1 = require("crypto");
function isJsonObject(value) {
    return typeof value === "object" && value !== null && !Array.isArray(value);
}
exports.isJsonObject = isJsonObject;
function parseJsonObject(text) {
    try {
        const parsed = JSON.parse(text);
        return isJsonObject(parsed) ? parsed : {};
    }
    catch (_a) {
        return {};
    }
}
exports.parseJsonObject = parseJsonObject;
function hashRequirements(contents) {
    return (0, crypto_1.createHash)("sha256").update(contents).digest("hex");
}
exports.hashRequirements = hashRequirements;
function requirementsNeedInstall(hash, state) {
    return !isJsonObject(state) || state.requirementsHash !== hash;
}
exports.requirementsNeedInstall = requirementsNeedInstall;
function hasInstalledRequirements(state) {
    return isJsonObject(state) && typeof state.requirementsHash === "string";
}
exports.hasInstalledRequirements = hasInstalledRequirements;
// The models manifest string predates the archive naming scheme: every
// install that already wrote "v952" holds the files from models-v10.tar.gz,
// so the two names must stay decoupled until the archive itself changes.
exports.RESOURCES = {
    nrrd: { version: "v91", filename: "nrrd-v91.tar.gz" },
    models: { version: "v952", filename: "models-v10.tar.gz" },
    embeddings: { version: "v6", filename: "embeddings-v6.tar.gz" },
};
function resourcesToDownload(manifest, table, dirsPresent) {
    const recorded = isJsonObject(manifest) ? manifest : {};
    return Object.keys(table).filter((key) => !dirsPresent[key] || recorded[key] !== table[key].version);
}
exports.resourcesToDownload = resourcesToDownload;
function naturalCompare(a, b) {
    return a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" });
}
exports.naturalCompare = naturalCompare;
function isTrue(value) {
    return (value === true ||
        (typeof value === "string" && value.trim().toLowerCase() === "true"));
}
exports.isTrue = isTrue;
function pythonBool(value) {
    return isTrue(value) ? "True" : "False";
}
function maxArgs(o) {
    return [
        "-o", o.output,
        "-i", o.input,
        "-d", pythonBool(o.dendrite),
        "-t", pythonBool(o.tophat),
        "-g", "False",
    ];
}
exports.maxArgs = maxArgs;
function adjustArgs(o) {
    return ["-i", o.images, "-s", o.structures, "-a", o.annotations];
}
exports.adjustArgs = adjustArgs;
function alignArgs(o) {
    return [
        "-o", o.output,
        "-i", o.input,
        "-w", pythonBool(o.whole),
        "-a", String(o.spacing),
        "-m", o.model,
        "-n", o.nrrd,
        "-c", o.map,
        "-l", pythonBool(o.legacy),
    ];
}
exports.alignArgs = alignArgs;
function intensityArgs(o) {
    const args = ["-i", o.images, "-o", o.output, "-a", o.annotations, "-m", o.map];
    if (isTrue(o.whole)) {
        args.push("--whole");
    }
    return args;
}
exports.intensityArgs = intensityArgs;
function countArgs(o) {
    const args = ["-p", o.predictions, "-a", o.annotations, "-o", o.output, "-m", o.structures];
    if (o.layers) {
        args.push("--layers");
    }
    return args;
}
exports.countArgs = countArgs;
function collateArgs(o) {
    return ["-o", o.output, "-i", o.input, "-r", o.regions, "-s", o.structures, "-g", "False"];
}
exports.collateArgs = collateArgs;
function sharpenArgs(o) {
    const args = ["-o", o.output, "-i", o.input, "-r", String(o.radius), "-a", String(o.amount)];
    if (o.equalize) {
        args.push("--equalize");
    }
    return args;
}
exports.sharpenArgs = sharpenArgs;
function detectionArgs(o) {
    const args = [
        "-i", o.input,
        "-o", o.output,
        "-c", String(o.confidence),
        "-t", String(o.tile),
        "-a", String(o.area),
        "-s", o.sam,
        "-e", String(o.eccentricity),
        "-m", o.model,
    ];
    if (o.multichannel) {
        args.push("--multichannel");
    }
    return args;
}
exports.detectionArgs = detectionArgs;
