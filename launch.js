"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.RESOURCES = void 0;
exports.isJsonObject = isJsonObject;
exports.parseJsonObject = parseJsonObject;
exports.hashRequirements = hashRequirements;
exports.requirementsNeedInstall = requirementsNeedInstall;
exports.hasInstalledRequirements = hasInstalledRequirements;
exports.environmentAction = environmentAction;
exports.brokenEnvName = brokenEnvName;
exports.resourcesToDownload = resourcesToDownload;
exports.naturalCompare = naturalCompare;
exports.isTrue = isTrue;
exports.maxArgs = maxArgs;
exports.adjustArgs = adjustArgs;
exports.alignArgs = alignArgs;
exports.intensityArgs = intensityArgs;
exports.countArgs = countArgs;
exports.collateArgs = collateArgs;
exports.sharpenArgs = sharpenArgs;
exports.detectionArgs = detectionArgs;
const crypto_1 = require("crypto");
function isJsonObject(value) {
    return typeof value === "object" && value !== null && !Array.isArray(value);
}
function parseJsonObject(text) {
    try {
        const parsed = JSON.parse(text);
        return isJsonObject(parsed) ? parsed : {};
    }
    catch (_a) {
        return {};
    }
}
function hashRequirements(contents) {
    return (0, crypto_1.createHash)("sha256").update(contents).digest("hex");
}
function requirementsNeedInstall(hash, state) {
    return !isJsonObject(state) || state.requirementsHash !== hash;
}
function hasInstalledRequirements(state) {
    return isJsonObject(state) && typeof state.requirementsHash === "string";
}
// A venv links back to the interpreter it was created from, so its directory
// can outlive a usable python: the env is only trusted when its own
// interpreter still resolves.
function environmentAction(probe) {
    if (!probe.pythonExists || !probe.envDirExists) {
        return "fresh-install";
    }
    return probe.envInterpreterResolves ? "ok" : "rebuild-env";
}
function brokenEnvName(now) {
    const stamp = now.toISOString().replace(/[-:]/g, "").replace(/\.\d+Z$/, "Z");
    return `benv.broken-${stamp}`;
}
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
function naturalCompare(a, b) {
    return a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" });
}
function isTrue(value) {
    return (value === true ||
        (typeof value === "string" && value.trim().toLowerCase() === "true"));
}
function pythonBool(value) {
    return isTrue(value) ? "True" : "False";
}
function maxArgs(o) {
    return ["-o", o.output, "-i", o.input, "-g", "False"];
}
function adjustArgs(o) {
    return ["-i", o.images, "-s", o.structures, "-a", o.annotations];
}
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
function intensityArgs(o) {
    const args = ["-i", o.images, "-o", o.output, "-a", o.annotations, "-m", o.map];
    if (isTrue(o.whole)) {
        args.push("--whole");
    }
    return args;
}
function countArgs(o) {
    const args = ["-p", o.predictions, "-a", o.annotations, "-o", o.output, "-m", o.structures];
    if (o.layers) {
        args.push("--layers");
    }
    return args;
}
function collateArgs(o) {
    return ["-o", o.output, "-i", o.input, "-r", o.regions, "-s", o.structures, "-g", "False"];
}
function sharpenArgs(o) {
    const args = ["-o", o.output, "-i", o.input, "-r", String(o.radius), "-a", String(o.amount)];
    if (o.equalize) {
        args.push("--equalize");
    }
    return args;
}
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
