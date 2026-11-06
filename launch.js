"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.resourcesToDownload = exports.RESOURCES = exports.hasInstalledRequirements = exports.requirementsNeedInstall = exports.hashRequirements = exports.parseJsonObject = exports.isJsonObject = void 0;
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
