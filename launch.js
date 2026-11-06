"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.hasInstalledRequirements = exports.requirementsNeedInstall = exports.hashRequirements = exports.parseJsonObject = exports.isJsonObject = void 0;
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
