"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const strict_1 = __importDefault(require("node:assert/strict"));
const node_test_1 = require("node:test");
const launch_1 = require("../launch");
const HASH = (0, launch_1.hashRequirements)("numpy<2.0.0\n");
(0, node_test_1.test)("requirementsNeedInstall is true when state is missing, malformed, or stale", () => {
    strict_1.default.equal((0, launch_1.requirementsNeedInstall)(HASH, {}), true);
    strict_1.default.equal((0, launch_1.requirementsNeedInstall)(HASH, undefined), true);
    strict_1.default.equal((0, launch_1.requirementsNeedInstall)(HASH, "garbage"), true);
    strict_1.default.equal((0, launch_1.requirementsNeedInstall)(HASH, { requirementsHash: "old" }), true);
});
(0, node_test_1.test)("requirementsNeedInstall is false when the stored hash matches", () => {
    strict_1.default.equal((0, launch_1.requirementsNeedInstall)(HASH, { requirementsHash: HASH }), false);
});
(0, node_test_1.test)("hasInstalledRequirements reports a previous successful install", () => {
    strict_1.default.equal((0, launch_1.hasInstalledRequirements)({ requirementsHash: "old" }), true);
    strict_1.default.equal((0, launch_1.hasInstalledRequirements)({}), false);
    strict_1.default.equal((0, launch_1.hasInstalledRequirements)(null), false);
});
(0, node_test_1.test)("hashRequirements is a stable sha256 hex digest", () => {
    strict_1.default.equal(HASH.length, 64);
    strict_1.default.equal((0, launch_1.hashRequirements)(Buffer.from("numpy<2.0.0\n")), HASH);
    strict_1.default.notEqual((0, launch_1.hashRequirements)("numpy<2.0.0\nscipy\n"), HASH);
});
(0, node_test_1.test)("parseJsonObject tolerates invalid or non-object JSON", () => {
    strict_1.default.deepEqual((0, launch_1.parseJsonObject)("{\"a\": 1}"), { a: 1 });
    strict_1.default.deepEqual((0, launch_1.parseJsonObject)("not json"), {});
    strict_1.default.deepEqual((0, launch_1.parseJsonObject)("[1, 2]"), {});
});
(0, node_test_1.test)("resource table pairs each manifest version with the archive that holds it", () => {
    strict_1.default.equal(launch_1.RESOURCES.nrrd.filename, "nrrd-v91.tar.gz");
    strict_1.default.equal(launch_1.RESOURCES.models.filename, "models-v10.tar.gz");
    strict_1.default.equal(launch_1.RESOURCES.embeddings.filename, "embeddings-v6.tar.gz");
});
const table = {
    nrrd: { version: "v91", filename: "nrrd-v91.tar.gz" },
    models: { version: "v952", filename: "models-v10.tar.gz" },
};
const allPresent = { nrrd: true, models: true };
(0, node_test_1.test)("resourcesToDownload downloads nothing when manifest matches and dirs are present", () => {
    strict_1.default.deepEqual((0, launch_1.resourcesToDownload)({ nrrd: "v91", models: "v952" }, table, allPresent), []);
});
(0, node_test_1.test)("resourcesToDownload downloads keys whose manifest version differs or is absent", () => {
    strict_1.default.deepEqual((0, launch_1.resourcesToDownload)({ nrrd: "v90", models: "v952" }, table, allPresent), ["nrrd"]);
    strict_1.default.deepEqual((0, launch_1.resourcesToDownload)({ nrrd: "v91" }, table, allPresent), ["models"]);
    strict_1.default.deepEqual((0, launch_1.resourcesToDownload)({}, table, allPresent), ["nrrd", "models"]);
    strict_1.default.deepEqual((0, launch_1.resourcesToDownload)(undefined, table, allPresent), ["nrrd", "models"]);
});
(0, node_test_1.test)("resourcesToDownload downloads keys whose directory is missing even when the manifest is current", () => {
    strict_1.default.deepEqual((0, launch_1.resourcesToDownload)({ nrrd: "v91", models: "v952" }, table, { nrrd: true, models: false }), ["models"]);
});
(0, node_test_1.test)("naturalCompare orders embedded numbers numerically", () => {
    const names = ["s10.tif", "s2.tif", "s1.tif", "s100.tif"];
    strict_1.default.deepEqual([...names].sort(launch_1.naturalCompare), ["s1.tif", "s2.tif", "s10.tif", "s100.tif"]);
});
const IN = "/data/my data ";
const OUT = "/out/my results ";
(0, node_test_1.test)("maxArgs keeps flag and value as separate tokens and preserves trailing spaces", () => {
    strict_1.default.deepEqual((0, launch_1.maxArgs)({ input: IN, output: OUT }), ["-o", OUT, "-i", IN, "-g", "False"]);
});
(0, node_test_1.test)("adjustArgs", () => {
    strict_1.default.deepEqual((0, launch_1.adjustArgs)({ images: IN, annotations: OUT, structures: "/s.pkl" }), [
        "-i", IN, "-s", "/s.pkl", "-a", OUT,
    ]);
});
(0, node_test_1.test)("alignArgs normalises booleans to the strings map.py expects", () => {
    strict_1.default.deepEqual((0, launch_1.alignArgs)({ input: IN, output: OUT, whole: "True", spacing: 10, legacy: "False", model: "/m.pt", nrrd: "/n", map: "/c.pkl" }), ["-o", OUT, "-i", IN, "-w", "True", "-a", "10", "-m", "/m.pt", "-n", "/n", "-c", "/c.pkl", "-l", "False"]);
    strict_1.default.deepEqual((0, launch_1.alignArgs)({ input: IN, output: OUT, whole: true, spacing: 0, legacy: false, model: "/m.pt", nrrd: "/n", map: "/c.pkl" }).slice(4, 6), ["-w", "True"]);
});
(0, node_test_1.test)("intensityArgs passes --whole only when the renderer selected a whole slice", () => {
    const base = { images: IN, output: OUT, annotations: "/ann", map: "/m.pkl" };
    strict_1.default.deepEqual((0, launch_1.intensityArgs)(Object.assign(Object.assign({}, base), { whole: "True" })), [
        "-i", IN, "-o", OUT, "-a", "/ann", "-m", "/m.pkl", "--whole",
    ]);
    strict_1.default.deepEqual((0, launch_1.intensityArgs)(Object.assign(Object.assign({}, base), { whole: "False" })), [
        "-i", IN, "-o", OUT, "-a", "/ann", "-m", "/m.pkl",
    ]);
    strict_1.default.deepEqual((0, launch_1.intensityArgs)(Object.assign(Object.assign({}, base), { whole: true })).slice(-1), ["--whole"]);
});
(0, node_test_1.test)("countArgs appends --layers only when requested", () => {
    const base = { predictions: IN, annotations: "/ann", output: OUT, structures: "/s.pkl" };
    strict_1.default.deepEqual((0, launch_1.countArgs)(Object.assign(Object.assign({}, base), { layers: true })), [
        "-p", IN, "-a", "/ann", "-o", OUT, "-m", "/s.pkl", "--layers",
    ]);
    strict_1.default.deepEqual((0, launch_1.countArgs)(Object.assign(Object.assign({}, base), { layers: false })).includes("--layers"), false);
});
(0, node_test_1.test)("collateArgs", () => {
    strict_1.default.deepEqual((0, launch_1.collateArgs)({ input: IN, output: OUT, regions: "a b", structures: "/s.pkl" }), [
        "-o", OUT, "-i", IN, "-r", "a b", "-s", "/s.pkl", "-g", "False",
    ]);
});
(0, node_test_1.test)("sharpenArgs stringifies numbers and appends --equalize only when set", () => {
    strict_1.default.deepEqual((0, launch_1.sharpenArgs)({ input: IN, output: OUT, radius: 3, amount: 2.5, equalize: "equalize" }), [
        "-o", OUT, "-i", IN, "-r", "3", "-a", "2.5", "--equalize",
    ]);
    strict_1.default.deepEqual((0, launch_1.sharpenArgs)({ input: IN, output: OUT, radius: 3, amount: 2, equalize: undefined }).includes("--equalize"), false);
});
(0, node_test_1.test)("detectionArgs", () => {
    strict_1.default.deepEqual((0, launch_1.detectionArgs)({ input: IN, output: OUT, confidence: 0.5, tile: 640, area: 200, sam: "/sam.pth", eccentricity: 0.95, model: "/m.pt", multichannel: true }), ["-i", IN, "-o", OUT, "-c", "0.5", "-t", "640", "-a", "200", "-s", "/sam.pth", "-e", "0.95", "-m", "/m.pt", "--multichannel"]);
});
(0, node_test_1.test)("environmentAction installs from scratch when python or the env is absent", () => {
    strict_1.default.equal((0, launch_1.environmentAction)({ pythonExists: false, envDirExists: false, envInterpreterResolves: false }), "fresh-install");
    strict_1.default.equal((0, launch_1.environmentAction)({ pythonExists: false, envDirExists: true, envInterpreterResolves: false }), "fresh-install");
    strict_1.default.equal((0, launch_1.environmentAction)({ pythonExists: true, envDirExists: false, envInterpreterResolves: false }), "fresh-install");
});
(0, node_test_1.test)("environmentAction rebuilds an env whose interpreter no longer resolves", () => {
    strict_1.default.equal((0, launch_1.environmentAction)({ pythonExists: true, envDirExists: true, envInterpreterResolves: false }), "rebuild-env");
});
(0, node_test_1.test)("environmentAction leaves a working env alone", () => {
    strict_1.default.equal((0, launch_1.environmentAction)({ pythonExists: true, envDirExists: true, envInterpreterResolves: true }), "ok");
});
(0, node_test_1.test)("brokenEnvName is a sortable name without characters Windows rejects", () => {
    const name = (0, launch_1.brokenEnvName)(new Date("2026-11-11T19:04:05.678Z"));
    strict_1.default.equal(name, "benv.broken-20261111T190405Z");
    strict_1.default.doesNotMatch(name, /[:\\/]/);
});
