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
