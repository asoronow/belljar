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
