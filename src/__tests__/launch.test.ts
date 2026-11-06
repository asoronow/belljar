import assert from "node:assert/strict";
import { test } from "node:test";
import {
  hasInstalledRequirements,
  hashRequirements,
  parseJsonObject,
  requirementsNeedInstall,
} from "../launch";

const HASH = hashRequirements("numpy<2.0.0\n");

test("requirementsNeedInstall is true when state is missing, malformed, or stale", () => {
  assert.equal(requirementsNeedInstall(HASH, {}), true);
  assert.equal(requirementsNeedInstall(HASH, undefined), true);
  assert.equal(requirementsNeedInstall(HASH, "garbage"), true);
  assert.equal(requirementsNeedInstall(HASH, { requirementsHash: "old" }), true);
});

test("requirementsNeedInstall is false when the stored hash matches", () => {
  assert.equal(requirementsNeedInstall(HASH, { requirementsHash: HASH }), false);
});

test("hasInstalledRequirements reports a previous successful install", () => {
  assert.equal(hasInstalledRequirements({ requirementsHash: "old" }), true);
  assert.equal(hasInstalledRequirements({}), false);
  assert.equal(hasInstalledRequirements(null), false);
});

test("hashRequirements is a stable sha256 hex digest", () => {
  assert.equal(HASH.length, 64);
  assert.equal(hashRequirements(Buffer.from("numpy<2.0.0\n")), HASH);
  assert.notEqual(hashRequirements("numpy<2.0.0\nscipy\n"), HASH);
});

test("parseJsonObject tolerates invalid or non-object JSON", () => {
  assert.deepEqual(parseJsonObject("{\"a\": 1}"), { a: 1 });
  assert.deepEqual(parseJsonObject("not json"), {});
  assert.deepEqual(parseJsonObject("[1, 2]"), {});
});
