import assert from "node:assert/strict";
import { test } from "node:test";
import {
  RESOURCES,
  hasInstalledRequirements,
  hashRequirements,
  parseJsonObject,
  requirementsNeedInstall,
  resourcesToDownload,
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

test("resource table pairs each manifest version with the archive that holds it", () => {
  assert.equal(RESOURCES.nrrd.filename, "nrrd-v91.tar.gz");
  assert.equal(RESOURCES.models.filename, "models-v10.tar.gz");
  assert.equal(RESOURCES.embeddings.filename, "embeddings-v6.tar.gz");
});

const table = {
  nrrd: { version: "v91", filename: "nrrd-v91.tar.gz" },
  models: { version: "v952", filename: "models-v10.tar.gz" },
};
const allPresent = { nrrd: true, models: true };

test("resourcesToDownload downloads nothing when manifest matches and dirs are present", () => {
  assert.deepEqual(resourcesToDownload({ nrrd: "v91", models: "v952" }, table, allPresent), []);
});

test("resourcesToDownload downloads keys whose manifest version differs or is absent", () => {
  assert.deepEqual(resourcesToDownload({ nrrd: "v90", models: "v952" }, table, allPresent), ["nrrd"]);
  assert.deepEqual(resourcesToDownload({ nrrd: "v91" }, table, allPresent), ["models"]);
  assert.deepEqual(resourcesToDownload({}, table, allPresent), ["nrrd", "models"]);
  assert.deepEqual(resourcesToDownload(undefined, table, allPresent), ["nrrd", "models"]);
});

test("resourcesToDownload downloads keys whose directory is missing even when the manifest is current", () => {
  assert.deepEqual(
    resourcesToDownload({ nrrd: "v91", models: "v952" }, table, { nrrd: true, models: false }),
    ["models"]
  );
});
