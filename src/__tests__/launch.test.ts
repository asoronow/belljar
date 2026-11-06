import assert from "node:assert/strict";
import { test } from "node:test";
import {
  RESOURCES,
  adjustArgs,
  alignArgs,
  collateArgs,
  countArgs,
  detectionArgs,
  hasInstalledRequirements,
  hashRequirements,
  intensityArgs,
  maxArgs,
  naturalCompare,
  parseJsonObject,
  requirementsNeedInstall,
  resourcesToDownload,
  sharpenArgs,
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

test("naturalCompare orders embedded numbers numerically", () => {
  const names = ["s10.tif", "s2.tif", "s1.tif", "s100.tif"];
  assert.deepEqual([...names].sort(naturalCompare), ["s1.tif", "s2.tif", "s10.tif", "s100.tif"]);
});

const IN = "/data/my data ";
const OUT = "/out/my results ";

test("maxArgs keeps flag and value as separate tokens and preserves trailing spaces", () => {
  assert.deepEqual(maxArgs({ input: IN, output: OUT, dendrite: false, tophat: false }), [
    "-o", OUT, "-i", IN, "-d", "False", "-t", "False", "-g", "False",
  ]);
});

test("adjustArgs", () => {
  assert.deepEqual(adjustArgs({ images: IN, annotations: OUT, structures: "/s.pkl" }), [
    "-i", IN, "-s", "/s.pkl", "-a", OUT,
  ]);
});

test("alignArgs normalises booleans to the strings map.py expects", () => {
  assert.deepEqual(
    alignArgs({ input: IN, output: OUT, whole: "True", spacing: 10, legacy: "False", model: "/m.pt", nrrd: "/n", map: "/c.pkl" }),
    ["-o", OUT, "-i", IN, "-w", "True", "-a", "10", "-m", "/m.pt", "-n", "/n", "-c", "/c.pkl", "-l", "False"]
  );
  assert.deepEqual(
    alignArgs({ input: IN, output: OUT, whole: true, spacing: 0, legacy: false, model: "/m.pt", nrrd: "/n", map: "/c.pkl" }).slice(4, 6),
    ["-w", "True"]
  );
});

test("intensityArgs passes --whole only when the renderer selected a whole slice", () => {
  const base = { images: IN, output: OUT, annotations: "/ann", map: "/m.pkl" };
  assert.deepEqual(intensityArgs({ ...base, whole: "True" }), [
    "-i", IN, "-o", OUT, "-a", "/ann", "-m", "/m.pkl", "--whole",
  ]);
  assert.deepEqual(intensityArgs({ ...base, whole: "False" }), [
    "-i", IN, "-o", OUT, "-a", "/ann", "-m", "/m.pkl",
  ]);
  assert.deepEqual(intensityArgs({ ...base, whole: true }).slice(-1), ["--whole"]);
});

test("countArgs appends --layers only when requested", () => {
  const base = { predictions: IN, annotations: "/ann", output: OUT, structures: "/s.pkl" };
  assert.deepEqual(countArgs({ ...base, layers: true }), [
    "-p", IN, "-a", "/ann", "-o", OUT, "-m", "/s.pkl", "--layers",
  ]);
  assert.deepEqual(countArgs({ ...base, layers: false }).includes("--layers"), false);
});

test("collateArgs", () => {
  assert.deepEqual(collateArgs({ input: IN, output: OUT, regions: "a b", structures: "/s.pkl" }), [
    "-o", OUT, "-i", IN, "-r", "a b", "-s", "/s.pkl", "-g", "False",
  ]);
});

test("sharpenArgs stringifies numbers and appends --equalize only when set", () => {
  assert.deepEqual(sharpenArgs({ input: IN, output: OUT, radius: 3, amount: 2.5, equalize: "equalize" }), [
    "-o", OUT, "-i", IN, "-r", "3", "-a", "2.5", "--equalize",
  ]);
  assert.deepEqual(sharpenArgs({ input: IN, output: OUT, radius: 3, amount: 2, equalize: undefined }).includes("--equalize"), false);
});

test("detectionArgs", () => {
  assert.deepEqual(
    detectionArgs({ input: IN, output: OUT, confidence: 0.5, tile: 640, area: 200, sam: "/sam.pth", eccentricity: 0.95, model: "/m.pt", multichannel: true }),
    ["-i", IN, "-o", OUT, "-c", "0.5", "-t", "640", "-a", "200", "-s", "/sam.pth", "-e", "0.95", "-m", "/m.pt", "--multichannel"]
  );
});
