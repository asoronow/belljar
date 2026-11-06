import { createHash } from "crypto";

export type JsonObject = Record<string, unknown>;

export function isJsonObject(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function parseJsonObject(text: string): JsonObject {
  try {
    const parsed: unknown = JSON.parse(text);
    return isJsonObject(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

export function hashRequirements(contents: string | Buffer): string {
  return createHash("sha256").update(contents).digest("hex");
}

export function requirementsNeedInstall(hash: string, state: unknown): boolean {
  return !isJsonObject(state) || state.requirementsHash !== hash;
}

export function hasInstalledRequirements(state: unknown): boolean {
  return isJsonObject(state) && typeof state.requirementsHash === "string";
}

export interface Resource {
  version: string;
  filename: string;
}

export type ResourceTable = Record<string, Resource>;

// The models manifest string predates the archive naming scheme: every
// install that already wrote "v952" holds the files from models-v10.tar.gz,
// so the two names must stay decoupled until the archive itself changes.
export const RESOURCES: ResourceTable = {
  nrrd: { version: "v91", filename: "nrrd-v91.tar.gz" },
  models: { version: "v952", filename: "models-v10.tar.gz" },
  embeddings: { version: "v6", filename: "embeddings-v6.tar.gz" },
};

export function resourcesToDownload(
  manifest: unknown,
  table: ResourceTable,
  dirsPresent: Record<string, boolean>
): string[] {
  const recorded = isJsonObject(manifest) ? manifest : {};
  return Object.keys(table).filter(
    (key) => !dirsPresent[key] || recorded[key] !== table[key].version
  );
}

export function naturalCompare(a: string, b: string): number {
  return a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" });
}

export function isTrue(value: unknown): boolean {
  return (
    value === true ||
    (typeof value === "string" && value.trim().toLowerCase() === "true")
  );
}

function pythonBool(value: unknown): string {
  return isTrue(value) ? "True" : "False";
}

export function maxArgs(o: { input: string; output: string }): string[] {
  return ["-o", o.output, "-i", o.input, "-g", "False"];
}

export function adjustArgs(o: {
  images: string;
  annotations: string;
  structures: string;
}): string[] {
  return ["-i", o.images, "-s", o.structures, "-a", o.annotations];
}

export function alignArgs(o: {
  input: string;
  output: string;
  whole: unknown;
  spacing: unknown;
  legacy: unknown;
  model: string;
  nrrd: string;
  map: string;
}): string[] {
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

export function intensityArgs(o: {
  images: string;
  output: string;
  annotations: string;
  map: string;
  whole: unknown;
}): string[] {
  const args = ["-i", o.images, "-o", o.output, "-a", o.annotations, "-m", o.map];
  if (isTrue(o.whole)) {
    args.push("--whole");
  }
  return args;
}

export function countArgs(o: {
  predictions: string;
  annotations: string;
  output: string;
  structures: string;
  layers: unknown;
}): string[] {
  const args = ["-p", o.predictions, "-a", o.annotations, "-o", o.output, "-m", o.structures];
  if (o.layers) {
    args.push("--layers");
  }
  return args;
}

export function collateArgs(o: {
  input: string;
  output: string;
  regions: string;
  structures: string;
}): string[] {
  return ["-o", o.output, "-i", o.input, "-r", o.regions, "-s", o.structures, "-g", "False"];
}

export function sharpenArgs(o: {
  input: string;
  output: string;
  radius: unknown;
  amount: unknown;
  equalize: unknown;
}): string[] {
  const args = ["-o", o.output, "-i", o.input, "-r", String(o.radius), "-a", String(o.amount)];
  if (o.equalize) {
    args.push("--equalize");
  }
  return args;
}

export function detectionArgs(o: {
  input: string;
  output: string;
  confidence: unknown;
  tile: unknown;
  area: unknown;
  sam: string;
  eccentricity: unknown;
  model: string;
  multichannel: unknown;
}): string[] {
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
