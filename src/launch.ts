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
