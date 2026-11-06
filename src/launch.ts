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
