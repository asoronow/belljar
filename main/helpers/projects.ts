// projects.ts

import fs from "fs";
import path from "path";
import { app } from "electron";
import { ProjectMetadata, AnimalMetadata } from "./types";

const bellJarFolder = require("os").homedir() + "/.belljar";
const projectsPath = path.join(bellJarFolder, "projects");

// Ensure the projects directory exists
if (!fs.existsSync(projectsPath)) {
  fs.mkdirSync(projectsPath, { recursive: true });
}

export function createProject(name: string, description: string): void {
  const projectDir = path.join(projectsPath, name);
  if (fs.existsSync(projectDir)) {
    throw new Error("Project already exists");
  }
  fs.mkdirSync(projectDir);
  const metadata: ProjectMetadata = {
    name,
    createdAt: new Date().toISOString(),
    lastModified: new Date().toISOString(),
    description,
    animals: {},
  };
  fs.writeFileSync(
    path.join(projectDir, "metadata.json"),
    JSON.stringify(metadata, null, 2)
  );
}

export function loadProject(name: string): ProjectMetadata {
  const projectDir = path.join(projectsPath, name);
  if (!fs.existsSync(projectDir)) {
    throw new Error("Project not found");
  }
  return JSON.parse(
    fs.readFileSync(path.join(projectDir, "metadata.json")).toString()
  );
}

export function deleteProject(name: string): void {
  const projectDir = path.join(projectsPath, name);
  if (!fs.existsSync(projectDir)) {
    throw new Error("Project not found");
  }
  fs.rmSync(projectDir, { recursive: true, force: true });
}

export function uploadFile(projectName: string, filePath: string): void {
  const projectDir = path.join(projectsPath, projectName);
  if (!fs.existsSync(projectDir)) {
    throw new Error("Project not found");
  }
  const fileName = path.basename(filePath);
  fs.copyFileSync(filePath, path.join(projectDir, fileName));
}

export function addAnimal(
  projectName: string,
  animalName: string,
  animalData: AnimalMetadata
): void {
  const projectDir = path.join(projectsPath, projectName);
  if (!fs.existsSync(projectDir)) {
    throw new Error("Project not found");
  }
  const metadataPath = path.join(projectDir, "metadata.json");
  const metadata: ProjectMetadata = JSON.parse(
    fs.readFileSync(metadataPath).toString()
  );

  metadata.animals[animalName] = animalData;
  metadata.lastModified = new Date().toISOString();

  fs.writeFileSync(metadataPath, JSON.stringify(metadata, null, 2));
  fs.mkdirSync(path.join(projectDir, animalName));
}
