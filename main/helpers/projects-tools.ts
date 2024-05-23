// projects.ts

import fs from "fs";
import path from "path";
import { ProjectMetadata, AnimalMetadata } from "../common/types";
import { ProjectDataType } from "../common/enums";
import { dialog } from "electron";
import archiver from "archiver";
import extract from "extract-zip";
import { c } from "tar";
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

export function saveAsDialog(name: string): string | undefined {
  const result = dialog.showSaveDialogSync({
    title: "Export Project",
    defaultPath: `${name}.zip`,
    filters: [{ name: "Zip Files", extensions: ["zip"] }],
  });
  return result;
}

export function openFile(): string[] | undefined {
  const result = dialog.showOpenDialogSync({
    properties: ["openFile"],
  });
  return result;
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

export function importProject(): void {
  const result = dialog.showOpenDialogSync({
    title: "Import Project",
    properties: ["openFile"],
    filters: [{ name: "Zip Files", extensions: ["zip"] }],
  });
  if (result) {
    const importPath = result[0];
    const importName = path.basename(importPath, ".zip");
    const projectDir = path.join(projectsPath, importName);
    if (fs.existsSync(projectDir)) {
      throw new Error("Project already exists");
    }
    // make dir
    fs.mkdirSync(projectDir);
    extract(importPath, { dir: projectDir });
    // check if metadata.json exists if not throw error, not a valid project
    if (!fs.existsSync(path.join(projectDir, "metadata.json"))) {
      fs.rmSync(projectDir, { recursive: true, force: true });
      throw new Error("Invalid project");
    }
  }
}

export async function exportProject(name: string): Promise<void> {
  // compress and export the project directory
  const projectDir = path.join(projectsPath, name);
  if (!fs.existsSync(projectDir)) {
    throw new Error("Project not found");
  }
  const exportPath = saveAsDialog(name);
  if (!exportPath) {
    return;
  }
  const output = fs.createWriteStream(exportPath);
  const archive = archiver("zip", { zlib: { level: 9 } });
  output.on("close", function () {
    console.log(archive.pointer() + " total bytes");
    console.log(
      "archiver has been finalized and the output file descriptor has closed."
    );
  });
  archive.on("warning", function (err) {
    if (err.code === "ENOENT") {
      console.log(err);
    } else {
      throw err;
    }
  });
  archive.on("error", function (err) {
    throw err;
  });
  archive.pipe(output);
  archive.directory(projectDir, false);
  console.log(`Files in ${projectDir}`, fs.readdirSync(projectDir));
  await archive.finalize();
}

export function deleteFile(
  projectName: string,
  animalName: string,
  dataType: ProjectDataType,
  fileName: string
): void {
  const projectDir = path.join(projectsPath, projectName);
  if (!fs.existsSync(projectDir)) {
    throw new Error("Project not found");
  }
  fs.rmSync(path.join(projectDir, animalName, dataType, fileName));
}

export function uploadFile(
  projectName: string,
  animalName: string,
  dataType: ProjectDataType,
  filePath: string
): void {
  const projectDir = path.join(projectsPath, projectName);
  if (!fs.existsSync(projectDir)) {
    throw new Error("Project not found");
  }
  const fileName = path.basename(filePath);
  fs.copyFileSync(
    filePath,
    path.join(projectDir, animalName, dataType, fileName)
  );
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

export function getProjects(): string[] {
  // Get all directories in the projects folder
  return fs.readdirSync(projectsPath).filter((file) => {
    return fs.statSync(path.join(projectsPath, file)).isDirectory();
  });
}
