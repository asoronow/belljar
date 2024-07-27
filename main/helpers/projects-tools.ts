// projects.ts

import fs from "fs";
import path from "path";
import { ProjectMetadata, AnimalMetadata } from "../common/types";
import { dialog } from "electron";
import archiver from "archiver";
import extract from "extract-zip";
import { v4 as uuid } from "uuid";
const bellJarFolder = require("os").homedir() + "/.belljar";
const projectsPath = path.join(bellJarFolder, "projects");

export interface ProjectFile {
  name: string;
  path: string;
  addedAt: string;
}
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
  dataType: string,
  fileName: string
): void {
  const projectDir = path.join(projectsPath, projectName);
  if (!fs.existsSync(projectDir)) {
    throw new Error("Project not found");
  }
  // remove meta
  fs.rmSync(path.join(projectDir, animalName, dataType, fileName + ".json"));
  // remove file
  fs.rmSync(path.join(projectDir, animalName, dataType, fileName));
  // Update metadata
  const metadataPath = path.join(projectDir, "metadata.json");
  const metadata: ProjectMetadata = JSON.parse(
    fs.readFileSync(metadataPath).toString()
  );
  metadata.lastModified = new Date().toISOString();
  if (
    fs.readdirSync(path.join(projectDir, animalName, dataType)).length === 0
  ) {
    // Set the data flag to false
    switch (dataType) {
      case "Background":
        metadata.animals[animalName].hasAlignmentData = false;
        break;
      case "Signal":
        metadata.animals[animalName].hasCellDetectionData = false;
        break;
    }
  }
  fs.writeFileSync(metadataPath, JSON.stringify(metadata, null, 2));
}

export function uploadFile(
  projectName: string,
  animalName: string,
  dataType: string,
  filePaths: [string]
): void {
  const projectDir = path.join(projectsPath, projectName);
  if (!fs.existsSync(projectDir)) {
    throw new Error("Project not found");
  }
  filePaths.forEach((filePath) => {
    const fileName = path.basename(filePath);
    const fileMetadata: ProjectFile = {
      name: fileName,
      path: filePath,
      addedAt: new Date().toISOString(),
    };
    const dataTypeDir = path.join(projectDir, animalName, dataType);
    if (!fs.existsSync(dataTypeDir)) {
      fs.mkdirSync(dataTypeDir);
    }
    fs.writeFileSync(
      path.join(dataTypeDir, fileName + ".json"),
      JSON.stringify(fileMetadata, null, 2)
    );
    // Make a symlink to the file for analysis
    fs.symlinkSync(filePath, path.join(dataTypeDir, fileName));
  });

  // Update metadata
  const metadataPath = path.join(projectDir, "metadata.json");
  const metadata: ProjectMetadata = JSON.parse(
    fs.readFileSync(metadataPath).toString()
  );
  metadata.lastModified = new Date().toISOString();
  switch (dataType) {
    case "Background":
      metadata.animals[animalName].hasAlignmentData = true;
      break;
    case "Signal":
      metadata.animals[animalName].hasCellDetectionData = true;
      break;
    default:
      break;
  }

  fs.writeFileSync(metadataPath, JSON.stringify(metadata, null, 2));
}

export function getAnimalData(
  projectName: string,
  animalName: string
): Array<{
  name: string;
  files: ProjectFile[];
}> {
  const projectDir = path.join(projectsPath, projectName);
  if (!fs.existsSync(projectDir)) {
    throw new Error("Project not found");
  }
  const animalDir = path.join(projectDir, animalName);
  if (!fs.existsSync(animalDir)) {
    throw new Error("Animal not found");
  }
  const animalDataFiles: Array<{
    name: string;
    files: ProjectFile[];
  }> = [];
  // get all sub directories in the animal directory
  const dataTypes = fs
    .readdirSync(animalDir)
    .filter((file) => fs.statSync(path.join(animalDir, file)).isDirectory());
  dataTypes.forEach((dataType) => {
    const dataTypeDir = path.join(animalDir, dataType);
    if (!fs.existsSync(dataTypeDir)) {
      return;
    }
    const files = fs.readdirSync(dataTypeDir);
    const animalDataFilesInDataTypeDir: ProjectFile[] = files
      .filter((file) => file.endsWith(".json"))
      .map((file) => {
        const filePath = path.join(dataTypeDir, file);
        const fileContent = fs.readFileSync(filePath, "utf8");
        return JSON.parse(fileContent);
      });
    animalDataFiles.push({
      name: dataType,
      files: animalDataFilesInDataTypeDir,
    });
  });
  return animalDataFiles;
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

export function deleteAnimal(projectName: string, animalName: string): void {
  const projectDir = path.join(projectsPath, projectName);
  if (!fs.existsSync(projectDir)) {
    throw new Error("Project not found");
  }

  const metadataPath = path.join(projectDir, "metadata.json");
  const metadata: ProjectMetadata = JSON.parse(
    fs.readFileSync(metadataPath).toString()
  );

  delete metadata.animals[animalName];
  metadata.lastModified = new Date().toISOString();

  fs.writeFileSync(metadataPath, JSON.stringify(metadata, null, 2));

  // Delete the animal folder
  fs.rmSync(path.join(projectDir, animalName), { recursive: true });
}

export function getProjects(): string[] {
  // Get all directories in the projects folder
  return fs.readdirSync(projectsPath).filter((file) => {
    return fs.statSync(path.join(projectsPath, file)).isDirectory();
  });
}

export function selectDirectory(): string {
  // Select a directory on the local machine with a dialog
  const filePaths = dialog.showOpenDialogSync({
    properties: ["openDirectory"],
    title: "Select a directory",
  });
  return filePaths[0];
}

/**
 * Creates a symlink for the given target path.
 * @param {string} targetPath - The target file path.
 * @param {string} symlinkPath - The symlink path.
 * @returns {Promise<void>}
 */
function createSymlink(targetPath, symlinkPath): Promise<void> {
  return new Promise((resolve, reject) => {
    fs.symlink(targetPath, symlinkPath, (err) => {
      if (err) reject(err);
      else resolve();
    });
  });
}

export function deleteAnimalDataDirectory(
  projectName: string,
  animalName: string,
  dirname: string
): void {
  const projectDir = path.join(projectsPath, projectName);
  if (!fs.existsSync(projectDir)) {
    throw new Error("Project not found");
  }
  // Check if dirname is a full path
  if (!dirname.startsWith(projectDir)) {
    // use basename
    dirname = path.basename(dirname);
  }
  fs.rmSync(path.join(projectDir, animalName, dirname), { recursive: true });
}

export async function getAnimalDataDirectory(
  projectName: string,
  animalName: string,
  dataType: string
): Promise<string> {
  const projectDir = path.join(projectsPath, projectName);
  if (!fs.existsSync(projectDir)) {
    throw new Error("Project not found");
  }
  // Get all file objs
  return path.join(projectDir, animalName, dataType);
}
