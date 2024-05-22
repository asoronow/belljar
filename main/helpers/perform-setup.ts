import axios, { AxiosResponse } from "axios";
import { BrowserWindow, ipcMain } from "electron";
import fs from "fs";
import { extract } from "tar";
import os from "os";
import { PythonShell, Options } from "python-shell";
// User folder
const homeFolder = os.homedir();
// Bell Jar folder
const belljarFolder = `${homeFolder}/.belljar`;
// Google Cloud Storage bucket
const bucket = "https://storage.googleapis.com/belljar_updates";
// Helper function to generate object path
const objectPath = (item: string, version: string) =>
  `${bucket}/${item}-${version}.tar.gz`;
// Manifest file
const manifest = {
  models: "v952",
  nrrd: "v91",
};

function checkManifest(directoryPath: string, callback: () => void): void {
  const manifestPath = `${directoryPath}/manifest.json`;

  fs.readFile(manifestPath, "utf8", (error, data) => {
    if (error) {
      // Handle error reading manifest file
      console.error("Error reading manifest file:", error);
      return;
    }

    try {
      const manifestData = JSON.parse(data);

      if (
        manifestData.models !== manifest.models ||
        manifestData.nrrd !== manifest.nrrd
      ) {
        // Manifest does not match current manifest
        callback();

        // Replace the manifest file with the current manifest
        fs.writeFile(manifestPath, JSON.stringify(manifest), (error) => {
          if (error) {
            console.error("Error writing manifest file:", error);
          } else {
            console.log("Manifest file updated");
          }
        });
      } else {
        // Manifest matches current manifest
        console.log("Manifest is up to date");
      }
    } catch (error) {
      // Handle error parsing manifest file
      console.error("Error parsing manifest file:", error);
    }
  });
}

async function downloadFile(
  url: string,
  path: string,
  filename: string
): Promise<void> {
  const response: AxiosResponse = await axios({
    url,
    method: "GET",
    responseType: "stream",
  });

  const totalLength: number = parseInt(response.headers["content-length"], 10);
  let downloadedLength: number = 0;

  const writer = fs.createWriteStream(`${path}/${filename}`);

  response.data.on("data", (chunk: Buffer) => {
    downloadedLength += chunk.length;
    const progress = (downloadedLength / totalLength) * 100;
    ipcMain.emit("download-progress", progress);
  });

  response.data.pipe(writer);

  return new Promise((resolve, reject) => {
    writer.on("finish", resolve);
    writer.on("error", reject);
  });
}

async function deleteFile(filePath: string): Promise<void> {
  return new Promise((resolve, reject) => {
    fs.unlink(filePath, (error) => {
      if (error) {
        reject(error);
      } else {
        resolve();
      }
    });
  });
}

async function extractTarball(
  filePath: string,
  destinationPath: string
): Promise<void> {
  return new Promise((resolve, reject) => {
    extract({
      file: filePath,
      cwd: destinationPath,
    });
  });
}

function createDirectoryInHomeFolder(
  directoryName: string,
  subdirs: string[]
): void {
  const directoryPath = `${homeFolder}/${directoryName}`;

  if (!fs.existsSync(directoryPath)) {
    fs.mkdirSync(directoryPath, { recursive: true });
  }

  subdirs.forEach((subdir) => {
    let localPath = `${directoryPath}/${subdir}`;
    if (!fs.existsSync(localPath)) {
      fs.mkdirSync(localPath, { recursive: true });
    }
  });
}

const pythonVersions = {
  win32: `${bucket}/cpython-3.10.13%2B20230826-x86_64-pc-windows-msvc-shared-install_only.tar.gz`,
  darwin: `${bucket}/cpython-3.10.13%2B20230826-x86_64-apple-darwin-install_only.tar.gz`,
  "darwin-arm64": `${bucket}/cpython-3.10.13%2B20230826-aarch64-apple-darwin-install_only.tar.gz`,
  linux: `${bucket}/cpython-3.10.13%2B20230826-x86_64-unknown-linux-gnu-install_only.tar.gz`,
};

async function setupPython(): Promise<void> {
  // check if python folder exists
  if (fs.existsSync(`${belljarFolder}/benv`)) {
    return;
  }
  const platform = os.platform();
  const architecture = os.arch();
  let pythonVersion = pythonVersions[platform];

  if (platform === "darwin" && architecture === "arm64") {
    pythonVersion = pythonVersions["darwin-arm64"];
  }

  // Download Python tarball
  const pythonTarballPath = `${belljarFolder}/python.tar.gz`;
  await downloadFile(pythonVersion, belljarFolder, "python.tar.gz");

  // Extract Python tarball
  await extractTarball(pythonTarballPath, belljarFolder);

  // Configure virtual environment
  const virtualEnvPath = `${belljarFolder}/benv`;
  let options: Options = {
    mode: "text",
    pythonPath: `${belljarFolder}/python`,
  };
  PythonShell.runString(`python -m venv ../benv`, options).then((messages) => {
    console.log("Virtual environment created");
  });
}

export function performSetup(window: BrowserWindow): void {
  // Create Bell Jar folder
  window.webContents.send("setup-progress", "Creating Bell Jar folder...");
  createDirectoryInHomeFolder(".belljar", ["models", "nrrd"]);

  setupPython();
}
