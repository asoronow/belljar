import axios, { AxiosResponse } from "axios";
import { BrowserWindow, ipcMain } from "electron";
import fs from "fs";
import { extract, r } from "tar";
import os from "os";
import { PythonShell, Options } from "python-shell";
import { exec } from "child_process";
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

const MANIFEST = {
  models: "v952",
  nrrd: "v91",
};

function checkManifest(directoryPath: string): Promise<string[]> {
  return new Promise((resolve, reject) => {
    const manifestPath = `${directoryPath}/manifest.json`;

    // check if manufest file exists
    if (!fs.existsSync(manifestPath)) {
      // If not, all files are outdated, and create the manifest file
      fs.writeFile(manifestPath, JSON.stringify(MANIFEST), (error) => {
        if (error) {
          console.error("Error writing manifest file:", error);
        }
      });
      resolve(Object.keys(MANIFEST));
    }

    fs.readFile(manifestPath, "utf8", (error, data) => {
      if (error) {
        // Handle error reading manifest file
        console.error("Error reading manifest file:", error);
        reject(error);
        return;
      }
      const manifestData = JSON.parse(data);
      const outdatedItems: string[] = [];

      Object.keys(MANIFEST).forEach((item) => {
        const version = MANIFEST[item];

        // Check if directory exists
        if (!fs.existsSync(`${directoryPath}/${item}`)) {
          outdatedItems.push(item);
        }

        // Check if version is outdated
        if (manifestData[item] !== version) {
          outdatedItems.push(item);
        }
      });

      // overwrite manifest file
      fs.writeFile(manifestPath, JSON.stringify(MANIFEST), (error) => {
        if (error) {
          console.error("Error writing manifest file:", error);
        }
      });
      resolve(outdatedItems);
    });
  });
}

async function downloadFromManifest(directoryPath: string): Promise<void> {
  try {
    const outdatedItems = await checkManifest(directoryPath);

    const downloadPromises = outdatedItems.map(async (item) => {
      const version = MANIFEST[item];

      // Download the file
      const url = objectPath(item, version);
      const filename = `${item}.tar.gz`;
      await downloadFile(url, directoryPath, filename);

      // Extract the tarball
      const tarballPath = `${directoryPath}/${filename}`;
      await extractTarball(tarballPath, directoryPath);

      // Delete the tarball
      await deleteFile(tarballPath);
    });

    await Promise.all(downloadPromises);
  } catch (error) {
    console.error("Error downloading from manifest:", error);
  }
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
    })
      .then(resolve)
      .catch(reject);
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

async function setupPython(
  window: BrowserWindow,
  pythonScriptsPath: string
): Promise<void> {
  return new Promise((resolve, reject) => {
    const platform = os.platform();
    const architecture = os.arch();
    let pythonVersion = pythonVersions[platform];

    if (platform === "darwin" && architecture === "arm64") {
      pythonVersion = pythonVersions["darwin-arm64"];
    }

    // check if python folder exists
    if (fs.existsSync(`${belljarFolder}/benv`)) {
      // try and install dependencies to make sure all packages are installed
      window.webContents.send("setup-progress", "Updating Python packages...");
      const requirementsPath = `${pythonScriptsPath}/requirements.txt`;
      const envPyCommand =
        platform === "win32"
          ? `${belljarFolder}/benv/Scripts/python`
          : `${belljarFolder}/benv/bin/python`;
      exec(`${envPyCommand} -m pip install -r ${requirementsPath}`, (error) => {
        if (error) {
          console.error("Error installing Python packages:", error);
        } else {
          resolve();
        }
      });
      return;
    }

    // Download Python tarball
    const pythonTarballPath = `${belljarFolder}/python.tar.gz`;
    downloadFile(pythonVersion, belljarFolder, "python.tar.gz").then(() => {
      // Extract Python tarball
      window.webContents.send("setup-progress", "Extracting Python tarball...");
      extractTarball(pythonTarballPath, belljarFolder).then(() => {
        // Delete Python tarball
        deleteFile(pythonTarballPath).then(() => {
          window.webContents.send(
            "setup-progress",
            "Creating virtual environment..."
          );
          // Create Python virtual environment
          const pyCommand = platform === "win32" ? "python" : "python3";
          exec(`${pyCommand} -m venv ${belljarFolder}/benv`, (error) => {
            if (error) {
              console.error(
                "Error creating Python virtual environment:",
                error
              );
            } else {
              // Install Python packages
              window.webContents.send(
                "setup-progress",
                "Installing Python packages..."
              );
              const requirementsPath = `${pythonScriptsPath}/requirements.txt`;
              const envPyCommand =
                platform === "win32"
                  ? `${belljarFolder}/benv/Scripts/python`
                  : `${belljarFolder}/benv/bin/python`;
              exec(
                `${envPyCommand} -m pip install -r ${requirementsPath}`,
                (error) => {
                  if (error) {
                    console.error("Error installing Python packages:", error);
                  } else {
                    resolve();
                  }
                }
              );
            }
          });
        });
      });
    });
  });
}

export function performSetup(
  window: BrowserWindow,
  pythonScriptsPath: string
): void {
  // Create Bell Jar folder
  setupPython(window, pythonScriptsPath).then(() => {
    downloadFromManifest(belljarFolder)
      .then(() => {
        window.webContents.send("setup-progress", "Setup complete!");
      })
      .catch((error) => {
        console.error("Error performing setup:", error);
      });
  });
}
