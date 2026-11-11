const { app, BrowserWindow, ipcMain, dialog, shell } = require("electron");
const { promisify } = require("util");
const { PythonShell } = require("python-shell");
const path = require("path");
const fs = require("fs");
const tar = require("tar");
const mv = promisify(fs.rename);
const exec = promisify(require("child_process").exec);
const stream = require("stream");
const https = require("https");
const semver = require("semver");
const serverFetch = require("node-fetch");
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
  parseJsonObject,
  requirementsNeedInstall,
  resourcesToDownload,
  sharpenArgs,
} from "./launch";

var appDir = app.getAppPath();

var win: typeof BrowserWindow = null;
var logWin: typeof BrowserWindow = null;
var isQuitting = false;
var log = console.log;
console.log = function () {
  var args = Array.from(arguments);
  let timestamp = new Date().toLocaleString();
  let prefix = `[${timestamp}]`;

  let message = [prefix, ...args];

  log.apply(console, message);
  try {
    logWin.webContents.send("log", message.join(" "));
  } catch (error) {
    // do nothing window was closed
  }
};

// Path variables for easy management of execution
const homeDir = path.join(app.getPath("home"), ".belljar");
// Mod is the proper path to the python/pip binary
var mod = process.platform === "win32" ? "python/" : "python/bin/";
var envMod = process.platform === "win32" ? "Scripts/" : "bin/";
// Make a constant with the cwd for running python commands
const envPath = path.join(homeDir, "benv");
const pythonPath = path.join(homeDir, mod);
const envPythonPath = path.join(envPath, envMod);
// Command choses wether to use the exe (windows) or alias (unix based)
var pyCommand = process.platform === "win32" ? "python.exe" : "./python3";
// Path to our python files
const pyScriptsPath = path.join(appDir, "/py");

const CURRENT_VERSION_TAG = getVersion();
const GITHUB_API_RELEASES =
  "https://api.github.com/repos/asoronow/belljar/releases/latest";

// Runs after the window is up and never blocks launch: an unreachable
// network only produces a log line.
async function checkForUpdates() {
  try {
    const response = await serverFetch(GITHUB_API_RELEASES);
    if (!response.ok) {
      console.log(`Update check skipped: GitHub API status ${response.status}`);
      return;
    }

    const data = await response.json();
    const latestVersionTag = data.tag_name;

    if (
      semver.valid(latestVersionTag) &&
      semver.gt(latestVersionTag, CURRENT_VERSION_TAG)
    ) {
      const userResponse = await dialog.showMessageBox({
        type: "info",
        title: "Update Available",
        message: "A new version of the application is available.",
        detail: `The latest version is ${latestVersionTag}. Would you like to download it?`,
        buttons: ["Yes", "No"],
        defaultId: 0,
        cancelId: 1,
      });

      if (userResponse.response === 0) {
        shell.openExternal(data.html_url); // URL to the latest release page
      }
    } else {
      console.log("No updates available.");
    }
  } catch (error) {
    console.log("Update check skipped:", error);
  }
}

// Promise version of file moving
function move(o: string, t: string) {
  return new Promise((resolve, reject) => {
    // move o to t, wrapped as promise
    const original = o;
    const target = t;
    mv(original, target).then(() => {
      resolve(0);
    });
  });
}

function createLogFile(message: string) {
  const logPath = path.join(homeDir, "belljar.log");
  fs.appendFileSync(logPath, message);
}

// Get files asynchonously
function downloadFile(url: string, target: string, win: typeof BrowserWindow) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(target, { highWaterMark: 64 * 1024 });
    // get the file, update the user loading screen with text on progress

    const progress = (receivedBytes: number, totalBytes: number) => {
      const percentage = (receivedBytes * 100) / totalBytes;
      if (percentage > 0) {
        win.webContents.send("updateStatus", {
          message: `Downloading ${target
            .split("/")
            .pop()}... ${percentage.toFixed(0)}%`,
          timestamp: Date.now(),
        });
      }
    };
    const dummy = new stream.PassThrough();
    const failed = (error: Error) => {
      file.close();
      fs.rm(target, { force: true }, () => reject(error));
    };
    const request = https.get(url, (response: any) => {
      if (response.statusCode !== 200) {
        response.resume();
        failed(new Error(`Download of ${url} failed: HTTP ${response.statusCode}`));
        return;
      }
      response.on("error", failed);
      response.on("aborted", () =>
        failed(new Error(`Download of ${url} was aborted`))
      );
      // create a dummy stream so we can update the user on progress
      var receivedBytes = 0;
      var totalBytes = parseInt(response.headers["content-length"]);
      response.pipe(dummy);
      let lastUpdateTimestamp = Date.now();

      dummy.on("data", (chunk: any) => {
        receivedBytes += chunk.length;
        const currentTimestamp = Date.now();
        if (currentTimestamp - lastUpdateTimestamp >= 1000) {
          // 1000 ms = 1 second
          progress(receivedBytes, totalBytes);
          lastUpdateTimestamp = currentTimestamp;
        }
      });
      // pipe the response to the file
      response.pipe(file);
      file.on("finish", () => {
        file.close();
        win.webContents.send(
          "updateStatus",
          `Extracting ${target.split("/").pop()}...`
        );
        resolve(true);
      });
    });
    request.on("error", failed);
  });
}

// Delete a file safely
function deleteFile(file: string) {
  return new Promise((resolve, reject) => {
    fs.unlinkSync(file);
    resolve(true);
  });
}

function getVersion() {
  // get version from package.json
  const packageJson = require(path.join(appDir, "package.json"));
  return packageJson.version;
}

function setupPython(win: typeof BrowserWindow) {
  const linuxURL = `${bucketParentPath}/cpython-3.10.13+20230826-x86_64-unknown-linux-gnu-install_only.tar.gz`;
  const winURL = `${bucketParentPath}/cpython-3.10.13+20230826-x86_64-pc-windows-msvc-shared-install_only.tar.gz`;
  const osxURL = `${bucketParentPath}/cpython-3.10.13+20230826-aarch64-apple-darwin-install_only.tar.gz`;
  const osxIntelURL = `${bucketParentPath}/cpython-3.10.13+20230826-x86_64-apple-darwin-install_only.tar.gz`;
  return new Promise((resolve, reject) => {
    if (!fs.existsSync(path.join(homeDir, "python"))) {
      win.webContents.send("updateStatus", "Settting up python...");
      switch (process.platform) {
        case "win32":
          // Download and extract python to the home directory
          downloadFile(
            winURL,
            path.join(
              homeDir,
              "cpython-3.10.13+20230826-x86_64-pc-windows-msvc-shared-install_only.tar.gz"
            ),
            win
          )
            .then(() => {
              // Extract the tarball
              tar
                .x({
                  cwd: homeDir,
                  preservePaths: true,
                  file: path.join(
                    homeDir,
                    "cpython-3.10.13+20230826-x86_64-pc-windows-msvc-shared-install_only.tar.gz"
                  ),
                })
                .then(() => {
                  win.webContents.send("updateStatus", "Extracted python...");
                  resolve(true);
                });
            })
            .catch((err: any) => {
              console.log(err);
            });
          break;
        case "linux":
          downloadFile(
            linuxURL,
            path.join(
              homeDir,
              "cpython-3.10.13+20230826-x86_64-unknown-linux-gnu-install_only.tar.gz"
            ),
            win
          ).then(() => {
            tar
              .x({
                cwd: homeDir,
                preservePaths: true,
                file: path.join(
                  homeDir,
                  "cpython-3.10.13+20230826-x86_64-unknown-linux-gnu-install_only.tar.gz"
                ),
              })
              .then(() => {
                win.webContents.send("updateStatus", "Extracted python...");
                resolve(true);
              });
          });
          break;
        case "darwin":
          // Check if we are on intel or arm
          if (process.arch === "x64") {
            downloadFile(
              osxIntelURL,
              path.join(
                homeDir,
                "cpython-3.10.13+20230826-x86_64-apple-darwin-install_only.tar.gz"
              ),
              win
            ).then(() => {
              tar
                .x({
                  cwd: homeDir,
                  preservePaths: true,
                  file: path.join(
                    homeDir,
                    "cpython-3.10.13+20230826-x86_64-apple-darwin-install_only.tar.gz"
                  ),
                })
                .then(() => {
                  win.webContents.send("updateStatus", "Extracted python...");
                  resolve(true);
                });
            });
          } else {
            downloadFile(
              osxURL,
              path.join(
                homeDir,
                "cpython-3.10.13+20230826-aarch64-apple-darwin-install_only.tar.gz"
              ),
              win
            ).then(() => {
              tar
                .x({
                  cwd: homeDir,
                  preservePaths: true,
                  file: path.join(
                    homeDir,
                    "cpython-3.10.13+20230826-aarch64-apple-darwin-install_only.tar.gz"
                  ),
                })
                .then(() => {
                  win.webContents.send("updateStatus", "Extracted python...");
                  resolve(true);
                });
            });
          }
          break;
        default:
          // If we don't have a supported platform, just resolve
          resolve(true);
          break;
      }
    } else {
      // Double check that the environment is setup by confirming if the benv folder exists
      if (!fs.existsSync(envPath)) {
        resolve(true);
      } else {
        resolve(false);
      }
    }
  });
}

const bucketParentPath = "https://storage.googleapis.com/belljar_updates";
const manifestPath = path.join(homeDir, "manifest.json");

// A resource directory counts as present only when it is non-empty and, for
// models, holds the detector weights that an interrupted extraction leaves out.
function resourcePresent(key: string): boolean {
  const dir = path.join(homeDir, key);
  if (!fs.existsSync(dir) || fs.readdirSync(dir).length === 0) {
    return false;
  }
  if (key === "models") {
    return fs.existsSync(path.join(dir, "chaosdruid.pt"));
  }
  return true;
}

// Downloads every resource whose directory is missing or whose manifest
// version differs from RESOURCES. The manifest entry for a key is written
// only after that key's archive has been downloaded, extracted and removed,
// so an interrupted download is retried on the next launch.
async function downloadResources(win: typeof BrowserWindow) {
  const manifest = readJsonFile(manifestPath);
  const present: Record<string, boolean> = {};
  for (const key of Object.keys(RESOURCES)) {
    present[key] = resourcePresent(key);
  }

  for (const key of resourcesToDownload(manifest, RESOURCES, present)) {
    const resource = RESOURCES[key];
    const archivePath = path.join(homeDir, `${key}.tar.gz`);
    win.webContents.send(
      "updateStatus",
      `Downloading ${key}...this may take a while`
    );
    await fs.promises.rm(path.join(homeDir, key), {
      recursive: true,
      force: true,
    });
    await downloadFile(
      `${bucketParentPath}/${resource.filename}`,
      archivePath,
      win
    );
    await tar.x({ cwd: homeDir, preservePaths: true, file: archivePath });
    await deleteFile(archivePath);

    manifest[key] = resource.version;
    try {
      fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
    } catch (error) {
      console.log(`Could not record ${key} in the manifest:`, error);
    }
    win.webContents.send("updateStatus", `Downloaded ${key}`);
  }
  return true;
}

// Creates the venv and installs the dependencies
function setupEnvironment(win: typeof BrowserWindow) {
  if (!fs.existsSync(envPath)) {
    // We have not created the venv yet, so we probably don't have the models, etc. either

    win.webContents.send(
      "updateStatus",
      "Preparing to download require files..."
    );

    downloadResources(win)
      .then(() => {
        win.webContents.send("updateStatus", "Installing venv...");
        return installVenv();
      })
      .then(({ stdout, stderr }) => {
        console.log(stdout);
        win.webContents.send("updateStatus", "Creating venv...");
        return createVenv();
      })
      .then(({ stdout, stderr }) => {
        console.log(stdout);
        win.webContents.send("updateStatus", "Installing packages...");
        return installDeps();
      })
      .then(({ stdout, stderr }) => {
        console.log(stdout);
        recordInstalledRequirements();
        win.webContents.send("updateStatus", "Setup complete!");
        win.loadFile("pages/index.html");
      })
      .catch((error) => {
        console.log("An error occurred during setup:", error);
        win.webContents.send("updateStatus", "An error occurred during setup.");
      });
  }

  // Install venv package
  async function installVenv() {
    const { stdout, stderr } = await exec(
      `${pyCommand} -m pip install --user virtualenv`,
      { cwd: pythonPath }
    );
    return { stdout, stderr };
  }

  // Create venv
  async function createVenv() {
    const envDir = process.platform === "win32" ? "../benv" : "../../benv";
    const { stdout, stderr } = await exec(`${pyCommand} -m venv ${envDir}`, {
      cwd: pythonPath,
    });
    return { stdout, stderr };
  }

  // Install pip packages
  async function installDeps() {
    let reqs = path.join(appDir, "py/requirements.txt");
    const { stdout, stderr } = await exec(
      `${pyCommand} -m pip install -r "${reqs}" --use-pep517`,
      { cwd: envPythonPath }
    );
    return { stdout, stderr };
  }
}

function readJsonFile(filePath: string): Record<string, unknown> {
  if (!fs.existsSync(filePath)) {
    return {};
  }
  return parseJsonObject(fs.readFileSync(filePath, "utf8"));
}

const requirementsPath = path.join(appDir, "py/requirements.txt");
const statePath = path.join(homeDir, "state.json");

function recordInstalledRequirements() {
  try {
    const requirementsHash = hashRequirements(fs.readFileSync(requirementsPath));
    const state = readJsonFile(statePath);
    fs.writeFileSync(
      statePath,
      JSON.stringify({ ...state, requirementsHash }, null, 2)
    );
  } catch (error) {
    console.log("Could not record the installed requirements:", error);
  }
}

// Reinstalls the python packages only when py/requirements.txt has changed
// since the last successful install, recorded in ~/.belljar/state.json.
async function updatePythonDependencies(win: typeof BrowserWindow) {
  const requirementsHash = hashRequirements(fs.readFileSync(requirementsPath));
  const state = readJsonFile(statePath);

  if (!requirementsNeedInstall(requirementsHash, state)) {
    return true;
  }

  win.webContents.send("updateStatus", "Updating packages...");
  try {
    const { stdout } = await exec(
      `${pyCommand} -m pip install -r "${requirementsPath}" --no-cache-dir  --use-pep517`,
      { cwd: envPythonPath }
    );
    console.log(stdout);
  } catch (error) {
    console.log(error);
    createLogFile(String(error));
    createLogFile("Failed to update python dependencies");
    createLogFile(appDir);
    if (hasInstalledRequirements(state)) {
      console.log("Continuing with the previously installed python packages");
      return true;
    }
    throw error;
  }

  recordInstalledRequirements();
  win.webContents.send("updateStatus", "Update complete!");
  return true;
}

// Ensure all required directories exist and if not, download them
function fixMissingDirectories(win: typeof BrowserWindow) {
  win.webContents.send("updateStatus", "Checking for updates...");
  return downloadResources(win);
}

// Makes the local user writable folder
function checkLocalDir() {
  if (!fs.existsSync(homeDir)) {
    fs.mkdirSync(homeDir, {
      recursive: true,
    });
  }
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1250,
    height: 750,
    resizable: true,
    autoHideMenuBar: true,
    webPreferences: { nodeIntegration: true, contextIsolation: false },
  });

  // Start with the load screen
  win.loadFile("pages/loading.html");

  return win;
}

function createLogWindow() {
  const logWin = new BrowserWindow({
    width: 500,
    height: 250,
    resizable: true,
    autoHideMenuBar: true,
    webPreferences: { nodeIntegration: true, contextIsolation: false },
    closeable: false,
  });

  logWin.loadFile("pages/log.html");

  return logWin;
}

app.on("ready", () => {
  win = createWindow();
  logWin = createLogWindow();
  // Uncomment if you want tools on launch
  // win.webContents.toggleDevTools()
  win.on("close", function (e: any) {
    const choice = dialog.showMessageBoxSync(win, {
      type: "question",
      buttons: ["Yes", "Cancel"],
      title: "Confrim Quit",
      message:
        "Are you sure you want to quit? Quitting will kill all running processes.",
    });
    if (choice === 1) {
      e.preventDefault();
    } else {
      try {
        logWin.webContents.send("savelogs", []);
        logWin.close();
      } catch (error) {
        // do nothing window was closed
      }
    }
  });

  win.webContents.once("did-finish-load", () => {
    void checkForUpdates();
    // Make a directory to house enviornment, settings, etc.yarn
    checkLocalDir();
    // Setup python for running the pipeline
    setupPython(win)
      .then((installed) => {
        // If we just installed python, we need to continue the complete
        // setup of the enviornment
        if (installed) {
          setupEnvironment(win);
        } else {
          // Otherwise, we can just update the dependencies
          return updatePythonDependencies(win).then(() => {
            // Check if any directories are missing
            return fixMissingDirectories(win).then(() => {
              win.loadFile("pages/index.html");
            });
          });
        }
      })
      .catch((error) => {
        console.log(error);
        win.webContents.send("updateStatus", `Setup failed: ${error}`);
      });
  });
});

app.whenReady().then(() => {
  app.on("activate", function () {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", function () {
  app.quit();
});

ipcMain.on("getVersion", (event: any) => {
  event.sender.send("version", getVersion());
});

// Handlers
// Electron opens every dialog in Downloads unless given a defaultPath, so the
// folder of the last selection is carried over to the next dialog.
let lastDialogDir: string | undefined;

// Directories
ipcMain.on("openDialog", function (event: any, data: any) {
  let window = BrowserWindow.getFocusedWindow();
  dialog
    .showOpenDialog(window, {
      properties: ["openDirectory"],
      defaultPath: lastDialogDir,
    })
    .then((result: { canceled: boolean; filePaths: any[] }) => {
      // Check for a valid result
      if (!result.canceled) {
        lastDialogDir = path.dirname(result.filePaths[0]);
        // console.log(result.filePaths)
        // Send back the dir and whether this is input or output
        event.sender.send("returnPath", [result.filePaths[0], data]);
      }
    })
    .catch((err: Error) => {
      console.log(err);
    });
});
// Files
ipcMain.on("openFileDialog", function (event: any, data: any) {
  let window = BrowserWindow.getFocusedWindow();
  dialog
    .showOpenDialog(window, {
      properties: ["openFile"],
      defaultPath: lastDialogDir,
    })
    .then((result: { canceled: boolean; filePaths: any[] }) => {
      // Check for a valid result
      if (!result.canceled) {
        lastDialogDir = path.dirname(result.filePaths[0]);
        // console.log(result.filePaths)
        // Send back the dir and whether this is input or output
        event.sender.send("returnPath", [result.filePaths[0], data]);
      }
    })
    .catch((err: Error) => {
      console.log(err);
    });
});

function openPDF(relativePath: string) {
  const pdfPath = path.join(appDir, relativePath);
  shell
    .openPath(pdfPath)
    .then(() => {
      console.log("Guide opened");
    })
    .catch((error: any) => {
      console.log(error);
    });
}

ipcMain.on("openGuide", function (event: any, data: any) {
  openPDF("docs/belljar_guide.pdf");
});

// Max Projection
ipcMain.on("runMax", function (event: any, data: any[]) {
  let options = {
    mode: "text",
    pythonPath: path.join(envPythonPath, pyCommand),
    scriptPath: pyScriptsPath,
    args: maxArgs({ input: data[0], output: data[1] }),
  };

  let pyshell = new PythonShell("max.py", options);
  var total: number = 0;
  var current: number = 0;
  pyshell.on("message", (message: string) => {
    if (total === 0) {
      total = Number(message);
    } else if (message == "Done!") {
      pyshell.end((err: string, code: any, signal: string) => {
        if (err) throw err;
        console.log("The exit code was: " + code);
        console.log("The exit signal was: " + signal);
        event.sender.send("maxResult");
        ipcMain.removeAllListeners("killMax");
      });
    } else {
      current++;
      console.log(message);
      event.sender.send("updateLoad", [
        Math.round((current / total) * 100),
        message,
      ]);
    }
  });

  ipcMain.once("killMax", function (event: any, data: any[]) {
    pyshell.kill();
  });
});

// Adjust
ipcMain.on("runAdjust", function (event: any, data: any[]) {
  var structPath = path.join(appDir, "csv/structure_map.pkl");

  let options = {
    mode: "text",
    pythonPath: path.join(envPythonPath, pyCommand),
    scriptPath: pyScriptsPath,
    args: adjustArgs({
      images: data[0],
      annotations: data[1],
      structures: structPath,
    }),
  };

  let pyshell = new PythonShell("adjust.py", options);
  var total: number = 0;
  var current: number = 0;
  pyshell.on("stderr", function (stderr: string) {
    console.log(stderr);
  });
  pyshell.on("message", (message: string) => {
    if (total === 0) {
      total = Number(message);
    } else if (message == "Done!") {
      pyshell.end((err: string, code: any, signal: string) => {
        if (err) throw err;
        console.log("The exit code was: " + code);
        console.log("The exit signal was: " + signal);
        event.sender.send("adjustResult");
        ipcMain.removeAllListeners("killAdjust");
      });
    } else {
      current++;
      console.log(message);
      event.sender.send("updateLoad", [
        Math.round((current / total) * 100),
        message,
      ]);
    }
  });
  ipcMain.once("killAdjust", function (event: any, data: any[]) {
    pyshell.kill();
  });
});

// Alignment
ipcMain.on("runAlign", function (event: any, data: any[]) {
  const modelPath = path.join(homeDir, "models/predictor.pt");
  const nrrdPath = path.join(homeDir, "nrrd");
  const mapPath = path.join(appDir, "csv/structure_map.pkl");

  let options = {
    mode: "text",
    pythonPath: path.join(envPythonPath, pyCommand),
    scriptPath: pyScriptsPath,
    args: alignArgs({
      input: data[0],
      output: data[1],
      whole: data[2],
      spacing: data[3],
      legacy: data[4],
      model: modelPath,
      nrrd: nrrdPath,
      map: mapPath,
    }),
  };
  let pyshell = new PythonShell("map.py", options);
  var total: number = 0;
  var current: number = 0;

  pyshell.on("stderr", function (stderr: string) {
    console.log(stderr);
  });

  pyshell.on("message", (message: string) => {
    console.log(message);
    if (total === 0) {
      total = Number(message);
    } else if (message == "Done!") {
      pyshell.end((err: string, code: any, signal: string) => {
        if (err) throw err;
        event.sender.send("alignResult");
        console.log("The exit code was: " + code);
        console.log("The exit signal was: " + signal);
        ipcMain.removeAllListeners("killAlign");
      });
    } else {
      current++;
      event.sender.send("updateLoad", [
        Math.round((current / total) * 100),
        message,
      ]);
    }
  });

  ipcMain.once("killAlign", function (event: any, data: any[]) {
    pyshell.kill();
  });
});

// Intensity by Region

ipcMain.on("runIntensity", function (event: any, data: any[]) {
  const structPath = path.join(appDir, "csv/structure_map.pkl");

  let options = {
    mode: "text",
    pythonPath: path.join(envPythonPath, pyCommand),
    scriptPath: pyScriptsPath,
    args: intensityArgs({
      images: data[0],
      output: data[1],
      annotations: data[2],
      map: structPath,
      whole: data[3],
    }),
  };

  let pyshell = new PythonShell("region.py", options);
  var total: number = 0;
  var current: number = 0;
  pyshell.on("stderr", function (stderr: string) {
    console.log(stderr);
  });
  pyshell.on("message", (message: string) => {
    console.log(message);
    if (total === 0) {
      total = Number(message);
    } else if (message == "Done!") {
      pyshell.end((err: string, code: any, signal: string) => {
        if (err) throw err;
        console.log("The exit code was: " + code);
        console.log("The exit signal was: " + signal);
        event.sender.send("intensityResult");
        ipcMain.removeAllListeners("killIntensity");
      });
    } else {
      current++;
      event.sender.send("updateLoad", [
        Math.round((current / total) * 100),
        message,
      ]);
    }
  });

  ipcMain.once("killIntensity", function (event: any, data: any[]) {
    pyshell.kill();
  });
});

// Counting
ipcMain.on("runCount", function (event: any, data: any[]) {
  var structPath = path.join(appDir, "csv/structure_map.pkl");

  let options = {
    mode: "text",
    pythonPath: path.join(envPythonPath, pyCommand),
    scriptPath: pyScriptsPath,
    args: countArgs({
      predictions: data[0],
      annotations: data[1],
      output: data[2],
      structures: structPath,
      layers: data[3],
    }),
  };
  let pyshell = new PythonShell("count.py", options);
  var total: number = 0;
  var current: number = 0;

  pyshell.on("stderr", function (stderr: string) {
    console.log(stderr);
  });

  pyshell.on("message", (message: string) => {
    console.log(message);
    if (total === 0) {
      total = Number(message);
    } else if (message == "Done!") {
      pyshell.end((err: string, code: any, signal: string) => {
        if (err) throw err;
        console.log("The exit code was: " + code);
        console.log("The exit signal was: " + signal);
        event.sender.send("countResult");
        ipcMain.removeAllListeners("killCount");
      });
    } else {
      current++;
      event.sender.send("updateLoad", [
        Math.round((current / total) * 100),
        message,
      ]);
    }
  });

  ipcMain.once("killCount", function (event: any, data: any[]) {
    pyshell.kill();
  });
});

// Collate
ipcMain.on("runCollate", function (event: any, data: any[]) {
  let options = {
    mode: "text",
    pythonPath: path.join(envPythonPath, pyCommand),
    scriptPath: pyScriptsPath,
    args: collateArgs({
      input: data[0],
      output: data[1],
      regions: data[2],
      structures: path.join(appDir, "csv/structure_map.pkl"),
    }),
  };
  let pyshell = new PythonShell("collate.py", options);

  pyshell.end((err: string, code: any, signal: string) => {
    if (err) throw err;
    console.log("The exit code was: " + code);
    console.log("The exit signal was: " + signal);
    event.sender.send("collateResult");
  });

  ipcMain.once("killCollate", function (event: any, data: any[]) {
    pyshell.kill();
  });
});

// Collate
ipcMain.on("runSharpen", function (event: any, data: any[]) {
  let options = {
    mode: "text",
    pythonPath: path.join(envPythonPath, pyCommand),
    scriptPath: pyScriptsPath,
    args: sharpenArgs({
      input: data[0],
      output: data[1],
      radius: data[2],
      amount: data[3],
      equalize: data[4],
    }),
  };
  let pyshell = new PythonShell("sharpen.py", options);
  var total: number = 0;
  var current: number = 0;
  pyshell.on("message", (message: string) => {
    console.log(message);
    if (total === 0) {
      total = Number(message);
    } else if (message == "Done!") {
      pyshell.end((err: string, code: any, signal: string) => {
        if (err) throw err;
        console.log("The exit code was: " + code);
        console.log("The exit signal was: " + signal);
        event.sender.send("sharpenResult");
        ipcMain.removeAllListeners("killSharpen");
      });
    } else {
      current++;
      event.sender.send("updateLoad", [
        Math.round((current / total) * 100),
        message,
      ]);
    }
  });

  ipcMain.once("killSharpen", function (event: any, data: any[]) {
    pyshell.kill();
  });
});

// Cell Detection
ipcMain.on("runDetection", function (event: any, data: any[]) {
  // Set model path
  var models: { [key: string]: string } = {
    somata: "models/chaosdruid.pt",
    nuclei: "models/ankou.pt",
  };

  var sam_model_path = path.join(homeDir, "models/sam_vit_b.pth");

  let selected = data[6] as string;
  var modelPath = path.join(homeDir, models[selected]);
  // Switch over to custom if necessary
  if (data[4].length > 0) {
    modelPath = data[4];
  }

  let options = {
    mode: "text",
    pythonPath: path.join(envPythonPath, pyCommand),
    scriptPath: pyScriptsPath,
    args: detectionArgs({
      input: data[0],
      output: data[1],
      confidence: data[2],
      tile: data[3],
      area: data[7],
      sam: sam_model_path,
      eccentricity: data[8],
      model: modelPath,
      multichannel: data[5],
    }),
  };

  let pyshell = new PythonShell("find_neurons.py", options);
  var total: number = 0;
  var current: number = 0;

  pyshell.on("stderr", function (stderr: string) {
    console.log(stderr);
  });

  pyshell.on("message", (message: string) => {
    console.log(message);
    if (total === 0) {
      total = Number(message);
    } else if (message == "Done!") {
      pyshell.end((err: string, code: any, signal: string) => {
        if (err) throw err;
        console.log("The exit code was: " + code);
        console.log("The exit signal was: " + signal);
        event.sender.send("detectResult");
        ipcMain.removeAllListeners("killDetect");
      });
    } else {
      current++;
      event.sender.send("updateLoad", [
        Math.round((current / total) * 100),
        message,
      ]);
    }
  });

  ipcMain.once("killDetect", function (event: any, data: any[]) {
    pyshell.kill();
  });
});
