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

async function checkForUpdates() {
  try {
    const response = await serverFetch(GITHUB_API_RELEASES);
    if (!response.ok) {
      throw new Error(`GitHub API response status: ${response.status}`);
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
    console.error("Failed to check for updates:", error);
    dialog.showErrorBox(
      "Update Check Failed",
      "There was an error checking for updates. Please try again later."
    );
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
    const request = https.get(url, (response: any) => {
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
  const bucketParentPath = "https://storage.googleapis.com/belljar_updates";
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

// Download the required tar files from the bucket
function downloadResources(win: typeof BrowserWindow, fresh: boolean) {
  // Download the tar files into the homeDir and extract them to their respective folders
  const currnet_versions = {
    nrrd: "v91",
    models: "v952",
    embeddings: "v6",
  };

  return new Promise((resolve, reject) => {
    const bucketParentPath = "https://storage.googleapis.com/belljar_updates";
    const embeddingsLink = `${bucketParentPath}/embeddings-v6.tar.gz`;
    const modelsLink = `${bucketParentPath}/models-v10.tar.gz`;
    const nrrdLink = `${bucketParentPath}/nrrd-v91.tar.gz`;
    const requiredDirs = ["models", "embeddings", "nrrd"];

    if (!fresh) {
      var downloading: Array<string> = [];
      var total = 0;

      // check the manifest.json and compare versions
      // if the versions are different, delete the dir and download
      const manifestPath = path.join(homeDir, "manifest.json");
      // Make sure the manifest exists and if not lets make one and then delte all these dirs and redownload
      if (!fs.existsSync(manifestPath)) {
        // Create manifest from current versions
        fs.writeFileSync(
          manifestPath,
          JSON.stringify(currnet_versions, null, 2)
        );
        // Delete existing
        downloading.push("models");
        downloading.push("embeddings");
        downloading.push("nrrd");
      }
      const manifest = require(manifestPath);

      // check if each directory exists and its not empty
      for (let i = 0; i < requiredDirs.length; i++) {
        const dir = requiredDirs[i];
        if (
          !fs.existsSync(path.join(homeDir, dir)) ||
          fs.readdirSync(path.join(homeDir, dir)).length === 0
        ) {
          // make sure we are not already downloading this dir
          if (downloading.indexOf(dir) === -1) {
            downloading.push(dir);
          }
        }
      }

      for (const [key, value] of Object.entries(currnet_versions)) {
        if (manifest[key] !== value) {
          downloading.push(key);
        }
      }

      if (downloading.indexOf("models") === -1) {
        // Check in the models dir if chaosdruid.pt exists do nothing, otherwise delete the dir and download
        if (!fs.existsSync(path.join(homeDir, "models/chaosdruid.pt"))) {
          downloading.push("models");
          // Delete existing
          if (fs.existsSync(path.join(homeDir, "models"))) {
            fs.rm(path.join(homeDir, "models"), { recursive: true });
          }
        }
      }

      // Delete and update manifest
      if (downloading.length > 0) {
        fs.writeFileSync(
          manifestPath,
          JSON.stringify(currnet_versions, null, 2)
        );
      }

      downloading.reduce((promiseChain, dir, i) => {
        return promiseChain
          .then(() => {
            win.webContents.send(
              "updateStatus",
              `Redownloading ${dir}...this may take a while`
            );

            if (fs.existsSync(path.join(homeDir, dir))) {
              fs.rmSync(path.join(homeDir, dir), { recursive: true });
            }

            let downloadPath = "";
            switch (dir) {
              case "models":
                downloadPath = modelsLink;
                break;
              case "embeddings":
                downloadPath = embeddingsLink;
                break;
              case "nrrd":
                downloadPath = nrrdLink;
                break;
              default:
                break;
            }

            return downloadFile(
              downloadPath,
              path.join(homeDir, `${dir}.tar.gz`),
              win
            );
          })
          .then(() => {
            return tar.x({
              cwd: homeDir,
              preservePaths: true,
              file: path.join(homeDir, `${dir}.tar.gz`),
            });
          })
          .then(() => {
            return deleteFile(path.join(homeDir, `${dir}.tar.gz`));
          })
          .then(() => {
            win.webContents.send("updateStatus", `Downloaded ${dir}`);
            total++;
            if (downloading.length === total) {
              resolve(true);
            }
          });
      }, Promise.resolve());

      if (downloading.length === 0) {
        resolve(true);
      }
    } else {
      // Since we are doing a fresh install, we need to ensure no remnants of the old install are left or partially downloaded
      // Check if these directories exist, if they do, we don't need to download any files
      let allDirsExist = true;
      requiredDirs.forEach((dir) => {
        if (!fs.existsSync(path.join(homeDir, dir))) {
          allDirsExist = false;
        }
      });

      // Creat the manifest
      fs.writeFileSync(
        path.join(homeDir, "manifest.json"),
        JSON.stringify(currnet_versions, null, 2)
      );

      if (!allDirsExist) {
        // Something is missing, delete everything and download again
        requiredDirs.forEach((dir) => {
          if (fs.existsSync(path.join(homeDir, dir))) {
            fs.rmSync(path.join(homeDir, dir), { recursive: true });
          }
        });

        // Download the embeddings
        downloadFile(
          embeddingsLink,
          path.join(homeDir, "embeddings.tar.gz"),
          win
        ).then(() => {
          // Extract the embeddings
          tar
            .x({
              cwd: homeDir,
              preservePaths: true,
              file: path.join(homeDir, "embeddings.tar.gz"),
            })
            .then(() => {
              // Delete the tar file
              deleteFile(path.join(homeDir, "embeddings.tar.gz")).then(() => {
                // Download the models
                downloadFile(
                  modelsLink,
                  path.join(homeDir, "models.tar.gz"),
                  win
                ).then(() => {
                  // Extract the models
                  tar
                    .x({
                      cwd: homeDir,
                      preservePaths: true,
                      file: path.join(homeDir, "models.tar.gz"),
                    })
                    .then(() => {
                      // Delete the tar file
                      deleteFile(path.join(homeDir, "models.tar.gz")).then(
                        () => {
                          // Download the nrrd
                          downloadFile(
                            nrrdLink,
                            path.join(homeDir, "nrrd.tar.gz"),
                            win
                          ).then(() => {
                            // Extract the nrrd
                            tar

                              .x({
                                cwd: homeDir,
                                preservePaths: true,
                                file: path.join(homeDir, "nrrd.tar.gz"),
                              })
                              .then(() => {
                                // Delete the tar file
                                deleteFile(
                                  path.join(homeDir, "nrrd.tar.gz")
                                ).then(() => {
                                  resolve(true);
                                });
                              });
                          });
                        }
                      );
                    });
                });
              });
            });
        });
      } else {
        resolve(true);
      }
    }
  });
}

// Creates the venv and installs the dependencies
function setupEnvironment(win: typeof BrowserWindow) {
  if (!fs.existsSync(envPath)) {
    // We have not created the venv yet, so we probably don't have the models, etc. either

    win.webContents.send(
      "updateStatus",
      "Preparing to download require files..."
    );

    downloadResources(win, true)
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

// Install the latest dependencies, could have changed after an update
function updatePythonDependencies(win: typeof BrowserWindow) {
  return new Promise((resolve, reject) => {
    win.webContents.send("updateStatus", "Updating packages...");
    // Run pip install -r requirements.txt --no-cache-dir to update the packages
    let reqsPath = path.join(appDir, "py/requirements.txt");
    exec(
      `${pyCommand} -m pip install -r "${reqsPath}" --no-cache-dir  --use-pep517`,
      { cwd: envPythonPath }
    )
      .then(({ stdout, stderr }: { stdout: string; stderr: string }) => {
        console.log(stdout);
        win.webContents.send("updateStatus", "Update complete!");
        resolve(true);
      })
      .catch((error: any) => {
        console.log(error);
        createLogFile(error);
        createLogFile("Failed to update python dependencies");
        createLogFile(appDir);
        reject(error);
      });
  });
}

// Ensure all required directories exist and if not, download them
function fixMissingDirectories(win: typeof BrowserWindow) {
  return new Promise((resolve, reject) => {
    win.webContents.send("updateStatus", "Checking for updatess...");
    downloadResources(win, false).then(() => {
      resolve(true);
    });
  });
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

  checkForUpdates();

  win.webContents.once("did-finish-load", () => {
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
          updatePythonDependencies(win).then(() => {
            // Check for new patch
            // Check if any directories are missing
            fixMissingDirectories(win).then(() => {
              win.loadFile("pages/index.html");
            });
          });
        }
      })
      .catch((error) => {
        // Python install failed
        console.log(error);
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
// Directories
ipcMain.on("openDialog", function (event: any, data: any) {
  let window = BrowserWindow.getFocusedWindow();
  dialog
    .showOpenDialog(window, {
      properties: ["openDirectory"],
    })
    .then((result: { canceled: boolean; filePaths: any[] }) => {
      // Check for a valid result
      if (!result.canceled) {
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
    })
    .then((result: { canceled: boolean; filePaths: any[] }) => {
      // Check for a valid result
      if (!result.canceled) {
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
    args: [
      `-o ${data[1]}`,
      `-i ${data[0]}`,
      `-d ${data[2]}`,
      `-t ${data[3]}`,
      "-g False",
    ],
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
    args: [`-i ${data[0]}`, `-s ${structPath}`, `-a ${data[1]}`],
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
    args: [
      `-o ${data[1]}`,
      `-i ${data[0]}`,
      `-w ${data[2]}`,
      `-a ${data[3]}`,
      `-m ${modelPath}`,
      `-n ${nrrdPath}`,
      `-c ${mapPath}`,
      `-l ${data[4]}`,
    ],
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
    args: [
      `-i ${data[0]}`,
      `-o ${data[1]}`,
      `-a ${data[2]}`,
      `-w ${data[3]}`,
      `-m ${structPath}`,
    ],
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

  let custom_args = [
    `-p ${data[0]}`,
    `-a ${data[1]}`,
    `-o ${data[2]}`,
    `-m ${structPath}`,
  ];

  if (data[3]) {
    custom_args.push(`--layers`);
  }

  let options = {
    mode: "text",
    pythonPath: path.join(envPythonPath, pyCommand),
    scriptPath: pyScriptsPath,
    args: custom_args,
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
    args: [
      String.raw`-o ${data[1]}`,
      String.raw`-i ${data[0]}`,
      `-r ${data[2]}`,
      String.raw`-s ${path.join(appDir, "csv/structure_map.pkl")}`,
      "-g False",
    ],
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
  let custom = [
    String.raw`-o ${data[1]}`,
    String.raw`-i ${data[0]}`,
    `-r ${data[2]}`,
    `-a ${data[3]}`,
  ];
  if (data[4]) {
    custom.push(`--equalize`);
  }
  let options = {
    mode: "text",
    pythonPath: path.join(envPythonPath, pyCommand),
    scriptPath: pyScriptsPath,
    args: custom,
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

  let custom_args = [
    `-i ${data[0]}`,
    `-o ${data[1]}`,
    `-c ${data[2]}`,
    `-t ${data[3]}`,
    `-a ${data[7]}`,
    `-s ${sam_model_path}`,
    `-e ${data[8]}`,
    `-m ${modelPath}`,
  ];

  if (data[5]) {
    custom_args.push(`--multichannel`);
  }

  let options = {
    mode: "text",
    pythonPath: path.join(envPythonPath, pyCommand),
    scriptPath: pyScriptsPath,
    args: custom_args,
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
