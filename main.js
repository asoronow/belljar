"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
// Required modules and structures
const { app, BrowserWindow, ipcMain, dialog } = require("electron");
const { promisify } = require("util");
const { PythonShell } = require("python-shell");
const path = require("path");
const fs = require("fs");
const tar = require("tar");
const mv = promisify(fs.rename);
const exec = promisify(require("child_process").exec);
const stream = require("stream");
const https = require("https");
var appDir = app.getAppPath();
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
// Promise version of file moving
function move(o, t) {
    return new Promise((resolve, reject) => {
        // move o to t, wrapped as promise
        const original = o;
        const target = t;
        mv(original, target).then(() => {
            resolve(0);
        });
    });
}
function createLogFile(message) {
    const logPath = path.join(homeDir, "belljar.log");
    fs.appendFileSync(logPath, message);
}
// Get files asynchonously
function downloadFile(url, target, win) {
    return new Promise((resolve, reject) => {
        const file = fs.createWriteStream(target);
        // get the file, update the user loading screen with text on progress
        const requestedFileName = url.split("/").pop();
        const progress = (receivedBytes, totalBytes) => {
            const percentage = (receivedBytes * 100) / totalBytes;
            win.webContents.send("updateStatus", `Downloading ${requestedFileName}... ${percentage.toFixed(0)}%`);
        };
        const request = https.get(url, (response) => {
            // create a dummy stream so we can update the user on progress
            const dummy = new stream.PassThrough();
            var receivedBytes = 0;
            var totalBytes = parseInt(response.headers["content-length"]);
            response.pipe(dummy);
            dummy.on("data", (chunk) => {
                receivedBytes += chunk.length;
                progress(receivedBytes, totalBytes);
            });
            // pipe the response to the file
            response.pipe(file);
            file.on("finish", () => {
                file.close();
                win.webContents.send("updateStatus", `Extracting ${requestedFileName}...`);
                resolve(true);
            });
        });
    });
}
// Delete a file safely
function deleteFile(file) {
    return new Promise((resolve, reject) => {
        fs.unlinkSync(file);
        resolve(true);
    });
}
function setupPython(win) {
    const bucketParentPath = "https://storage.googleapis.com/belljar_updates";
    const linuxURL = `${bucketParentPath}/cpython-3.9.6-x86_64-unknown-linux-gnu-install_only-20210724T1424.tar.gz`;
    const winURL = `${bucketParentPath}/cpython-3.9.6-x86_64-pc-windows-msvc-shared-install_only-20210724T1424.tar.gz`;
    const osxURL = `${bucketParentPath}/cpython-3.9.6-aarch64-apple-darwin-install_only-20210724T1424.tar.gz`;
    return new Promise((resolve, reject) => {
        if (!fs.existsSync(path.join(homeDir, "python"))) {
            win.webContents.send("updateStatus", "Settting up python...");
            switch (process.platform) {
                case "win32":
                    // Download and extract python to the home directory
                    downloadFile(winURL, path.join(homeDir, "cpython-3.9.6-x86_64-pc-windows-msvc-shared-install_only-20210724T1424.tar.gz"), win)
                        .then(() => {
                        // Extract the tarball  
                        tar
                            .x({
                            cwd: homeDir,
                            preservePaths: true,
                            file: path.join(homeDir, "cpython-3.9.6-x86_64-pc-windows-msvc-shared-install_only-20210724T1424.tar.gz"),
                        })
                            .then(() => {
                            win.webContents.send("updateStatus", "Extracted python...");
                            resolve(true);
                        });
                    })
                        .catch((err) => {
                        console.log(err);
                    });
                    break;
                case "linux":
                    downloadFile(linuxURL, path.join(homeDir, "cpython-3.9.6-x86_64-unknown-linux-gnu-install_only-20210724T1424.tar.gz"), win)
                        .then(() => {
                        tar
                            .x({
                            cwd: homeDir,
                            preservePaths: true,
                            file: path.join(homeDir, "cpython-3.9.6-x86_64-unknown-linux-gnu-install_only-20210724T1424.tar.gz"),
                        })
                            .then(() => {
                            win.webContents.send("updateStatus", "Extracted python...");
                            resolve(true);
                        });
                    });
                    break;
                case "darwin":
                    downloadFile(osxURL, path.join(homeDir, "cpython-3.9.6-aarch64-apple-darwin-install_only-20210724T1424.tar.gz"), win)
                        .then(() => {
                        tar
                            .x({
                            cwd: homeDir,
                            preservePaths: true,
                            file: path.join(homeDir, "cpython-3.9.6-aarch64-apple-darwin-install_only-20210724T1424.tar.gz"),
                        })
                            .then(() => {
                            win.webContents.send("updateStatus", "Extracted python...");
                            resolve(true);
                        });
                    });
                    break;
                default:
                    // If we don't have a supported platform, just resolve
                    resolve(true);
                    break;
            }
        }
        else {
            // Double check that the environment is setup by confirming if the benv folder exists
            if (!fs.existsSync(envPath)) {
                resolve(true);
            }
            else {
                resolve(false);
            }
        }
    });
}
// Download the required tar files from the bucket
function downloadResources(win, fresh) {
    // Download the tar files into the homeDir and extract them to their respective folders
    return new Promise((resolve, reject) => {
        const bucketParentPath = "https://storage.googleapis.com/belljar_updates";
        const embeddingsLink = `${bucketParentPath}/embeddings.tar.gz`;
        const modelsLink = `${bucketParentPath}/models.tar.gz`;
        const nrrdLink = `${bucketParentPath}/nrrd.tar.gz`;
        const requiredDirs = ["models", "embeddings", "nrrd"];
        if (!fresh) {
            var downloading = [];
            var total = 0;
            // Just check if each directory exists and its not empty
            for (let i = 0; i < requiredDirs.length; i++) {
                const dir = requiredDirs[i];
                if (!fs.existsSync(path.join(homeDir, dir)) || fs.readdirSync(path.join(homeDir, dir)).length === 0) {
                    downloading.push(dir);
                }
            }
            for (let i = 0; i < downloading.length; i++) {
                const dir = downloading[i];
                win.webContents.send("updateStatus", `Redownloading ${dir}...this may take a while`);
                // Remove the directory if it exists, download tar and extract
                if (fs.existsSync(path.join(homeDir, dir))) {
                    fs.rmdirSync(path.join(homeDir, dir), { recursive: true });
                }
                // Download the tar file
                downloadFile(`${bucketParentPath}/${dir}.tar.gz`, path.join(homeDir, `${dir}.tar.gz`), win).then(() => {
                    // Extract the tar file
                    tar.x({ cwd: homeDir, preservePaths: true, file: path.join(homeDir, `${dir}.tar.gz`) }).then(() => {
                        // Delete the tar file
                        deleteFile(path.join(homeDir, `${dir}.tar.gz`)).then(() => {
                            win.webContents.send("updateStatus", `Downloaded ${dir}`);
                            total++;
                            if (downloading.length === total) {
                                resolve(true);
                            }
                        });
                    });
                });
            }
            if (downloading.length === 0) {
                resolve(true);
            }
        }
        // Since we are doing a fresh install, we need to ensure no remnants of the old install are left or partially downloaded
        // Check if these directories exist, if they do, we don't need to download any files
        let allDirsExist = true;
        requiredDirs.forEach((dir) => {
            if (!fs.existsSync(path.join(homeDir, dir))) {
                allDirsExist = false;
            }
        });
        if (!allDirsExist) {
            // Something is missing, delete everything and download again
            requiredDirs.forEach((dir) => {
                if (fs.existsSync(path.join(homeDir, dir))) {
                    fs.rmdirSync(path.join(homeDir, dir), { recursive: true });
                }
            });
            // Download the embeddings
            downloadFile(embeddingsLink, path.join(homeDir, "embeddings.tar.gz"), win).then(() => {
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
                        downloadFile(modelsLink, path.join(homeDir, "models.tar.gz"), win).then(() => {
                            // Extract the models
                            tar
                                .x({
                                cwd: homeDir,
                                preservePaths: true,
                                file: path.join(homeDir, "models.tar.gz"),
                            })
                                .then(() => {
                                // Delete the tar file
                                deleteFile(path.join(homeDir, "models.tar.gz")).then(() => {
                                    // Download the nrrd
                                    downloadFile(nrrdLink, path.join(homeDir, "nrrd.tar.gz"), win).then(() => {
                                        // Extract the nrrd
                                        tar
                                            .x({
                                            cwd: homeDir,
                                            preservePaths: true,
                                            file: path.join(homeDir, "nrrd.tar.gz"),
                                        })
                                            .then(() => {
                                            // Delete the tar file
                                            deleteFile(path.join(homeDir, "nrrd.tar.gz")).then(() => {
                                                resolve(true);
                                            });
                                        });
                                    });
                                });
                            });
                        });
                    });
                });
            });
        }
        else {
            //TODO: Error handling for unsupported platforms
            resolve(true);
        }
    });
}
// Creates the venv and installs the dependencies
function setupEnvironment(win) {
    if (!fs.existsSync(envPath)) {
        // We have not created the venv yet, so we probably don't have the models, etc. either
        // Download the required files, checking if their directories exist
        win.webContents.send("updateStatus", "Preparing to download require files...");
        downloadResources(win, true).then(() => {
            // Promise chain to setup the python enviornment
            win.webContents.send("updateStatus", "Installing venv...");
            installVenv()
                .then(({ stdout, stderr }) => {
                console.log(stdout);
                win.webContents.send("updateStatus", "Creating venv...");
                createVenv()
                    .then(({ stdout, stderr }) => {
                    console.log(stdout);
                    win.webContents.send("updateStatus", "Installing packages...");
                    installDeps()
                        .then(({ stdout, stderr }) => {
                        console.log(stdout);
                        win.webContents.send("updateStatus", "Setup complete!");
                        win.loadFile("pages/index.html");
                    })
                        .catch((error) => {
                        console.log(error);
                    });
                })
                    .catch((error) => {
                    console.log(error);
                });
            })
                .catch((error) => {
                console.log(error);
            });
        });
        // Install venv package
        function installVenv() {
            return __awaiter(this, void 0, void 0, function* () {
                const { stdout, stderr } = yield exec(`${pyCommand} -m pip install --user virtualenv`, { cwd: pythonPath });
                return { stdout, stderr };
            });
        }
        // Create venv
        function createVenv() {
            return __awaiter(this, void 0, void 0, function* () {
                const envDir = process.platform === "win32" ? "../benv" : "../../benv";
                const { stdout, stderr } = yield exec(`${pyCommand} -m venv ${envDir}`, {
                    cwd: pythonPath,
                });
                return { stdout, stderr };
            });
        }
        // Install pip packages
        function installDeps() {
            return __awaiter(this, void 0, void 0, function* () {
                const reqs = path.join(appDir, "py/requirements.txt");
                const { stdout, stderr } = yield exec(`${pyCommand} -m pip install -r ${reqs} --use-pep517`, { cwd: envPythonPath });
                return { stdout, stderr };
            });
        }
    }
}
// Install the latest dependencies, could have changed after an update
function updatePythonDependencies(win) {
    return new Promise((resolve, reject) => {
        win.webContents.send("updateStatus", "Updating packages...");
        // Run pip install -r requirements.txt --no-cache-dir to update the packages
        exec(`${pyCommand} -m pip install -r ${path.join(appDir, "py/requirements.txt")} --no-cache-dir  --use-pep517`, { cwd: envPythonPath })
            .then(({ stdout, stderr }) => {
            console.log(stdout);
            win.webContents.send("updateStatus", "Update complete!");
            resolve(true);
        }).catch((error) => {
            console.log(error);
            createLogFile(error);
            createLogFile("Failed to update python dependencies");
            createLogFile(appDir);
            reject(error);
        });
    });
}
// Ensure all required directories exist and if not, download them
function fixMissingDirectories(win) {
    return new Promise((resolve, reject) => {
        win.webContents.send("updateStatus", "Checking for missing files...");
        downloadResources(win, false).then(() => {
            resolve(true);
        });
    });
}
// Makes the local user writable folder
// TODO: Version checking to see if we need to update the files
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
let win = null;
app.on("ready", () => {
    const win = createWindow();
    // Uncomment if you want tools on launch
    // win.webContents.toggleDevTools()
    win.on("close", function (e) {
        const choice = dialog.showMessageBoxSync(win, {
            type: "question",
            buttons: ["Yes", "Cancel"],
            title: "Confrim Quit",
            message: "Are you sure you want to quit? Quitting will kill all running processes.",
        });
        if (choice === 1) {
            e.preventDefault();
        }
    });
    win.webContents.once("did-finish-load", () => {
        // Make a directory to house enviornment, settings, etc.
        checkLocalDir();
        // Setup python for running the pipeline
        setupPython(win)
            .then((installed) => {
            // If we just installed python, we need to continue the complete
            // setup of the enviornment
            if (installed) {
                setupEnvironment(win);
            }
            else {
                // Otherwise, we can just update the dependencies
                updatePythonDependencies(win).then(() => {
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
        if (BrowserWindow.getAllWindows().length === 0)
            createWindow();
    });
});
app.on("window-all-closed", function () {
    app.quit();
});
// Handlers
// Directories
ipcMain.on("openDialog", function (event, data) {
    let window = BrowserWindow.getFocusedWindow();
    dialog
        .showOpenDialog(window, {
        properties: ["openDirectory"],
    })
        .then((result) => {
        // Check for a valid result
        if (!result.canceled) {
            // console.log(result.filePaths)
            // Send back the dir and whether this is input or output
            event.sender.send("returnPath", [result.filePaths[0], data]);
        }
    })
        .catch((err) => {
        console.log(err);
    });
});
// Files
ipcMain.on("openFileDialog", function (event, data) {
    let window = BrowserWindow.getFocusedWindow();
    dialog
        .showOpenDialog(window, {
        properties: ["openFile"],
    })
        .then((result) => {
        // Check for a valid result
        if (!result.canceled) {
            // console.log(result.filePaths)
            // Send back the dir and whether this is input or output
            event.sender.send("returnPath", [result.filePaths[0], data]);
        }
    })
        .catch((err) => {
        console.log(err);
    });
});
// Max Projection
ipcMain.on("runMax", function (event, data) {
    let options = {
        mode: "text",
        pythonPath: path.join(envPythonPath, pyCommand),
        scriptPath: pyScriptsPath,
        args: [`-o ${data[1]}`, `-i ${data[0]}`, "-g False"],
    };
    let pyshell = new PythonShell("batchMaxProjection.py", options);
    var total = 0;
    var current = 0;
    pyshell.on("message", (message) => {
        if (total === 0) {
            total = Number(message);
        }
        else if (message == "Done!") {
            pyshell.end((err, code, signal) => {
                if (err)
                    throw err;
                console.log("The exit code was: " + code);
                console.log("The exit signal was: " + signal);
                event.sender.send("maxResult");
                ipcMain.removeAllListeners("killMax");
            });
        }
        else {
            current++;
            event.sender.send("updateLoad", [
                Math.round((current / total) * 100),
                message,
            ]);
        }
    });
    ipcMain.once("killMax", function (event, data) {
        pyshell.kill();
    });
});
// Adjust
ipcMain.on("runAdjust", function (event, data) {
    var structPath = path.join(appDir, "csv/structure_tree_safe_2017.csv");
    let options = {
        mode: "text",
        pythonPath: path.join(envPythonPath, pyCommand),
        scriptPath: pyScriptsPath,
        args: [`-i ${data[0]}`, `-s ${structPath}`, `-m ${data[1]}`],
    };
    let pyshell = new PythonShell("adjustAlignment.py", options);
    var total = 0;
    var current = 0;
    pyshell.on("stderr", function (stderr) {
        console.log(stderr);
    });
    pyshell.on("message", (message) => {
        if (total === 0) {
            total = Number(message);
        }
        else if (message == "Done!") {
            pyshell.end((err, code, signal) => {
                if (err)
                    throw err;
                console.log("The exit code was: " + code);
                console.log("The exit signal was: " + signal);
                event.sender.send("adjustResult");
                ipcMain.removeAllListeners("killAdjust");
            });
        }
        else {
            current++;
            event.sender.send("updateLoad", [
                Math.round((current / total) * 100),
                message,
            ]);
        }
    });
    ipcMain.once("killAdjust", function (event, data) {
        pyshell.kill();
    });
});
// Alignment
ipcMain.on("runAlign", function (event, data) {
    const modelPath = data[2] == "False"
        ? path.join(homeDir, "models/predictor_encoder.pt")
        : path.join(homeDir, "models/predictor_full_encoder.pt");
    const embedPath = data[2] == "False"
        ? path.join(homeDir, "embeddings/hemisphere_embeddings.pkl")
        : path.join(homeDir, "embeddings/whole_embeddings.pkl");
    const nrrdPath = path.join(homeDir, "nrrd");
    const structPath = path.join(appDir, "csv/structure_tree_safe_2017.csv");
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
            `-e ${embedPath}`,
            `-n ${nrrdPath}`,
            `-s ${structPath}`,
        ],
    };
    let pyshell = new PythonShell("mapToAtlas.py", options);
    var total = 0;
    var current = 0;
    pyshell.on("stderr", function (stderr) {
        console.log(stderr);
    });
    pyshell.on("message", (message) => {
        console.log(message);
        if (total === 0) {
            total = Number(message);
        }
        else if (message == "Done!") {
            pyshell.end((err, code, signal) => {
                if (err)
                    throw err;
                event.sender.send("alignResult");
                console.log("The exit code was: " + code);
                console.log("The exit signal was: " + signal);
                ipcMain.removeAllListeners("killAlign");
            });
        }
        else {
            current++;
            event.sender.send("updateLoad", [
                Math.round((current / total) * 100),
                message,
            ]);
        }
    });
    ipcMain.once("killAlign", function (event, data) {
        pyshell.kill();
    });
});
// Intensity by Region
ipcMain.on("runIntensity", function (event, data) {
    const structPath = path.join(appDir, "csv/structure_tree_safe_2017.csv");
    let options = {
        mode: "text",
        pythonPath: path.join(envPythonPath, pyCommand),
        scriptPath: pyScriptsPath,
        args: [
            `-i ${data[0]}`,
            `-o ${data[1]}`,
            `-a ${data[2]}`,
            `-w ${data[3]}`,
            `-s ${structPath}`,
        ],
    };
    let pyshell = new PythonShell("intensityByRegion.py", options);
    var total = 0;
    var current = 0;
    pyshell.on("stderr", function (stderr) {
        console.log(stderr);
    });
    pyshell.on("message", (message) => {
        if (total === 0) {
            total = Number(message);
        }
        else if (message == "Done!") {
            pyshell.end((err, code, signal) => {
                if (err)
                    throw err;
                console.log("The exit code was: " + code);
                console.log("The exit signal was: " + signal);
                event.sender.send("intensityResult");
                ipcMain.removeAllListeners("killIntensity");
            });
        }
        else {
            current++;
            event.sender.send("updateLoad", [
                Math.round((current / total) * 100),
                message,
            ]);
        }
    });
    ipcMain.once("killIntensity", function (event, data) {
        pyshell.kill();
    });
});
// Counting
ipcMain.on("runCount", function (event, data) {
    var structPath = path.join(appDir, "csv/structure_tree_safe_2017.csv");
    let options = {
        mode: "text",
        pythonPath: path.join(envPythonPath, pyCommand),
        scriptPath: pyScriptsPath,
        args: [
            `-p ${data[0]}`,
            `-a ${data[1]}`,
            `-o ${data[2]}`,
            `-s ${structPath}`,
        ],
    };
    let pyshell = new PythonShell("countBrain.py", options);
    var total = 0;
    var current = 0;
    pyshell.on("stderr", function (stderr) {
        console.log(stderr);
    });
    pyshell.on("message", (message) => {
        console.log(message);
        if (total === 0) {
            total = Number(message);
        }
        else if (message == "Done!") {
            pyshell.end((err, code, signal) => {
                if (err)
                    throw err;
                console.log("The exit code was: " + code);
                console.log("The exit signal was: " + signal);
                event.sender.send("countResult");
                ipcMain.removeAllListeners("killCount");
            });
        }
        else {
            current++;
            event.sender.send("updateLoad", [
                Math.round((current / total) * 100),
                message,
            ]);
        }
    });
    ipcMain.once("killCount", function (event, data) {
        pyshell.kill();
    });
});
// Top Hat
ipcMain.on("runTopHat", function (event, data) {
    let options = {
        mode: "text",
        pythonPath: path.join(envPythonPath, pyCommand),
        scriptPath: pyScriptsPath,
        args: [
            `-o ${data[1]}`,
            `-i ${data[0]}`,
            `-f ${data[2]}`,
            `-c ${data[3]}`,
            "-g False",
        ],
    };
    let pyshell = new PythonShell("topHat.py", options);
    var total = 0;
    var current = 0;
    pyshell.on("message", (message) => {
        if (total === 0) {
            total = Number(message);
        }
        else if (message == "Done!") {
            pyshell.end((err, code, signal) => {
                if (err)
                    throw err;
                console.log("The exit code was: " + code);
                console.log("The exit signal was: " + signal);
                event.sender.send("topHatResult");
                ipcMain.removeAllListeners("killTopHat");
            });
        }
        else {
            current++;
            event.sender.send("updateLoad", [
                Math.round((current / total) * 100),
                message,
            ]);
        }
    });
    ipcMain.once("killTopHat", function (event, data) {
        pyshell.kill();
    });
});
// Collate
ipcMain.on("runCollate", function (event, data) {
    let options = {
        mode: "text",
        pythonPath: path.join(envPythonPath, pyCommand),
        scriptPath: pyScriptsPath,
        args: [
            String.raw `-o ${data[1]}`,
            String.raw `-i ${data[0]}`,
            `-r ${data[2]}`,
            String.raw `-s ${path.join(appDir, "csv/structure_tree_safe_2017.csv")}`,
            "-g False",
        ],
    };
    let pyshell = new PythonShell("collateCounts.py", options);
    pyshell.end((err, code, signal) => {
        if (err)
            throw err;
        console.log("The exit code was: " + code);
        console.log("The exit signal was: " + signal);
        event.sender.send("collateResult");
    });
    ipcMain.once("killCollate", function (event, data) {
        pyshell.kill();
    });
});
// Cell Detection
ipcMain.on("runDetection", function (event, data) {
    // Set model path
    var modelPath = path.join(homeDir, "models/ancientwizard.pt");
    // Switch over to custom if necessary
    if (data[4].length > 0) {
        modelPath = data[4];
    }
    let options = {
        mode: "text",
        pythonPath: path.join(envPythonPath, pyCommand),
        scriptPath: pyScriptsPath,
        args: [
            `-i ${data[0]}`,
            `-o ${data[1]}`,
            `-c ${data[2]}`,
            `-t ${data[3]}`,
            `-m ${modelPath}`,
            "-g False",
        ],
    };
    let pyshell = new PythonShell("findNeurons.py", options);
    var total = 0;
    var current = 0;
    pyshell.on("message", (message) => {
        console.log(message);
        if (total === 0) {
            total = Number(message);
        }
        else if (message == "Done!") {
            pyshell.end((err, code, signal) => {
                if (err)
                    throw err;
                console.log("The exit code was: " + code);
                console.log("The exit signal was: " + signal);
                event.sender.send("detectResult");
                ipcMain.removeAllListeners("killDetect");
            });
        }
        else if (message.includes("Processing")) {
            current++;
            event.sender.send("updateLoad", [
                Math.round((current / total) * 100),
                message,
            ]);
        }
    });
    ipcMain.once("killDetect", function (event, data) {
        pyshell.kill();
    });
});
