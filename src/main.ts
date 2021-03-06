// Required modules and structures
const { app, BrowserWindow, ipcMain, dialog} = require('electron');
const {promisify} = require('util');
const {PythonShell} = require('python-shell');
const path = require('path');
const fs = require('fs');
const tar = require('tar');
const mv = promisify(fs.rename);
const exec = promisify(require('child_process').exec);

var appDir = app.getAppPath()

// Path variables for easy management of execution
const homeDir = path.join(app.getPath('home'), '.belljar');
// Mod is the proper path to the python/pip binary
var mod = (process.platform === 'win32') ? 'python/':'python/bin/'
var envMod = (process.platform === 'win32') ? 'Scripts/':'bin/'
// Make a constant with the cwd for running python commands
const envPath = path.join(homeDir, 'benv');
const pythonPath = path.join(homeDir, mod);
const envPythonPath = path.join(envPath, envMod);
// Command choses wether to use the exe (windows) or alias (unix based)
var pyCommand = (process.platform === 'win32') ? 'python.exe':'./python3'
// Path to our python files
const pyScriptsPath = path.join(appDir, '/resources/py');

// Promise version of file moving
function move(o: string, t: string){
  return new Promise((resolve, reject) => {
    // move o to t, wrapped as promise
    const original = o
    const target = t
    mv(original, target).then(() => {
      resolve(0);
    })
  })
}

function setupPython(win: typeof BrowserWindow) {
  var standalone = path.join(appDir, 'standalone')
  return new Promise((resolve, reject) => {
    if (!fs.existsSync(path.join(homeDir, 'python'))) {
      win.webContents.send('updateStatus', 'Settting up python...');
      switch (process.platform) {
        case 'win32':
          tar.x(
            {
              cwd: homeDir,
              preservePaths: true,
              file: path.join(standalone, 'win/cpython-3.9.6-x86_64-pc-windows-msvc-shared-install_only-20210724T1424.tar.gz')
            }
          ).then(() => {
            win.webContents.send('updateStatus', 'Extracted python...');
            resolve(true);
          });
          break;
        case 'linux':
          tar.x(
            {
              cwd: homeDir,
              preservePaths: true,
              file: path.join(standalone, 'linux/cpython-3.9.6-x86_64-unknown-linux-gnu-install_only-20210724T1424.tar.gz')
            }
          ).then(() => {
            win.webContents.send('updateStatus', 'Extracted python...');
            resolve(true);
          });
          break;
        case 'darwin':
          tar.x(
            {
              cwd: homeDir,
              preservePaths: true,
              file: path.join(standalone, 'osx/cpython-3.9.6-aarch64-apple-darwin-install_only-20210724T1424.tar.gz')
            }
          ).then(() => {
            win.webContents.send('updateStatus', 'Extracted python...');
            resolve(true);
          });
          break;
        default:
          tar.x(
            {
              cwd: homeDir,
              preservePaths: true,
              file: path.join(standalone, 'linux/cpython-3.9.6-x86_64-unknown-linux-gnu-install_only-20210724T1424.tar.gz')
            }
          ).then(() => {
            win.webContents.send('updateStatus', 'Extracted python...');
            resolve(true);
          });
          break;
      }
    } else {
      resolve(false);
    }
  });
}

// Creates the venv and installs the dependencies
function setupVenv(win: typeof BrowserWindow) {
  win.webContents.send('updateStatus', 'Installing venv...');
  // Check if the enviornment was already made
  if (!fs.existsSync(envPath)) {
    // Promise chain to setup the enviornment
    installVenv().then(({stdout, stderr}) => {
      console.log(stdout);
      win.webContents.send('updateStatus', 'Creating venv...');
      createVenv().then(({stdout, stderr}) => {
        console.log(stdout);
        win.webContents.send('updateStatus', 'Installing packages...');
        installDeps().then(({stdout, stderr}) => {
          console.log(stdout);
          win.webContents.send('updateStatus', 'Setup complete!');
          win.loadFile('pages/index.html')
        }).catch((error) => {
          console.log(error);
        });
      }).catch((error) => {
        console.log(error);
      });
    }).catch((error) => {
      console.log(error);
    });

    // Install venv package
    async function installVenv() {
      const {stdout, stderr} = await exec(`${pyCommand} -m pip install --user virtualenv`, {cwd: pythonPath});
      return {stdout, stderr};
    }

    // Create venv
    async function createVenv() {
      const envDir = (process.platform === 'win32') ? '../benv':'../../benv'
      const {stdout, stderr} = await exec(`${pyCommand} -m venv ${envDir}`, {cwd: pythonPath});
      return {stdout, stderr};
    }

    // Install pip packages
    async function installDeps() {
      const reqs = path.join(appDir, 'resources/py/requirements.txt')
      const {stdout, stderr} = await exec(`${pyCommand} -m pip install -r ${reqs}`, {cwd: envPythonPath});
      return {stdout, stderr};
    }
  } 
}

// Makes the local user writable folder
function checkLocalDir() {
  if (!fs.existsSync(homeDir)) {
    fs.mkdirSync(homeDir, {
      recursive: true
    });
  }
}

function createWindow () {
  const win = new BrowserWindow({
    width: 1250,
    height: 750,
    resizable: true,
    autoHideMenuBar: true,
    webPreferences: {nodeIntegration: true, contextIsolation: false }
  })

  win.loadFile('pages/loading.html')

  return win
}

let win: typeof BrowserWindow = null;

app.on("ready", () => {
  let win = createWindow()
  // Uncomment if you want tools on launch
  // win.webContents.toggleDevTools()
  win.on('close', function(e: any) {
    const choice = dialog.showMessageBoxSync(win,
      {
        type: 'question',
        buttons: ['Yes', 'Cancel'],
        title: 'Confrim Quit',
        message: 'Are you sure you want to quit? Quitting will kill all running processes.'
      });
    if (choice === 1) {
      e.preventDefault();
    }
  });

  win.webContents.once('did-finish-load', () => {
    // Make a directory to house enviornment, settings, etc.
    checkLocalDir();
    // Setup python for running the pipeline
    setupPython(win).then((installed) => {
      // Prepare depedencies
      if (installed) {
        setupVenv(win);
      } else {
        win.loadFile('pages/index.html');
      }
    }).catch((error) => {
      // Python install failed
      console.log(error);
    });
  });
})

app.whenReady().then(() => {
  app.on('activate', function () {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', function () {
  app.quit()
})

// Handlers
// Directories
ipcMain.on('openDialog', function(event: any, data: any){
  let window = BrowserWindow.getFocusedWindow()
  dialog.showOpenDialog(window, {
    properties: ['openDirectory']
  }).then((result: { canceled: boolean; filePaths: any[]; }) => {
    // Check for a valid result
    if (!result.canceled) {
      // console.log(result.filePaths)
      // Send back the dir and whether this is input or output
      event.sender.send('returnPath', [result.filePaths[0], data])
    }
  }).catch((err: Error) => {
    console.log(err)
  });
});
// Files
ipcMain.on('openFileDialog', function(event: any, data: any){
  let window = BrowserWindow.getFocusedWindow()
  dialog.showOpenDialog(window, {
    properties: ['openFile']
  }).then((result: { canceled: boolean; filePaths: any[]; }) => {
    // Check for a valid result
    if (!result.canceled) {
      // console.log(result.filePaths)
      // Send back the dir and whether this is input or output
      event.sender.send('returnPath', [result.filePaths[0], data])
    }
  }).catch((err: Error) => {
    console.log(err)
  });
});

// Max Projection
ipcMain.on('runMax', function(event: any, data: any[]){
  let options = {
    mode: 'text',
    pythonPath: path.join(envPythonPath, pyCommand),
    scriptPath: pyScriptsPath,
    args: [
            `-o ${data[1]}`, 
            `-i ${data[0]}`,
            '-g False'
          ]
  };
  
  let pyshell = new PythonShell('batchMaxProjection.py', options);
  var total: number = 0;
  var current: number = 0;
  pyshell.on('message', (message: string) => {
    if (total === 0) {
      total = Number(message);
    } else if (message == 'Done!') {
      pyshell.end((err: string, code: any, signal: string) => {
        if (err) throw err;
        console.log('The exit code was: ' + code);
        console.log('The exit signal was: ' + signal);
        event.sender.send('maxResult')
      });
    } else {
      current++;
      event.sender.send('updateLoad', [Math.round((current/total)*100), message]);
    }
  });

  ipcMain.once('killMax', function(event: any, data: any[]) {
    pyshell.kill();
  });
});

// Alignment
ipcMain.on('runAlign', function(event: any, data: any[]){
  var modelPath = (data[2] == "False" ? path.join(appDir, 'resources/models/predictor_encoder.pt'):path.join(appDir, 'resources/models/predictor_full_encoder.pt'));
  var embedPath = path.join(appDir, 'resources/py/atlasEmbeddings.pkl')
  var structPath = path.join(appDir, 'resources/csv/structure_tree_safe_2017.csv')

  let options = {
    mode: 'text',
    pythonPath: path.join(envPythonPath, pyCommand),
    scriptPath: pyScriptsPath,
    args: [
            `-o ${data[1]}`, 
            `-i ${data[0]}`,
            `-w ${data[2]}`,
            `-m ${modelPath}`,
            `-e ${embedPath}`,
            `-s ${structPath}`
          ]
  };
  let pyshell = new PythonShell('mapToAtlas.py', options);
  var total: number = 0;
  var current: number = 0;

  pyshell.on('stderr', function (stderr: string) {
    console.log(stderr);
  });

  pyshell.on('message', (message: string) => {
    console.log(message);
    if (total === 0) {
      total = Number(message);
    } else if (message == 'Done!') {
      pyshell.end((err: string, code: any, signal: string) => {
        if (err) throw err;
        event.sender.send('alignResult');
        console.log('The exit code was: ' + code);
        console.log('The exit signal was: ' + signal);
      });
    } else {
      current++;
      event.sender.send('updateLoad', [Math.round((current/total)*100), message]);
    }
  });

  ipcMain.once('killAlign', function(event: any, data: any[]) {
    pyshell.kill();
  });
});

// Counting
ipcMain.on('runCount', function(event: any, data: any[]){
  var structPath = path.join(appDir, 'resources/csv/structure_tree_safe_2017.csv')

  let options = {
    mode: 'text',
    pythonPath: path.join(envPythonPath, pyCommand),
    scriptPath: pyScriptsPath,
    args: [
            `-p ${data[0]}`, 
            `-a ${data[1]}`,
            `-o ${data[2]}`,
            `-s ${structPath}`
          ]
  };
  let pyshell = new PythonShell('countBrain.py', options);
  var total: number = 0;
  var current: number = 0;

  pyshell.on('stderr', function (stderr: string) {
    console.log(stderr);
  });

  pyshell.on('message', (message: string) => {
    console.log(message);
    if (total === 0) {
      total = Number(message);
    } else if (message == 'Done!') {
      pyshell.end((err: string, code: any, signal: string) => {
        if (err) throw err;
        console.log('The exit code was: ' + code);
        console.log('The exit signal was: ' + signal);
        event.sender.send('countResult');
      });
    } else {
      current++;
      event.sender.send('updateLoad', [Math.round((current/total)*100), message]);
    }
  });

  ipcMain.once('killCount', function(event: any, data: any[]) {
    pyshell.kill();
  });
});

// Top Hat
ipcMain.on('runTopHat', function(event: any, data: any[]){
  let options = {
    mode: 'text',
    pythonPath: path.join(envPythonPath, pyCommand),
    scriptPath: pyScriptsPath,
    args: [
            `-o ${data[1]}`, 
            `-i ${data[0]}`, 
            `-f ${data[2]}`, 
            `-c ${data[3]}`, 
            '-g False'
          ]
  };
  
  let pyshell = new PythonShell('topHat.py', options);
  var total: number = 0;
  var current: number = 0;
  pyshell.on('message', (message: string) => {
    if (total === 0) {
      total = Number(message);
    } else if (message == 'Done!') {
      pyshell.end((err: string, code: any, signal: string) => {
        if (err) throw err;
        console.log('The exit code was: ' + code);
        console.log('The exit signal was: ' + signal);
        event.sender.send('topHatResult')
      });
    } else {
      current++;
      event.sender.send('updateLoad', [Math.round((current/total)*100), message]);
    }
  });

  ipcMain.once('killTopHat', function(event: any, data: any[]) {
    pyshell.kill();
  });
});

// Collate
ipcMain.on('runCollate', function(event: any, data: any[]){
  let options = {
    mode: 'text',
    pythonPath: path.join(envPythonPath, pyCommand),
    scriptPath: pyScriptsPath,
    args: [
            String.raw`-o ${data[1]}`, 
            String.raw`-i ${data[0]}`, 
            `-r ${data[2]}`, 
            String.raw`-s ${path.join(appDir, 'resources/csv/structure_tree_safe_2017.csv')}`, 
            '-g False'
          ]
  };
  let pyshell = new PythonShell('collateCounts.py', options);

  pyshell.end((err: string, code: any, signal: string) => {
    if (err) throw err;
    console.log('The exit code was: ' + code);
    console.log('The exit signal was: ' + signal);
    event.sender.send('collateResult')
  });

  ipcMain.once('killCollate', function(event: any, data: any[]) {
    pyshell.kill();
  });
});

// Cell Detection
ipcMain.on('runDetection', function(event: any, data: any[]){
  // Set model path
  var modelPath = path.join(appDir, 'resources/models/ancientwizard.pt')
  // Switch over to custom if necessary
  if (data[4].length > 0) {
    modelPath = data[4]
  }

  let options = {
    mode: 'text',
    pythonPath: path.join(envPythonPath, pyCommand),
    scriptPath: pyScriptsPath,
    args: [
            `-i ${data[0]}`, 
            `-o ${data[1]}`, 
            `-c ${data[2]}`, 
            `-t ${data[3]}`, 
            `-m ${modelPath}`, 
            '-g False'
          ]
  };

  let pyshell = new PythonShell('findNeurons.py', options);
  var total: number = 0;
  var current: number = 0;
  pyshell.on('message', (message: string) => {
    if (total === 0) {
      total = Number(message);
    } else if (message == 'Done!') {
      pyshell.end((err: string, code: any, signal: string) => {
        if (err) throw err;
        console.log('The exit code was: ' + code);
        console.log('The exit signal was: ' + signal);
        event.sender.send('detectResult')
      });
    } else if (message.includes("Processing")) {
      current++;
      event.sender.send('updateLoad', [Math.round((current/total)*100), message]);
    }
  });

  ipcMain.once('killDetect', function(event: any, data: any[]) {
    pyshell.kill();
  });
});

