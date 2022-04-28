// Required modules and structures
const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const {promisify} = require('util');
const {PythonShell} = require('python-shell');
const path = require('path');
const fs = require('fs');
const tar = require('tar');
const mv = promisify(fs.rename);
const exec = promisify(require('child_process').exec);

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
const pyScriptsPath = path.join(__dirname, '/resources/py');

// Promise version of file moving
function move(o, t){
  return new Promise((resolve, reject) => {
    // move o to t, wrapped as promise
    const original = o
    const target = t
    mv(original, target).then(_ => {
      resolve(0);
    })
  })
}

function setupPython(win) {
  return new Promise((resolve, reject) => {
    if (!fs.existsSync(path.join(homeDir, 'python'))) {
      win.webContents.send('updateStatus', 'Settting up python...');
      switch (process.platform) {
        case 'win32':
          tar.x(
            {
              cwd: __dirname,
              file: 'standalone/win/cpython-3.9.6-x86_64-pc-windows-msvc-shared-install_only-20210724T1424.tar.gz'
            }
          ).then(_ => {
            win.webContents.send('updateStatus', 'Extracted python...');
            move(path.join(__dirname, 'python'), path.join(homeDir, 'python')).then(_ => {
              resolve(true);
              fs.rmdir(path.join(__dirname, 'python'), (error) => {
                if (error) {
                  console.log(error);
                }
              });
            });
          });
          break;
        case 'linux':
          tar.x(
            {
              cwd: __dirname,
              file: 'standalone/linux/cpython-3.9.6-x86_64-unknown-linux-gnu-install_only-20210724T1424.tar.gz'
            }
          ).then(_ => {
            win.webContents.send('updateStatus', 'Extracted python...');
            move(path.join(__dirname, 'python'), path.join(homeDir, 'python')).then(_ => {
              resolve(true);
              fs.rmdir(path.join(__dirname, 'python'), (error) => {
                if (error) {
                  console.log(error);
                }
              });
            });
          });
          break;
        case 'darwin':
          tar.x(
            {
              cwd: __dirname,
              file: 'standalone/osx/cpython-3.9.6-aarch64-apple-darwin-install_only-20210724T1424.tar.gz'
            }
          ).then(_ => {
            win.webContents.send('updateStatus', 'Extracted python...');
            move(path.join(__dirname, 'python'), path.join(homeDir, 'python')).then(_ => {
              resolve(true);
              fs.rmdir(path.join(__dirname, 'python'), (error) => {
                if (error) {
                  console.log(error);
                }
              });
            });
          });
          break;
        default:
          tar.x(
            {
              cwd: __dirname,
              file: 'standalone/linux/cpython-3.9.6-x86_64-unknown-linux-gnu-install_only-20210724T1424.tar.gz'
            }
          ).then(_ => {
            win.webContents.send('updateStatus', 'Extracted python...');
            move(path.join(__dirname, 'python'), path.join(homeDir, 'python')).then(_ => {
              resolve(true);
              fs.rmdir(path.join(__dirname, 'python'), (error) => {
                if (error) {
                  console.log(error);
                }
              });
            });
          });
          break;
      }
    } else {
      resolve(false);
    }
  });
}

// Creates the venv and installs the dependencies
function setupVenv(win) {
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
      const reqs = path.join(__dirname, 'resources/py/requirements.txt')
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
    width: 1000,
    height: 525,
    resizable: true,
    autoHideMenuBar: true,
    webPreferences: {nodeIntegration: true, contextIsolation: false }
  })

  win.loadFile('pages/loading.html')

  return win
}

app.on("ready", () => {
  let win = createWindow()
  // Uncomment if you want tools on launch
  // win.webContents.toggleDevTools()
  
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
  if (process.platform !== 'darwin') app.quit()
})

// Handlers
// Directories
ipcMain.on('openDialog', function(event, data){
  let window = BrowserWindow.getFocusedWindow()
  dialog.showOpenDialog(window, {
    properties: ['openDirectory']
  }).then(result => {
    // Check for a valid result
    if (!result.canceled) {
      // console.log(result.filePaths)
      // Send back the dir and whether this is input or output
      event.sender.send('returnPath', [result.filePaths[0], data])
    }
  }).catch(err => {
    console.log(err)
  });
});

// Max Projection
ipcMain.on('runMax', function(event, data){
  let options = {
    mode: 'text',
    pythonPath: path.join(envPythonPath, pyCommand),
    scriptPath: pyScriptsPath,
    args: [`-o ${data[1]}`, `-i ${data[0]}`, '-g False']
  };

  PythonShell.run('batchMaxProjection.py', options, function (err, results) {
    if (err){
      console.log(err);
      event.sender.send('maxError');
    };
    // results is an array consisting of messages collected during execution
    console.log('results: %j', results);
    event.sender.send('maxResult');
  });
});

