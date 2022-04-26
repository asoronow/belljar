const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const {promisify} = require('util');
const {PythonShell} = require('python-shell');
const path = require('path');
const fs = require('fs');
const tar = require('tar');
const mv = promisify(fs.rename);
const exec = promisify(require('child_process').exec);

function move(o, t){
  return new Promise((resolve, reject) => {
    // move o to t, wrapped as promise
    const original = o
    const target = t
    mv(original, target).then(_ => {
      return resolve;
    })
  })
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



function setupPython(win, homeDir) {
  const pythonPath = path.join(homeDir, 'python');
  if (!fs.existsSync(pythonPath)) {
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
          move(path.join(__dirname, 'python'), pythonPath).then(_ => {
            fs.rmdir(path.join(__dirname, 'python'), (error) => {
              console.log('here')
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
          move(path.join(__dirname, 'python'), pythonPath).then(_ => {
            fs.rmdir(path.join(__dirname, 'python'), (error) => {
              console.log('here')
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
          move(path.join(__dirname, 'python'), pythonPath).then(_ => {
            fs.rmdir(path.join(__dirname, 'python'), (error) => {
              console.log('here')
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
          move(path.join(__dirname, 'python'), pythonPath).then(_ => {
            fs.rmdir(path.join(__dirname, 'python'), (error) => {
              console.log('here')
              if (error) {
                console.log(error);
              }
            });
          });
        });
        break;
    }
  }
}

// Creates the venv and installs the dependencies
function setupVenv(win, homeDir) {
  win.webContents.send('updateStatus', 'Setting up venv...');

  var mod = (process.platform === 'win32') ? 'python/':'python/bin/'
  var command = (process.platform === 'win32') ? 'python.exe':'python3'
  const pythonPath = path.join(homeDir, mod);

  installVenv().then(({stdout, stderr}) => {
    console.log(stdout);
    activateVenv().then(({stdout, stderr}) => {
        console.log(stdout);
    });
  });

  async function installVenv() {
    const {stdout, stderr} = await exec(`${command} -m pip install --user virtualenv`, {cwd: pythonPath});
    return {stdout, stderr};
  }

  async function activateVenv() {
    const check = `${command} ${path.join(__dirname, 'resources/py/checkVenv.py')}`;
    const activate = (process.platform === 'win32') ? 
      String.raw`.\env\Scripts\activate && ${check}`:
      'source env/bin/activate';
    const {stdout, stderr} = await exec(activate, {cwd: pythonPath});
    return {stdout, stderr};
  }

}

// Makes the local user writable folder
function checkLocalDir() {
  const homeDir = path.join(app.getPath('home'), '.belljar');
  if (!fs.existsSync(homeDir)) {
    fs.mkdirSync(homeDir, {
      recursive: true
    });
  }
  return homeDir;
}

app.on("ready", () => {
  let win = createWindow()
  // Uncomment if you want tools on launch
  win.webContents.toggleDevTools()
  
  win.webContents.once('did-finish-load', () => {
    // Make a directory to house enviornment, settings, etc.
    const homeDir = checkLocalDir();
    // Setup python for running the pipeline
    setupPython(win, homeDir);
    // Prepare depedencies
    setupVenv(win, homeDir);
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
    pythonPath: String.raw`C:\Users\Alec\anaconda3\envs\allen\python.exe`,
    scriptPath: `${__dirname}/resources/py`,
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

