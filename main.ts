const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const {PythonShell} = require('python-shell');
const path = require('path');

function createWindow () {
    const win = new BrowserWindow({
      width: 1000,
      height: 525,
      resizable: true,
      autoHideMenuBar: true,
      webPreferences: {nodeIntegration: true, contextIsolation: false }
    })

    win.loadFile('pages/index.html')

    return win
}

app.on("ready", () => {
  let win = createWindow()
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

