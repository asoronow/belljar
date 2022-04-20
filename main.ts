const { app, BrowserWindow, ipcMain, dialog } = require('electron')

function createWindow () {
    const win = new BrowserWindow({
      width: 1000,
      height: 525,
      resizable: true,
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

ipcMain.on('openDialog', function(event, data){
  let window = BrowserWindow.getFocusedWindow()
  dialog.showOpenDialog(window, {
    properties: ['openDirectory']
  }).then(result => {
    if (!result.canceled) {
      console.log(result.filePaths)
      event.sender.send('returnPath', [result.filePaths[0], data])
    }
  }).catch(err => {
    console.log(err)
  });
});