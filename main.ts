const { app, BrowserWindow } = require('electron')


function createWindow () {
    const win = new BrowserWindow({
      width: 400,
      height: 865,
      resizable: false
    })

    win.loadFile('index.html')

    return win
}

app.on("ready", () => {
  createWindow()
})

app.whenReady().then(() => {
  app.on('activate', function () {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', function () {
  if (process.platform !== 'darwin') app.quit()
})