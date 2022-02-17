const { app, BrowserWindow } = require('electron')

function createWindow () {
    const win = new BrowserWindow({
      width: 525,
      height: 250,
      resizable: false,
    })

    win.loadFile('pages/index.html')

    return win
}

let win;
app.on("ready", () => {
  win = createWindow()
})

app.whenReady().then(() => {
  app.on('activate', function () {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', function () {
  if (process.platform !== 'darwin') app.quit()
})