const { app, BrowserWindow } = require('electron')


function createWindow () {
    const win = new BrowserWindow({
      width: 800,
      height: 600
    })

    win.loadFile('index.html')

    return win
}

app.on("ready", () => {
  const win = createWindow()
  win.setTitle("My App")
})