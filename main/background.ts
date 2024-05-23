import path from "path";
import { app } from "electron";
import serve from "electron-serve";
import { createWindow, performSetup } from "./helpers";
import { setupHandlers } from "./helpers/handlers";

const isProd = process.env.NODE_ENV === "production";
let mainWindow: Electron.BrowserWindow;
if (isProd) {
  serve({ directory: "app" });
} else {
  app.setPath("userData", `${app.getPath("userData")} (development)`);
}

(async () => {
  await app.whenReady();

  mainWindow = createWindow("main", {
    width: 1000,
    height: 600,
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
    },
  });

  if (isProd) {
    await mainWindow.loadURL("app://./start");
  } else {
    const port = process.argv[2];
    await mainWindow.loadURL(`http://localhost:${port}/start`);
    mainWindow.webContents.openDevTools();
  }

  const pythonScriptsPath: string = isProd
    ? path.join(process.resourcesPath, "py")
    : path.join(__dirname, "../py");

  setupHandlers(mainWindow);
  performSetup(mainWindow, pythonScriptsPath);
})();

app.on("window-all-closed", () => {
  app.quit();
});
