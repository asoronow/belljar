// handlers.ts
import { ipcMain, IpcMainEvent } from "electron";
import {
  createProject,
  loadProject,
  deleteAnimal,
  deleteProject,
  selectDirectory,
  getAnimalDataDirectory,
  deleteAnimalDataDirectory,
  deleteFile,
  getAnimalData,
  importProject,
  exportProject,
  getProjects,
  uploadFile,
  addAnimal,
} from "./projects-tools";
import { AnimalMetadata } from "../common/types";
import { PythonShell, PythonShellError, Options } from "python-shell";
import path from "path";
import os from "os";
interface PythonScriptOptions {
  scriptName: string;
  args: string[];
  resultEvent: string;
  killEvent: string;
}

function runPythonScript(
  event: IpcMainEvent,
  options: PythonScriptOptions,
  pyScriptsPath: string
) {
  const { scriptName, args, resultEvent, killEvent } = options;
  const envPythonPath = path.join(os.homedir(), ".belljar", "benv");
  const platform = os.platform();
  const pyCommand =
    platform === "win32"
      ? `${envPythonPath}/Scripts/python`
      : `${envPythonPath}/bin/python`;

  const scriptOptions: Options = {
    mode: "text",
    pythonPath: pyCommand,
    scriptPath: pyScriptsPath,
    args,
  };

  const pyshell = new PythonShell(scriptName, scriptOptions);
  let total = 0;
  let current = 0;

  pyshell.on("stderr", (stderr: string) => {
    console.error(stderr);
  });

  pyshell.on("message", (message: string) => {
    if (total === 0) {
      total = Number(message);
    } else if (message === "Done!") {
      pyshell.end((err: PythonShellError, code: number, signal: string) => {
        if (err) throw err;
        event.sender.send(resultEvent);
        ipcMain.removeAllListeners(killEvent);
      });
    } else {
      current++;
      event.sender.send("updateLoad", [
        Math.round((current / total) * 100),
        message,
      ]);
    }
  });

  ipcMain.once(killEvent, () => {
    pyshell.kill();
  });
}

/**
 * Sets up event handlers for various IPC events.
 *
 * @param {string} pyScriptsPath - The path to the Python scripts.
 * @return {void}
 */
export function setupHandlers(pyScriptsPath: string) {
  const homeDir = os.homedir();
  const isProd = process.env.NODE_ENV === "production";
  const resourceDir = isProd
    ? path.join(process.resourcesPath, "py")
    : path.join(__dirname, "../py");

  /**
   * Handler for the "create-project" event.
   * Creates a new Bell jar project.
   * @param {Electron.IpcMainInvokeEvent} _event - The event object.
   * @param {string} name - The name of the new project.
   * @param {string} description - The description of the new project.
   * @return {Promise<{ success: boolean; error?: string; }>} - A promise that resolves to an object indicating success or failure.
   */
  ipcMain.handle(
    "create-project",
    async (_event, name: string, description: string) => {
      try {
        createProject(name, description);
        return { success: true };
      } catch (error) {
        return { success: false, error: error.message };
      }
    }
  );

  /**
   * Handler for the "import-project" event.
   *  Imports a Bell jar project.
   * @param {Electron.IpcMainInvokeEvent} _event - The event object.
   *
   */
  ipcMain.handle("import-project", async (_event) => {
    try {
      importProject();
      return { success: true };
    } catch (error) {
      return { success: false, error: error.message };
    }
  });

  /**
   * Handler for the "export-project" event.
   * Exports a project by name.
   * @param {Electron.IpcMainInvokeEvent} _event - The event object.
   * @param {string} name - The name of the project to export.
   * @return {Promise<{ success: boolean; error?: string; }>} - A promise that resolves to an object indicating success or failure.
   */
  ipcMain.handle("export-project", async (_event, name: string) => {
    try {
      exportProject(name).then(() => {
        return { success: true };
      });
    } catch (error) {
      return { success: false, error: error.message };
    }
  });

  /**
   * Handler for the "load-project" event.
   * Loads a project by name.
   *
   * @param {Electron.IpcMainInvokeEvent} _event - The event object.
   * @param {string} name - The name of the project to load.
   * @return {Promise<{ success: boolean; error?: string; project?: ProjectMetadata; }>} - A promise that resolves to an object indicating success or failure, along with an optional project object.
   */
  ipcMain.handle("load-project", async (_event, name: string) => {
    try {
      const project = loadProject(name);
      return { success: true, project };
    } catch (error) {
      return { success: false, error: error.message };
    }
  });

  /**
   * Handler for the "delete-project" event.
   * Deletes a project by name.
   *
   * @param {Electron.IpcMainInvokeEvent} _event - The event object.
   * @param {string} name - The name of the project to delete.
   * @return {Promise<{ success: boolean; error?: string; }>} - A promise that resolves to an object indicating success or failure, along with an optional error message.
   */
  ipcMain.handle("delete-project", async (_event, name: string) => {
    try {
      deleteProject(name);
      return { success: true };
    } catch (error) {
      return { success: false, error: error.message };
    }
  });

  /**
   * Handler for uploading files to a project.
   *
   * @param {IpcMainEvent} _event - The IPC event.
   * @param {string} projectName - The name of the project.
   * @param {string} animalName - The name of the animal.
   * @param {string} dataType - The type of data.
   * @param {string[]} filePaths - The paths of the files to upload.
   * @returns {Promise<{ success: boolean, error?: string }>} - A promise that resolves to an object indicating success or failure.
   */
  ipcMain.handle(
    "upload-files",
    async (
      _event,
      projectName: string,
      animalName: string,
      dataType: string,
      filePaths: [string]
    ) => {
      try {
        uploadFile(projectName, animalName, dataType, filePaths);
        return { success: true };
      } catch (error) {
        return { success: false, error: error.message };
      }
    }
  );

  /**
   * Handler for getting animal data.
   *
   * @param {IpcMainEvent} _event - The IPC event.
   * @param {string} projectName - The name of the project.
   * @param {string} animalName - The name of the animal.
   * @returns {Promise<{ success: boolean, data?: AnimalData, error?: string }>} - A promise that resolves to an object indicating success or failure, along with an optional animal data object.
   */
  ipcMain.handle(
    "get-animal-data",
    async (_event, projectName: string, animalName: string) => {
      try {
        const animalData = getAnimalData(projectName, animalName);
        return { success: true, data: animalData };
      } catch (error) {
        return { success: false, error: error.message };
      }
    }
  );

  /**
   * Handler for deleting an animal.
   *
   * @param {IpcMainEvent} _event - The IPC event.
   * @param {string} projectName - The name of the project.
   * @param {string} animalName - The name of the animal.
   * @returns {Promise<{ success: boolean, error?: string }>} - A promise that resolves to an object indicating success or failure.
   */
  ipcMain.handle(
    "delete-animal",
    async (_event, projectName: string, animalName: string) => {
      try {
        deleteAnimal(projectName, animalName);
        return { success: true };
      } catch (error) {
        return { success: false, error: error.message };
      }
    }
  );

  /**
   * Handler for deleting a file.
   * @param {IpcMainEvent} _event - The IPC event.
   * @param {string} projectName - The name of the project.
   * @param {string} animalName - The name of the animal.
   * @param {string} dataType - The type of data.
   * @param {string} filePath - The path of the file to delete.
   */
  ipcMain.handle(
    "delete-file",
    async (
      _event,
      projectName: string,
      animalName: string,
      dataType: string,
      fileName: string
    ) => {
      try {
        deleteFile(projectName, animalName, dataType, fileName);
        return { success: true };
      } catch (error) {
        return { success: false, error: error.message };
      }
    }
  );

  /**
   * Handler for adding an animal.
   * @param {IpcMainEvent} _event - The IPC event.
   * @param {string} projectName - The name of the project.
   * @param {string} animalName - The name of the animal.
   * @param {AnimalMetadata} animalData - The metadata of the animal.
   * @returns {Promise<{ success: boolean, error?: string }>} - A promise that resolves to an object indicating success or failure.
   */
  ipcMain.handle(
    "add-animal",
    async (
      _event,
      projectName: string,
      animalName: string,
      animalData: AnimalMetadata
    ) => {
      try {
        addAnimal(projectName, animalName, animalData);
        return { success: true };
      } catch (error) {
        return { success: false, error: error.message };
      }
    }
  );

  /**
   * Handler for getting project data.
   * @param {IpcMainEvent} _event - The IPC event.
   * @returns {Promise<{ success: boolean, projects?: ProjectData, error?: string }>} - A promise that resolves to an object indicating success or failure, along with an optional project data object.
   */
  ipcMain.handle("get-projects", async () => {
    try {
      const projects = getProjects();
      return { success: true, projects: projects };
    } catch (error) {
      return { success: false, error: error.message };
    }
  });

  ipcMain.handle(
    "get-animal-data-directory",
    async (
      _event: IpcMainEvent,
      projectName: string,
      animalName: string,
      dataType: string
    ) => {
      try {
        // Gets a symlinked temp dir to the animal background data
        const tempDir = await getAnimalDataDirectory(
          projectName,
          animalName,
          dataType
        );
        return { success: true, directory: tempDir };
      } catch (error) {
        return { success: false, error: error.message };
      }
    }
  );

  ipcMain.handle(
    "delete-animal-data-directory",
    async (
      event: IpcMainEvent,
      projectName: string,
      animalName: string,
      directory: string
    ) => {
      try {
        deleteAnimalDataDirectory(projectName, animalName, directory);
        return { success: true };
      } catch (error) {
        return { success: false, error: error.message };
      }
    }
  );

  ipcMain.handle("get-directory", async () => {
    const selectedDirectory = selectDirectory();
    if (!selectedDirectory) {
      return { success: false, error: "No directory selected" };
    }
    return { success: true, directory: selectedDirectory };
  });

  ipcMain.on("runMax", (event: IpcMainEvent, data: any[]) => {
    const args = [
      `-o ${data[1]}`,
      `-i ${data[0]}`,
      `-d ${data[2]}`,
      `-t ${data[3]}`,
      "-g False",
    ];

    runPythonScript(
      event,
      {
        scriptName: "max.py",
        args,
        resultEvent: "maxResult",
        killEvent: "killMax",
      },
      pyScriptsPath
    );
  });

  ipcMain.on("runAdjust", (event: IpcMainEvent, data: any[]) => {
    const structPath = path.join(resourceDir, "csv/structure_map.pkl");
    const args = [`-i ${data[0]}`, `-s ${structPath}`, `-a ${data[1]}`];

    runPythonScript(
      event,
      {
        scriptName: "adjust.py",
        args,
        resultEvent: "adjustResult",
        killEvent: "killAdjust",
      },
      pyScriptsPath
    );
  });

  ipcMain.handle("runAlign", (event: IpcMainEvent, data: any[]) => {
    const modelPath = path.join(homeDir, ".belljar", "models/predictor.pt");
    const nrrdPath = path.join(homeDir, ".belljar", "nrrd");
    const mapPath = path.join(resourceDir, "csv/structure_map.pkl");
    const args = [
      `-o ${data[1]}`,
      `-i ${data[0]}`,
      `-w ${data[2]}`,
      `-a ${data[3]}`,
      `-m ${modelPath}`,
      `-n ${nrrdPath}`,
      `-c ${mapPath}`,
      `-l ${data[4]}`,
    ];

    runPythonScript(
      event,
      {
        scriptName: "map.py",
        args,
        resultEvent: "alignResult",
        killEvent: "killAlign",
      },
      pyScriptsPath
    );
  });

  ipcMain.on("runIntensity", (event: IpcMainEvent, data: any[]) => {
    const structPath = path.join(resourceDir, "csv/structure_map.pkl");
    const args = [
      `-i ${data[0]}`,
      `-o ${data[1]}`,
      `-a ${data[2]}`,
      `-w ${data[3]}`,
      `-m ${structPath}`,
    ];

    runPythonScript(
      event,
      {
        scriptName: "region.py",
        args,
        resultEvent: "intensityResult",
        killEvent: "killIntensity",
      },
      pyScriptsPath
    );
  });

  ipcMain.on("runCount", (event: IpcMainEvent, data: any[]) => {
    const structPath = path.join(resourceDir, "csv/structure_map.pkl");
    const customArgs = [
      `-p ${data[0]}`,
      `-a ${data[1]}`,
      `-o ${data[2]}`,
      `-m ${structPath}`,
    ];
    if (data[3]) {
      customArgs.push(`--layers`);
    }

    runPythonScript(
      event,
      {
        scriptName: "count.py",
        args: customArgs,
        resultEvent: "countResult",
        killEvent: "killCount",
      },
      pyScriptsPath
    );
  });

  ipcMain.on("runCollate", (event: IpcMainEvent, data: any[]) => {
    const args = [
      `-o ${data[1]}`,
      `-i ${data[0]}`,
      `-r ${data[2]}`,
      `-s ${path.join(resourceDir, "csv/structure_map.pkl")}`,
      "-g False",
    ];

    runPythonScript(
      event,
      {
        scriptName: "collate.py",
        args,
        resultEvent: "collateResult",
        killEvent: "killCollate",
      },
      pyScriptsPath
    );
  });

  ipcMain.on("runSharpen", (event: IpcMainEvent, data: any[]) => {
    const customArgs = [
      `-o ${data[1]}`,
      `-i ${data[0]}`,
      `-r ${data[2]}`,
      `-a ${data[3]}`,
    ];
    if (data[4]) {
      customArgs.push(`--equalize`);
    }

    runPythonScript(
      event,
      {
        scriptName: "sharpen.py",
        args: customArgs,
        resultEvent: "sharpenResult",
        killEvent: "killSharpen",
      },
      pyScriptsPath
    );
  });

  ipcMain.on("runDetection", (event: IpcMainEvent, data: any[]) => {
    const models: { [key: string]: string } = {
      somata: "models/chaosdruid.pt",
      nuclei: "models/ankou.pt",
    };

    let modelPath = path.join(homeDir, models[data[6]]);
    if (data[4].length > 0) {
      modelPath = data[4];
    }

    const customArgs = [
      `-i ${data[0]}`,
      `-o ${data[1]}`,
      `-c ${data[2]}`,
      `-t ${data[3]}`,
      `-a ${data[7]}`,
      `-m ${modelPath}`,
    ];
    if (data[5]) {
      customArgs.push(`--multichannel`);
    }

    runPythonScript(
      event,
      {
        scriptName: "find_neurons.py",
        args: customArgs,
        resultEvent: "detectResult",
        killEvent: "killDetect",
      },
      pyScriptsPath
    );
  });
}
