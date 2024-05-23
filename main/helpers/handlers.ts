// handlers.ts

import { ipcMain, BrowserWindow } from "electron";
import {
  createProject,
  loadProject,
  deleteProject,
  uploadFile,
  addAnimal,
} from "./projects";
import { AnimalMetadata } from "./types";

export function setupHandlers(mainWindow: BrowserWindow) {
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

  ipcMain.handle("load-project", async (_event, name: string) => {
    try {
      const project = loadProject(name);
      return { success: true, project };
    } catch (error) {
      return { success: false, error: error.message };
    }
  });

  ipcMain.handle("delete-project", async (_event, name: string) => {
    try {
      deleteProject(name);
      return { success: true };
    } catch (error) {
      return { success: false, error: error.message };
    }
  });

  ipcMain.handle(
    "upload-file",
    async (_event, projectName: string, filePath: string) => {
      try {
        uploadFile(projectName, filePath);
        return { success: true };
      } catch (error) {
        return { success: false, error: error.message };
      }
    }
  );

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
}
