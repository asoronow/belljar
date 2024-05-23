// handlers.ts

import { ipcMain } from "electron";
import {
  createProject,
  loadProject,
  deleteProject,
  importProject,
  exportProject,
  getProjects,
  uploadFile,
  addAnimal,
} from "./projects-tools";
import { AnimalMetadata } from "../common/types";
import { ProjectDataType } from "../common/enums";

export function setupHandlers() {
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

  ipcMain.handle("import-project", async (_event) => {
    try {
      importProject();
      return { success: true };
    } catch (error) {
      return { success: false, error: error.message };
    }
  });

  ipcMain.handle("export-project", async (_event, name: string) => {
    try {
      exportProject(name).then(() => {
        return { success: true };
      });
    } catch (error) {
      return { success: false, error: error.message };
    }
  });

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
    async (
      _event,
      projectName: string,
      animalName: string,
      dataType: ProjectDataType,
      filePath: string
    ) => {
      try {
        uploadFile(projectName, animalName, dataType, filePath);
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

// Get projects
ipcMain.handle("get-projects", async () => {
  try {
    const projects = getProjects();
    return { success: true, projects };
  } catch (error) {
    return { success: false, error: error.message };
  }
});
