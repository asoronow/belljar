// handlers.ts
import { ipcMain, IpcMainEvent } from "electron";
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

export function setupHandlers(pyScriptsPath: string) {
	const homeDir = os.homedir();
	const isProd = process.env.NODE_ENV === "production";
	const resourceDir = isProd
		? path.join(process.resourcesPath, "py")
		: path.join(__dirname, "../py");

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
	// Get projects
	ipcMain.handle("get-projects", async () => {
		try {
			const projects = getProjects();
			return { success: true, projects };
		} catch (error) {
			return { success: false, error: error.message };
		}
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

	ipcMain.on("runAlign", (event: IpcMainEvent, data: any[]) => {
		const modelPath = path.join(homeDir, "models/predictor.pt");
		const nrrdPath = path.join(homeDir, "nrrd");
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
