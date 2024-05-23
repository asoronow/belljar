import React, { useState } from "react";
import Head from "next/head";
import { Button } from "../components/button";
import { Input } from "../components/input";
import {
  Dialog,
  DialogActions,
  DialogBody,
  DialogDescription,
  DialogTitle,
} from "../components/dialog";
import { useIpcListener } from "../hooks/useIpcListener";
import { ScaleLoader } from "react-spinners";
import { ipcRenderer } from "electron";
export default function HomePage() {
  const [message, setMessage] = useState("Getting things ready...");
  const [isComplete, setIsComplete] = useState(false);
  const [recentProjects, setRecentProjects] = useState<string[]>([]);
  const [selectedProject, setSelectedProject] = useState<string | null>(null);
  const [showCreateProject, setShowCreateProject] = useState(false);

  useIpcListener("setup-progress", (message: string) => {
    if (message === "Setup complete!") {
      setIsComplete(true);
    }
    setMessage(message);
  });

  return (
    <React.Fragment>
      <Head>
        <title>Bell Jar</title>
      </Head>
      <Dialog open={showCreateProject} onClose={setShowCreateProject}>
        <DialogTitle>Create Project</DialogTitle>
        <DialogDescription>
          Enter some details to create a new project.
        </DialogDescription>
        <DialogBody className="space-y-2">
          <Input
            type="text"
            placeholder="Project name"
            className={"bg-zinc-200 rounded-xl text-black"}
          />
          <Input
            type="text"
            placeholder="Project description"
            className={"bg-zinc-200 rounded-xl text-black"}
          />
        </DialogBody>
        <DialogActions>
          <Button color="dark" onClick={() => setShowCreateProject(false)}>
            Cancel
          </Button>
          <Button color="blue">Create</Button>
        </DialogActions>
      </Dialog>
      <div className="flex flex-row items-center justify-center min-h-screen w-full text-center">
        <div className="flex flex-col items-center justify-center h-full bg-white text-center transition-all basis-1/2">
          <h1 className="text-4xl font-bold text-black">Bell Jar</h1>
          <h2 className="text-2xl font-semibold text-gray-600">v10.0.0</h2>
          {isComplete ? (
            <div className="flex flex-col mt-10 space-y-4">
              <Button
                color="blue"
                onClick={() => {
                  setShowCreateProject(true);
                }}
              >
                Create Project
              </Button>
              <Button color="dark" href="/project?id=">
                Load Project
              </Button>
            </div>
          ) : (
            <>
              <ScaleLoader height={15} color="#000" />
              <p className="text-lg text-gray-500 my-4">{message}</p>
            </>
          )}
        </div>
        {isComplete ? (
          <div className="flex flex-col items-start justify-start h-96 transition-all rounded-y-3xl rounded-l-3xl border p-4 gap-y-2 bg-gray-100 basis-1/2 overflow-y-scroll">
            <div className="flex flex-row items-center justify-between w-full">
              <h3 className="text-lg font-semibold text-black w-full text-left">
                Projects
              </h3>
              <Input
                type="text"
                placeholder="Search projects"
                className={"bg-white rounded-3xl text-black"}
              />
            </div>
            {recentProjects.length > 0 ? (
              <>
                {recentProjects.map((project) => (
                  <Button
                    key={project}
                    color={selectedProject === project ? "amber" : "dark"}
                    onClick={() => setSelectedProject(project)}
                    className={"w-full"}
                  >
                    {project}
                  </Button>
                ))}
              </>
            ) : (
              <div className="w-full h-96 flex items-center justify-center">
                <p className="text-md text-gray-500 w-full">
                  No projects yet. Create one to get started!
                </p>
              </div>
            )}
          </div>
        ) : null}
      </div>
    </React.Fragment>
  );
}
