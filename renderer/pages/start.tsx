import React, { useState, useEffect, use } from "react";
import Head from "next/head";
import { Button } from "@/components/button";
import { Input } from "@/components/input";
import { Dialog, DialogTitle } from "@headlessui/react";
import { useIpcListener } from "@/hooks/useIpcListener";
import { useRouter } from "next/router";
import { ScaleLoader } from "react-spinners";
import clsx from "clsx";

export default function HomePage() {
  const [message, setMessage] = useState("Getting things ready...");
  const [isComplete, setIsComplete] = useState(false);
  const [recentProjects, setRecentProjects] = useState<string[]>([]);
  const [newProjectName, setNewProjectName] = useState("");
  const [newProjectDescription, setNewProjectDescription] = useState("");
  const [projectSearch, setProjectSearch] = useState<string>("");
  const [projectInvalid, setProjectInvalid] = useState(false);
  const [selectedProject, setSelectedProject] = useState<string | null>(null);
  const [showCreateProject, setShowCreateProject] = useState(false);

  const router = useRouter();
  const { loaded } = router.query;

  useEffect(() => {
    if (loaded === "true") {
      setIsComplete(true);
    }
  });

  const createProject = () => {
    switch (true) {
      case newProjectName === "" || newProjectDescription === "":
        setProjectInvalid(true);
        return;
      case recentProjects.includes(newProjectName):
        setProjectInvalid(true);
        return;
      case !/^[a-zA-Z0-9-_]+$/.test(newProjectName):
        setProjectInvalid(true);
        return;
      default:
        break;
    }

    setProjectInvalid(false);

    setShowCreateProject(false);
    window.ipc
      .invoke("create-project", newProjectName, newProjectDescription)
      .then(() => {
        window.ipc.invoke("get-projects").then((result) => {
          if (result.success) {
            setRecentProjects(result.projects);
          } else {
            alert(result.error);
          }
        });
      });
  };

  const getProjects = () => {
    window.ipc.invoke("get-projects").then((result) => {
      if (result.success) {
        setRecentProjects(result.projects);
      } else {
        alert(result.error);
      }
    });
  };

  const importProject = () => {
    window.ipc.invoke("import-project").then((result) => {
      if (result.success) {
        window.ipc.invoke("get-projects").then((result) => {
          if (result.success) {
            setRecentProjects(result.projects);
          }
        });
      } else {
        alert(result.error);
      }
    });
  };

  const exportProject = () => {
    if (!selectedProject) {
      return;
    }
    window.ipc.invoke("export-project", selectedProject);
  };

  useEffect(() => {
    getProjects();
  }, []);

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
        Enter some details to create a new project.
        <div className="space-y-2">
          <p className="text-gray-400 text-sm">
            Project names can only contain letters, numbers, hyphens, and
            underscores. (e.g. my_project_1)
          </p>
          <Input
            type="text"
            placeholder="Project name"
            onChange={(e) => {
              setNewProjectName(e.target.value);
            }}
            invalid={projectInvalid}
            className={"bg-zinc-200 rounded-xl text-black"}
          />
          <Input
            type="text"
            onChange={(e) => {
              setNewProjectDescription(e.target.value);
            }}
            invalid={projectInvalid}
            placeholder="Project description"
            className={"bg-zinc-200 rounded-xl text-black"}
          />
        </div>
        <Button color="dark" onClick={() => setShowCreateProject(false)}>
          Cancel
        </Button>
        <Button color="blue" onClick={createProject}>
          Create
        </Button>
      </Dialog>
      <div className="flex flex-row items-center justify-center min-h-screen w-full text-center max-w-7xl mx-auto">
        <div className="flex flex-col items-center justify-center h-full bg-white text-center transition-all basis-1/2">
          <h1 className="text-4xl font-bold text-black">Bell Jar</h1>
          <h2 className="text-2xl font-semibold text-gray-600">v10.0.0</h2>
          {isComplete ? (
            <div className="flex flex-col mt-10 space-y-4">
              <Button
                color="dark"
                onClick={() => {
                  setShowCreateProject(true);
                }}
              >
                Create Project
              </Button>
              <Button
                color="blue"
                href={`/project?id=${selectedProject}`}
                disabled={!selectedProject}
              >
                Load Project
              </Button>
              <Button
                color="amber"
                disabled={!selectedProject}
                onClick={exportProject}
              >
                Export Project
              </Button>
              <Button color="emerald" onClick={importProject}>
                Import Project
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
                onChange={(e) => {
                  setProjectSearch(e.target.value);
                }}
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
                    className={clsx(
                      projectSearch === "" ||
                        project.toLowerCase().includes(projectSearch)
                        ? ""
                        : "hidden",
                      "w-full"
                    )}
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
