import React, { useState, useEffect, use } from "react";
import Head from "next/head";
import { Button } from "@/components/button";
import { Input } from "@/components/input";
import { CreateProjectDialog } from "@/components/create_project";
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
      <div className="flex flex-row items-center justify-center min-h-screen w-full text-center max-w-3xl mx-auto">
        <div className="flex flex-col items-center justify-center h-full bg-white text-center transition-all basis-1/2">
          <h1 className="text-4xl font-bold text-black">Bell Jar</h1>
          <h2 className="text-2xl font-semibold text-gray-600">v10.0.0</h2>
          {isComplete ? (
            <div className="flex flex-col mt-10 space-y-4">
              <Button
                type="primary"
                onClick={() => {
                  setShowCreateProject(true);
                }}
              >
                Create Project
              </Button>
              <Button
                type={selectedProject ? "selected" : "secondary"}
                href={`/project?id=${selectedProject}`}
                disabled={!selectedProject}
              >
                Load Project
              </Button>
              <Button
                type={selectedProject ? "success" : "secondary"}
                disabled={!selectedProject}
                onClick={exportProject}
              >
                Export Project
              </Button>
              <Button type="warning" onClick={importProject}>
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
          <div className="flex flex-col items-start justify-start h-96 transition-all p-4 gap-y-2 bg-gray-200 basis-1/2 overflow-y-scroll rounded-sm">
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
                className={"bg-white text-black"}
              />
            </div>
            {recentProjects.length > 0 ? (
              <>
                {recentProjects.map((project) => (
                  <Button
                    type={
                      project === selectedProject ? "selected" : "secondary"
                    }
                    key={project}
                    onClick={() => setSelectedProject(project)}
                    className={clsx(
                      projectSearch === "" ||
                        project.toLowerCase().includes(projectSearch)
                        ? ""
                        : "hidden"
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
        <CreateProjectDialog
          showCreateProject={showCreateProject}
          setShowCreateProject={setShowCreateProject}
          createProject={createProject}
          projectInvalid={projectInvalid}
          setNewProjectDescription={setNewProjectDescription}
          setNewProjectName={setNewProjectName}
        />
      </div>
    </React.Fragment>
  );
}
