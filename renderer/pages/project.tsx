import React, { useState, useEffect } from "react";
import { useRouter } from "next/router";
import { AnimalDataPanel } from "@/components/animaldatapanel";
import { AnimalsPanel } from "@/components/animalspanel";
import { ToolsPanel } from "@/components/toolspanel";
import {
  ChevronLeftIcon,
  TrashIcon,
  QuestionMarkCircleIcon,
  PlusIcon,
} from "@heroicons/react/24/solid";
import { AddAnimalDialog } from "../components/add_animal";
import Link from "next/link";

export interface AnimalMetadata {
  hasCellDetectionData: boolean;
  hasAlignmentData: boolean;
  cellDetectionRun: boolean;
  alignmentRun: boolean;
}

export interface ProjectMetadata {
  name: string;
  createdAt: string;
  lastModified: string;
  description: string;
  animals: Record<string, AnimalMetadata>;
}

export default function ProjectPage() {
  const router = useRouter();
  const { id } = router.query;
  const [project, setProject] = useState(null);
  const [selectedAnimal, setSelectedAnimal] = useState(null);
  const [animals, setAnimals] = useState({});
  const [selectedData, setSelectedData] = useState(null);
  const [taskQueue, setTaskQueue] = useState([]);

  const loadProject = (id) => {
    window.ipc.invoke("load-project", id).then((result) => {
      if (result.success) {
        setProject(result.project);
        setAnimals(result.project.animals);
      } else {
        alert(result.error);
      }
    });
  };

  useEffect(() => {
    if (id) {
      loadProject(id);
    }
  }, [id]);

  return (
    <React.Fragment>
      <div className="flex flex-col items-center justify-start w-full min-h-screen p-6 max-w-7xl mx-auto">
        {project ? (
          <>
            <div className="flex flex-row items-center justify-between w-full">
              <Link
                href="/start?loaded=true"
                className="text-blue-500 flex flex-row w-full"
              >
                <div className="p-2 m-2 bg-black rounded-sm">
                  <ChevronLeftIcon className="w-6 h-6 text-white" />
                </div>
              </Link>
              <div className="flex flex-col items-center justify-center text-center w-full">
                <h1 className="text-lg font-bold">{project.name}</h1>
                <h2 className="text-sm text-gray-400">{project.description}</h2>
              </div>
              <div className="flex flex-row items-center justify-end w-full">
                <button
                  className="flex flex-row items-center justify-center p-2 m-2 bg-red-500 rounded-sm"
                  onClick={() => {
                    // confirm delete
                    const confirm = window.confirm(
                      "Are you sure you want to delete this project? All data will be lost FOREVER."
                    );
                    if (!confirm) {
                      return;
                    }
                    window.ipc.invoke("delete-project", project.name);
                    router.push("/start?loaded=true");
                  }}
                >
                  <TrashIcon className="w-6 h-6 text-white" />
                </button>
              </div>
            </div>
            <div className="grid grid-cols-3 gap-4 w-full mt-10">
              <AnimalsPanel
                setSelectedAnimal={setSelectedAnimal}
                selectedAnimal={selectedAnimal}
                project={project}
                animals={animals}
                didAdd={() => {
                  loadProject(id);
                }}
              />
              <AnimalDataPanel
                name={selectedAnimal}
                meta={animals[selectedAnimal]}
                project={project}
              />
              <ToolsPanel />
            </div>
          </>
        ) : (
          <div>Loading...</div>
        )}
      </div>
    </React.Fragment>
  );
}
