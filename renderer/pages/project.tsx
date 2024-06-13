import React, { useState, useEffect } from "react";
import { useRouter } from "next/router";
import {
  ChevronLeftIcon,
  TrashIcon,
  QuestionMarkCircleIcon,
  PlusIcon,
} from "@heroicons/react/24/solid";
import { AddAnimalDialog } from "../components/add_animal";
import { AddDataDialog } from "../components/add_data";
import Link from "next/link";
import clsx from "clsx";
import { Description } from "../components/fieldset";
export interface AnimalMetadata {
  hasCellDetectionData: boolean;
  hasAlignmentData: boolean;
  cellDetectionRun: boolean;
  alignmentRun: boolean;
}

const dataTypes = [
  {
    name: "Signal Data",
    description: "Images with signals to be detected, e.g. cells.",
    pathName: "signal-data",
  },
  {
    name: "Background Data",
    description: "Images of sections with background to stain for alignment.",
    pathName: "background-data",
  },
];

export interface ProjectMetadata {
  name: string;
  createdAt: string;
  lastModified: string;
  description: string;
  animals: Record<string, AnimalMetadata>;
}

function Pill({ className, children, tooltip = "" }) {
  return (
    <div
      className={clsx(
        className,
        "inline-flex items-center px-2 py-0.5 text-xs rounded-full font-medium group"
      )}
    >
      <span
        className={clsx(
          "absolute hidden inset-x-0 -translate-y-5 z-10 bg-black/50 px-1 py-0.5 text-xs rounded-full w-fit group-hover:block"
        )}
      >
        {tooltip}
      </span>
      {children}
    </div>
  );
}

export default function ProjectPage() {
  const router = useRouter();
  const { id } = router.query;
  const [project, setProject] = useState(null);
  const [selectedAnimal, setSelectedAnimal] = useState(null);
  const [showAddAnimal, setShowAddAnimal] = useState(false);
  const [showAddData, setShowAddData] = useState(false);
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
                <div className="p-2 m-2 bg-black rounded-xl">
                  <ChevronLeftIcon className="w-6 h-6 text-white" />
                </div>
              </Link>
              <div className="flex flex-col items-center justify-center w-full">
                <h1 className="text-xl font-bold">{project.name}</h1>
                <h2 className="text-lg text-gray-400">{project.description}</h2>
              </div>
              <div className="flex flex-row items-center justify-end w-full">
                <button
                  className="flex flex-row items-center justify-center p-2 m-2 bg-red-500 rounded-xl"
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
              <div className="flex flex-col items-start gap-y-2 justify-start bg-gray-100 rounded-y-3xl rounded-l-3xl border p-4 h-[300px] overflow-x-hidden overflow-y-scroll">
                <div
                  className={
                    "flex flex-row items-center justify-between w-full mb-4"
                  }
                >
                  <div className="flex flex-row items-center justify-start w-full gap-x-2">
                    <h1 className="text-xl font-bold">Animals</h1>
                    <div className="group">
                      <QuestionMarkCircleIcon className="w-6 h-6 text-gray-900" />
                      <span className="absolute hidden text-white -translate-y-12 z-10 bg-black/50 px-1 py-0.5 text-xs rounded-full w-fit group-hover:block">
                        Each animal holds its own experimental data. Create an
                        animal to get started.
                      </span>
                    </div>
                  </div>
                  <button
                    className="flex flex-row items-center justify-center p-1 bg-blue-500 rounded-lg"
                    onClick={() => {
                      setSelectedAnimal(null);
                      setShowAddAnimal(true);
                    }}
                  >
                    <PlusIcon className="w-6 h-6 text-white" />
                  </button>
                </div>
                {Object.keys(animals).map((animal) => (
                  <div
                    key={animal}
                    className={clsx(
                      selectedAnimal === animal ? "bg-amber-500" : "bg-black",
                      "flex p-2 flex-row items-center justify-between rounded-lg text-white"
                    )}
                    onClick={() => {
                      if (selectedAnimal === animal) {
                        setSelectedAnimal(null);
                        return;
                      }
                      setSelectedAnimal(animal);
                    }}
                  >
                    <p className="text-lg font-bold">{animal}</p>
                    <div className="flex flex-row flex-wrap items-center justify-end w-full gap-2">
                      <Pill
                        tooltip={
                          "Data for perfoming cell detection. Red indicates no data. Green indicates data."
                        }
                        className={
                          animals[animal].hasCellDetectionData
                            ? "bg-green-500"
                            : "bg-red-500"
                        }
                      >
                        Signal
                      </Pill>
                      <Pill
                        tooltip={
                          "Data for perfoming alignment. Red indicates no data. Green indicates data."
                        }
                        className={
                          animals[animal].hasAlignmentData
                            ? "bg-green-500"
                            : "bg-red-500"
                        }
                      >
                        Background
                      </Pill>
                      <Pill
                        tooltip={
                          "Has the animal been aligned. Red indicates no. Green indicates yes."
                        }
                        className={
                          animals[animal].alignmentRun
                            ? "bg-green-500"
                            : "bg-red-500"
                        }
                      >
                        Aligned
                      </Pill>
                      <Pill
                        tooltip={
                          "Has the signal data been detected. Red indicates no. Green indicates yes."
                        }
                        className={
                          animals[animal].cellDetectionRun
                            ? "bg-green-500"
                            : "bg-red-500"
                        }
                      >
                        Detected
                      </Pill>
                    </div>
                  </div>
                ))}
                <AddAnimalDialog
                  isOpen={showAddAnimal}
                  setIsOpen={setShowAddAnimal}
                  project={project.name}
                  didAdd={() => {
                    // reload projects
                    loadProject(id);
                  }}
                />
              </div>
              <div className="flex flex-col items-start justify-start w-full bg-gray-100 rounded-y-3xl rounded-l-3xl border p-4 h-[300px] overflow-y-scroll">
                <h1 className="text-xl font-bold">Data</h1>
                {selectedAnimal ? (
                  <>
                    <div className="flex flex-row items-center justify-between w-full mb-4">
                      <h1 className="text-xl font-bold">{selectedAnimal}</h1>
                      <button
                        className="flex flex-row items-center justify-center p-1 bg-blue-500 rounded-lg"
                        onClick={() => {
                          setSelectedData(null);
                          setShowAddData(true);
                        }}
                      >
                        <PlusIcon className="w-6 h-6 text-white" />
                      </button>
                    </div>
                    <AddDataDialog
                      isOpen={showAddData}
                      setIsOpen={setShowAddData}
                      project={project.name}
                      animal={selectedAnimal}
                      didAdd={() => {
                        // reload projects
                        loadProject(id);
                      }}
                    />
                  </>
                ) : (
                  <p className="text-sm m-auto">
                    Select an animal to manage data.
                  </p>
                )}
              </div>
              <div className="flex flex-col items-start justify-start w-full bg-gray-100 rounded-y-3xl rounded-l-3xl border p-4 h-[300px] overflow-y-scroll">
                <h1 className="text-xl font-bold">Tools</h1>
              </div>
            </div>
          </>
        ) : (
          <div>Loading...</div>
        )}
      </div>
    </React.Fragment>
  );
}
