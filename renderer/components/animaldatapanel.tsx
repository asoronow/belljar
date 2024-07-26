import clsx from "clsx";
import {
  QuestionMarkCircleIcon,
  PlusIcon,
  ChevronDownIcon,
  TrashIcon,
} from "@heroicons/react/24/outline";
import { useState, useEffect } from "react";
import { ProjectFile } from "../../main/helpers/projects-tools";
import { AddDataDialog } from "./add_data";

function DataDirectory({
  directory,
  project,
  animalName,
  didDelete,
}: {
  directory: {
    name: string;
    files: ProjectFile[];
  };
  project: {
    name: string;
  };
  animalName: string;
  didDelete: () => void;
}) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div className="flex flex-col items-start justify-start w-full gap-y-1 mb-1">
      <div
        className="flex flex-row items-center justify-start w-full gap-x-2 bg-black text-white font-medium p-2 w-full rounded-sm hover:shadow-md hover:scale-[1.02] cursor-pointer transition-all duration-200"
        onClick={() => setExpanded(!expanded)}
      >
        {directory.name}
        <ChevronDownIcon
          className={clsx(
            "ml-auto w-4 text-white transition-all duration-200",
            expanded ? "rotate-180" : "rotate-0"
          )}
        />
      </div>
      {expanded &&
        directory.files.map((file, index) => {
          const style = {
            "--delay": `${index * 0.01}s`,
          } as React.CSSProperties;
          return (
            <div
              key={file.name}
              className="flex flex-row items-center justify-start w-full gap-x-2 bg-gray-700 p-2 rounded-sm text-white text-xs cursor-pointer hover:bg-gray-600 opactiy-0 animate-slide-in text-ellipsis overflow-hidden select-none"
              style={style}
            >
              <div>{file.name}</div>
              <TrashIcon
                className="w-4 text-white bg-red-500 rounded-sm p-px ml-auto hover:bg-red-900"
                onClick={() => {
                  window.ipc
                    .invoke(
                      "delete-file",
                      project.name,
                      animalName,
                      directory.name,
                      file.name
                    )
                    .then((result) => {
                      if (result.success) {
                        didDelete();
                      }
                    });
                }}
              />
            </div>
          );
        })}
    </div>
  );
}

export function AnimalDataPanel({ name, meta, project, didChange }) {
  const [showAddData, setShowAddData] = useState(false);
  const [currentData, setCurrentData] = useState(null);

  const refreshData = () => {
    window.ipc.invoke("get-animal-data", project.name, name).then((result) => {
      if (result.success) {
        setCurrentData(result.data);
        didChange();
      }
    });
  };

  useEffect(() => {
    if (project && name && project.name !== "" && name !== "") {
      refreshData();
    } else {
      setCurrentData(null);
    }
  }, [project, name]);

  return (
    <div className="flex flex-col items-start justify-start w-full bg-gray-200 p-4 h-[300px] overflow-y-scroll">
      <div className="flex flex-row items-center justify-between w-full mb-4">
        <div className="flex flex-row items-center justify-start w-full gap-x-2">
          <h1 className="text-xl font-bold">Data</h1>
          <div className="group">
            <QuestionMarkCircleIcon className="w-4 text-black" />
            <span className="absolute hidden text-white -translate-y-[130%] z-10 bg-black/75 px-1 py-0.5 text-xs rounded-sm w-[200px] group-hover:block">
              Add data to an animal to run workflows on it.
            </span>
          </div>
        </div>
        {name && meta && (
          <button
            className="flex flex-row items-center justify-center p-1 bg-sky-500 rounded-sm"
            onClick={() => setShowAddData(true)}
          >
            <PlusIcon className="w-6 h-6 text-white" />
          </button>
        )}
      </div>
      {currentData && currentData.length > 0
        ? currentData.map((directory) => {
            if (directory.files.length === 0) return null;
            return (
              <DataDirectory
                key={directory.name}
                directory={directory}
                project={project}
                animalName={name}
                didDelete={refreshData}
              />
            );
          })
        : null}
      <AddDataDialog
        isOpen={showAddData}
        setIsOpen={setShowAddData}
        project={project}
        animal={name}
        didAdd={() => {
          refreshData();
        }}
      />
      {!meta && (
        <p className="text-sm m-auto">Select an animal to manage data.</p>
      )}
    </div>
  );
}
