import {
  Description,
  Dialog,
  DialogPanel,
  DialogTitle,
} from "@headlessui/react";
import { ChevronDownIcon } from "@heroicons/react/24/solid";
import { useState, useEffect, useRef } from "react";
import { AnimalMetadata } from "../pages/project";
import { Button } from "./button";
import clsx from "clsx";

const datatTypes = [
  {
    name: "Signal",
    ext: "*.tiff, *.tif, *.png, *.jpg",
    description: "Images that contain signal for analysis",
  },
  {
    name: "Background",
    ext: "*.tiff, *.tif, *.png, *.jpg",
    description: "Stained (DAPI, Nissl) background images for alignment",
  },
];
export function AddDataDialog({
  isOpen,
  setIsOpen,
  project,
  animal,
  didAdd,
}: {
  isOpen: boolean;
  setIsOpen: (value: boolean) => void;
  project: {
    [key: string]: any;
  };
  animal: string;
  didAdd: () => void;
}) {
  const [showMore, setShowMore] = useState(false);
  const [selectedType, setSelectedType] = useState(datatTypes[0]);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  return (
    <>
      <Dialog
        open={isOpen}
        onClose={() => setIsOpen(false)}
        className="relative z-50"
      >
        <div className="fixed inset-0 bg-black/30 backdrop-blur-sm" />
        <div className="fixed inset-0 flex w-screen items-center justify-center p-4 rounded-lg sm:p-6">
          <DialogPanel className="max-w-lg space-y-4 border bg-white p-12 rounded-sm transition-all w-full duration-300">
            <DialogTitle className="font-bold">Add Data</DialogTitle>
            <Description>
              Add files to an existing animal in your project.
            </Description>
            <button
              onClick={() => setShowMore(!showMore)}
              className="text-xs flex flex-row text-zinc-700"
            >
              {showMore ? "Show Help" : "Show Help"}
              <ChevronDownIcon
                className={clsx(
                  showMore ? "rotate-180" : "",
                  "ml-2 h-4 w-4 transition-all duration-200"
                )}
              />
            </button>
            {showMore && (
              <p className="text-xs text-zinc-500">
                You can use this to add files to an existing animal in your
                project. Some tools require multiple data types. Please check
                each tool for more information. Each file will be added to the
                animal with the same name as a replicate to ensure portability.
              </p>
            )}
            <div className="flex flex-col gap-4">
              <div className="flex flex-col gap-2">
                <label htmlFor="data-type" className="font-bold text-zinc-700">
                  Data Type
                </label>
                <select
                  id="data-type"
                  name="data-type"
                  value={selectedType.name}
                  onChange={(e) => {
                    setSelectedType(
                      datatTypes.find((d) => d.name === e.target.value) ||
                        datatTypes[0]
                    );
                  }}
                  className="border border-gray-300 rounded-sm p-2 focus:outline-none focus:ring-2 focus:ring-blue-600"
                >
                  {datatTypes.map((d) => (
                    <option key={d.name} value={d.name} className="text-sm">
                      {d.name}
                    </option>
                  ))}
                </select>
                {selectedType.description && (
                  <p className="text-xs text-gray-500">
                    {selectedType.description}
                  </p>
                )}
              </div>
              <div className="flex flex-col gap-2">
                <label htmlFor="data-file" className="font-bold text-zinc-700">
                  Data File
                </label>
                <input
                  type="file"
                  id="data-file"
                  name="data-file"
                  multiple
                  className="border border-gray-300 rounded-sm p-2"
                  onChange={(e) => {
                    setSelectedFiles(Array.from(e.target.files || []));
                  }}
                />
              </div>
              <Button
                type="primary"
                className="text-sm flex flex-row text-white bg-black rounded-lg w-full items-center justify-center p-2"
                onClick={() => {
                  if (selectedFiles.length > 0) {
                    const filePaths = selectedFiles.map((file) => file.path);
                    window.ipc
                      .invoke(
                        "upload-files",
                        project.name,
                        animal,
                        selectedType.name,
                        filePaths
                      )
                      .then((result) => {
                        if (result.success) {
                          setIsOpen(false);
                          didAdd();
                        } else {
                          alert(result.error);
                        }
                      });
                  }
                }}
              >
                Add
              </Button>
            </div>
          </DialogPanel>
        </div>
      </Dialog>
    </>
  );
}
