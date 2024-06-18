import {
  Description,
  Dialog,
  DialogPanel,
  DialogTitle,
} from "@headlessui/react";
import { ChevronDownIcon } from "@heroicons/react/24/solid";
import { useState, useEffect } from "react";
import { AnimalMetadata } from "../pages/project";
import clsx from "clsx";
export function AddAnimalDialog({
  isOpen,
  setIsOpen,
  project,
  didAdd,
}: {
  isOpen: boolean;
  setIsOpen: (value: boolean) => void;
  project: string;
  didAdd: () => void;
}) {
  const [showMore, setShowMore] = useState(false);
  const [animalName, setAnimalName] = useState("");
  const [isValid, setIsValid] = useState(false);

  useEffect(() => {
    if (animalName !== "") {
      if (animalName.match(/^[a-zA-Z0-9-_]+$/) && !animalName.includes(" ")) {
        setIsValid(true);
      } else {
        setIsValid(false);
      }
    } else {
      setIsValid(false);
    }
  }, [animalName]);

  return (
    <>
      <Dialog
        open={isOpen}
        onClose={() => setIsOpen(false)}
        className="relative z-50"
      >
        <div className="fixed inset-0 bg-black/30 backdrop-blur-sm" />
        <div className="fixed inset-0 flex w-screen items-center justify-center p-4 rounded-lg sm:p-6">
          <DialogPanel className="max-w-lg space-y-4 border bg-white p-12 rounded-lg transition-all w-full duration-300">
            <DialogTitle className="font-bold">Add Animal</DialogTitle>
            <Description>
              Enter a name for a new experimental animal to add to your project.
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
            {showMore ? (
              <p className="text-xs text-zinc-500">
                Adding an animal to your project will allow you to use it in
                analysis workflows. You must upload the required data files for
                each animal.
              </p>
            ) : null}
            <div className="flex flex-col gap-4">
              <input
                type="text"
                placeholder="Animal Name"
                onChange={(e) => {
                  setAnimalName(e.target.value);
                }}
                className={clsx(
                  isValid ? "border-zinc-500" : "border-red-500",
                  "w-full border rounded-lg p-3"
                )}
              />
              <button
                type="submit"
                className="w-full bg-zinc-900 text-white rounded-lg p-2"
                onClick={() => {
                  if (isValid) {
                    let emptyAnimal: AnimalMetadata = {
                      hasCellDetectionData: false,
                      hasAlignmentData: false,
                      cellDetectionRun: false,
                      alignmentRun: false,
                    };
                    window.ipc
                      .invoke("add-animal", project, animalName, emptyAnimal)
                      .then((result) => {
                        if (result.success) {
                          didAdd();
                          setIsOpen(false);
                        } else {
                          alert(result.error);
                        }
                      });
                  }
                }}
              >
                Add
              </button>
            </div>
          </DialogPanel>
        </div>
      </Dialog>
    </>
  );
}
