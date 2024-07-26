import {
  PlusIcon,
  QuestionMarkCircleIcon,
  TrashIcon,
} from "@heroicons/react/24/outline";
import { AddAnimalDialog } from "./add_animal";
import { AnimalCard } from "./animalcard";
import { useState } from "react";
export function AnimalsPanel({
  setSelectedAnimal,
  selectedAnimal,
  project,
  animals,
  didAdd,
}: {
  setSelectedAnimal: (value: string | null) => void;
  selectedAnimal: string | null;
  project: {
    [key: string]: any;
  };
  animals: Record<string, any>;
  didAdd: () => void;
}) {
  const [showAddAnimal, setShowAddAnimal] = useState(false);
  return (
    <div className="flex flex-col items-start gap-y-2 justify-start bg-gray-200 border p-4 h-[300px] overflow-x-hidden overflow-y-scroll">
      <div
        className={
          "flex flex-row items-center justify-between w-full mb-2 gap-x-2"
        }
      >
        <div className="flex flex-row items-center justify-start w-full gap-x-2">
          <h1 className="text-xl font-bold">Animals</h1>
          <div className="group">
            <QuestionMarkCircleIcon className="w-4 text-black" />
            <span className="absolute hidden text-white -translate-y-[130%] z-10 bg-black/75 px-1 py-0.5 text-xs rounded-sm w-[200px] group-hover:block">
              Each animal holds its own experimental data. Create an animal to
              get started.
            </span>
          </div>
        </div>
        {!selectedAnimal ? null : (
          <button
            className="flex flex-row items-center justify-center p-1 bg-red-500 rounded-sm"
            onClick={() => {
              // TODO: Delete selected animal
            }}
          >
            <TrashIcon className="w-6 h-6 text-white" />
          </button>
        )}
        <button
          className="flex flex-row items-center justify-center p-1 bg-sky-500 rounded-sm"
          onClick={() => {
            setSelectedAnimal(null);
            setShowAddAnimal(true);
          }}
        >
          <PlusIcon className="w-6 h-6 text-white" />
        </button>
      </div>
      {Object.keys(animals).map((animal) => (
        <AnimalCard
          key={animal}
          name={animal}
          meta={animals[animal]}
          selected={selectedAnimal === animal}
          onClick={() => {
            if (selectedAnimal === animal) {
              setSelectedAnimal(null);
              return;
            }
            setSelectedAnimal(animal);
          }}
        />
      ))}
      <AddAnimalDialog
        isOpen={showAddAnimal}
        setIsOpen={setShowAddAnimal}
        project={project.name}
        didAdd={didAdd}
      />
    </div>
  );
}
