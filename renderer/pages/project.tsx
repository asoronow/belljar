import React, { useState, useEffect } from "react";
import { useRouter } from "next/router";
import { ChevronLeftIcon, TrashIcon } from "@heroicons/react/24/solid";
import Link from "next/link";
export default function ProjectPage() {
  const router = useRouter();
  const { id } = router.query;
  const [project, setProject] = useState(null);
  const [selectedAnimal, setSelectedAnimal] = useState(null);
  const [selectedData, setSelectedData] = useState(null);

  useEffect(() => {
    if (id) {
      window.ipc.invoke("load-project", id).then((result) => {
        if (result.success) {
          setProject(result.project);
        } else {
          alert(result.error);
        }
      });
    }
  }, [id]);

  return (
    <React.Fragment>
      <div className="flex flex-col items-start justify-start w-full min-h-screen p-6">
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
                      "Are you sure you want to delete this project?"
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
              <div className="flex flex-col items-start justify-start w-full bg-gray-100 rounded-y-3xl rounded-l-3xl border-y border-l p-4 h-[300px] overflow-y-scroll">
                <h1 className="text-xl font-bold">Animals</h1>
              </div>
              <div className="flex flex-col items-start justify-start w-full bg-gray-100 rounded-y-3xl rounded-l-3xl border-y border-l p-4 h-[300px] overflow-y-scroll">
                <h1 className="text-xl font-bold">Data</h1>
                {selectedAnimal ? (
                  <></>
                ) : (
                  <p className="text-sm m-auto">
                    Select an animal to manage data.
                  </p>
                )}
              </div>
              <div className="flex flex-col items-start justify-start w-full bg-gray-100 rounded-y-3xl rounded-l-3xl border-y border-l p-4 h-[300px] overflow-y-scroll">
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
