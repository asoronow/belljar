import clsx from "clsx";
import { useState } from "react";
import { Dialog, DialogTitle, DialogPanel } from "@headlessui/react";
import { Button } from "./button";
import { Input } from "./input";
export function SharpenTool({ project, animal, ...props }) {
  const [open, setOpen] = useState(false);
  const [outputDir, setOutputDir] = useState("");
  const [inputDir, setInputDir] = useState("");
  const [radius, setRadius] = useState(3);
  const [amount, setAmount] = useState(2);
  const [spacing, setSpacing] = useState(0);
  const [running, setRunning] = useState(false);
  return (
    <>
      <div
        onClick={() => {
          if (animal) setOpen(true);
        }}
        className={clsx(
          animal ? "bg-sky-500 hover:bg-sky-600" : "bg-gray-500",
          "flex flex-row items-center justify-start w-full gap-x-2 mb-1 hover:shadow-lg cursor-pointer hover:scale-[1.02] transition-all duration-200 text-white font-medium p-2 w-full rounded-sm mt-1"
        )}
      >
        <h1 className="text-md font-bold text-sm">Sharpen</h1>
      </div>
      <Dialog open={open} onClose={setOpen} className="relative z-50">
        <div className="fixed inset-0 bg-black/30 backdrop-blur-sm" />
        <div className="fixed inset-0 flex w-screen items-center justify-center p-4 sm:p-6">
          <DialogPanel className="max-w-lg space-y-4 border bg-white p-12 rounded-sm transition-all w-full duration-300">
            <DialogTitle className="font-bold text-lg">Sharpen</DialogTitle>
            <p className="text-black text-sm">
              Sharpen lets you ehance the contrast of your cell images for
              better detections.
            </p>
            <div>
              <p className="text-gray-600 text-sm mb-px">Inputs</p>
              <Input
                type="text"
                placeholder="Input Directory"
                value={inputDir}
                invalid={inputDir.length === 0}
                onClick={() => {
                  window.ipc
                    .invoke("get-directory")
                    .then((result) => {
                      if (result.success) {
                        setInputDir(result.directory);
                      }
                    })
                    .catch((err) => {
                      console.log(err);
                    });
                }}
                onChange={(e) => {
                  e.preventDefault();
                }}
              />
            </div>
            <div>
              <p className="text-gray-600 text-sm mb-px">Outputs</p>
              <Input
                type="text"
                placeholder="Output Directory"
                value={outputDir}
                invalid={outputDir.length === 0}
                onClick={() => {
                  window.ipc
                    .invoke("get-directory")
                    .then((result) => {
                      if (result.success) {
                        setOutputDir(result.directory);
                      }
                    })
                    .catch((err) => {
                      console.log(err);
                    });
                }}
                onChange={(e) => {
                  e.preventDefault();
                }}
              />
            </div>
            <div className="relative space-y-2">
              <p className="text-gray-600 text-sm mb-px">Options</p>
              <div className="flex flex-row flex-wrap items-center justify-start w-fit gap-x-2">
                <div className="group flex flex-row items-center justify-start w-full gap-x-2 p-2 bg-gray-200 w-fit rounded-sm">
                  <div className="absolute w-[150px] -translate-y-10 inset-x-0 z-10 text-white text-md hidden group-hover:block text-[10px] bg-black/75 px-1 py-0.5 rounded-sm">
                    The amount of sharpening to apply. A weight of 0 will result
                    in no sharpening.
                  </div>
                  Amount
                  <input
                    type="number"
                    className="border rounded-sm border-black w-12 p-1"
                    value={amount}
                    onChange={(e) => {
                      setAmount(parseInt(e.target.value));
                    }}
                  />
                </div>
              </div>
              <div className="group flex flex-row items-center justify-start w-fit gap-x-2 p-2 bg-gray-200 w-fit rounded-sm">
                <div className="absolute w-[150px] -translate-y-12 inset-x-0 z-10 text-white text-md hidden group-hover:block text-[10px] bg-black/75 px-1 py-0.5 rounded-sm">
                  The radius of the sharpening kernel.
                </div>
                Radius
                <input
                  type="number"
                  className="border rounded-sm border-black w-12 p-1"
                  value={radius}
                  onChange={(e) => {
                    setRadius(parseInt(e.target.value));
                  }}
                />
              </div>
            </div>
            {running && (
              <p className="text-red-500 animate-pulse">Running...</p>
            )}
            <Button
              className="mt-4"
              type={running ? "danger" : "primary"}
              onClick={() => {
                if (!outputDir || outputDir === "" || !animal) {
                  return;
                }

                if (running) {
                  // kill the process
                  setRunning(false);
                  window.ipc.send("killSharpen", null);
                  return;
                }
                window.ipc.once("sharpenResult", () => {
                  setRunning(false);
                  // remove the listener
                });

                setRunning(true);
                window.ipc
                  .invoke(
                    "get-animal-data-directory",
                    project.name,
                    animal,
                    "Background"
                  )
                  .then((result) => {
                    if (result.success) {
                      setInputDir(result.directory);
                      window.ipc.invoke("runSharpen", [
                        inputDir,
                        outputDir,
                        radius,
                        amount,
                      ]);
                    }
                  });
              }}
            >
              {running ? "Cancel" : "Run"}
            </Button>
          </DialogPanel>
        </div>
      </Dialog>
    </>
  );
}
