import { useState } from "react";
import { Dialog, DialogTitle, DialogPanel } from "@headlessui/react";
import { Button } from "./button";
import { Input } from "./input";
export function CreateProjectDialog({
  showCreateProject,
  setShowCreateProject,
  createProject,
  projectInvalid,
  setNewProjectDescription,
  setNewProjectName,
}) {
  return (
    <Dialog open={showCreateProject} onClose={setShowCreateProject}>
      <div className="fixed inset-0 bg-black/30 backdrop-blur-sm" />
      <div className="fixed inset-0 flex w-screen items-center justify-center p-4 rounded-lg sm:p-6">
        <DialogPanel className="max-w-lg space-y-4 border bg-white p-12 rounded-sm transition-all w-full duration-300 flex flex-col">
          <DialogTitle className="font-bold">Create Project</DialogTitle>
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
            />
            <Input
              type="text"
              onChange={(e) => {
                setNewProjectDescription(e.target.value);
              }}
              invalid={projectInvalid}
              placeholder="Project description"
            />
          </div>
          <Button type="primary" onClick={createProject}>
            Create
          </Button>
          <Button type="secondary" onClick={() => setShowCreateProject(false)}>
            Cancel
          </Button>
        </DialogPanel>
      </div>
    </Dialog>
  );
}
