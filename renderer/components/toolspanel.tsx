import { AlignTool } from "./align_tool";
export function ToolsPanel({ project, animal, ...props }) {
  return (
    <div className="flex flex-col items-start justify-start w-full bg-gray-200 p-4 h-[300px] overflow-y-scroll">
      <h1 className="text-xl font-bold mb-4">Tools</h1>
      <AlignTool project={project} animal={animal} {...props} />
    </div>
  );
}
