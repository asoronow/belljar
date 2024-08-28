import { AlignTool } from "./align_tool";
import { SharpenTool } from "./sharpen_tool";
import { MaxTool } from "./max_tool";
import { DetectionTool } from "./detection_tool";
import { ViewerTool } from "./viewer_tool";
import { CountTool } from "./count_tool";
export function ToolsPanel({ project, animal, ...props }) {
  return (
    <div className="flex flex-col items-start justify-start w-full bg-gray-200 p-4 h-[300px] overflow-y-scroll">
      <h1 className="text-xl font-bold mb-4">Tools</h1>
      <AlignTool project={project} animal={animal} {...props} />
      <SharpenTool project={project} animal={animal} {...props} />
      <MaxTool project={project} animal={animal} {...props} />
      <DetectionTool project={project} animal={animal} {...props} />
      <ViewerTool project={project} animal={animal} {...props} />
      <CountTool project={project} animal={animal} {...props} />
    </div>
  );
}
