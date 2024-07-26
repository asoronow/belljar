import clsx from "clsx";
function Pill({ className, children, tooltip = "" }) {
  return (
    <div
      className={clsx(
        className,
        "flex items-center px-2 py-0.5 text-[10px] rounded-sm font-medium group cursor-pointer justify-center"
      )}
    >
      <div
        className={clsx(
          "absolute hidden absolute -translate-y-10 inset-x-0 z-10 bg-black/75 px-1 py-0.5 text-[10px] w-[200px] group-hover:block z-10"
        )}
      >
        {tooltip}
      </div>
      {children}
    </div>
  );
}

export function AnimalCard({
  name,
  selected,
  meta,
  ...props
}: {
  name: string;
  selected: boolean;
  meta: { [key: string]: any };
  [key: string]: any;
}) {
  return (
    <div
      key={name}
      className={clsx(
        selected ? "bg-indigo-500" : "bg-black",
        "relative flex p-2 flex-row w-full items-center justify-between rounded-sm text-white hover:scale-[1.02] hover:shadow-lg transition-all duration-200 cursor-pointer"
      )}
      {...props}
    >
      <p className="text-sm font-bold truncate">{name}</p>
      <div className="flex grid grid-row-flow md:grid-cols-2 gap-1 w-fit">
        <Pill
          tooltip={
            "Data for perfoming cell detection. Green indicates data exists."
          }
          className={meta.hasCellDetectionData ? "bg-green-500" : "bg-gray-500"}
        >
          Signal
        </Pill>
        <Pill
          tooltip={"Data for perfoming alignment. Green indicates data exists."}
          className={meta.hasAlignmentData ? "bg-green-500" : "bg-gray-500"}
        >
          Background
        </Pill>
        <Pill
          tooltip={"Has the animal been aligned. Green indicates yes."}
          className={meta.alignmentRun ? "bg-green-500" : "bg-gray-500"}
        >
          Aligned
        </Pill>
        <Pill
          tooltip={"Has the signal data been detected. Green indicates yes."}
          className={meta.cellDetectionRun ? "bg-green-500" : "bg-gray-500"}
        >
          Detected
        </Pill>
      </div>
    </div>
  );
}
