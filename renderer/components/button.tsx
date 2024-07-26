import clsx from "clsx";
import { useState, useEffect } from "react";
import { useRouter } from "next/router";
import Link from "next/link";
const buttonTypes = [
  {
    name: "primary",
    color: "bg-sky-500",
    textColor: "text-white",
    hoverColor: "hover:bg-sky-600",
    activeColor: "active:bg-sky-700",
  },
  {
    name: "secondary",
    color: "bg-gray-500",
    textColor: "text-white",
    hoverColor: "hover:bg-gray-600",
    activeColor: "active:bg-gray-700",
  },
  {
    name: "warning",
    color: "bg-yellow-500",
    textColor: "text-white",
    hoverColor: "hover:bg-yellow-600",
    activeColor: "active:bg-yellow-700",
  },
  {
    name: "danger",
    color: "bg-red-500",
    textColor: "text-white",
    hoverColor: "hover:bg-red-600",
    activeColor: "active:bg-red-700",
  },
  {
    name: "selected",
    color: "bg-indigo-500",
    textColor: "text-white",
    hoverColor: "hover:bg-indigo-600",
    activeColor: "active:bg-indigo-700",
  },
  {
    name: "success",
    color: "bg-green-500",
    textColor: "text-white",
    hoverColor: "hover:bg-green-600",
    activeColor: "active:bg-green-700",
  },
];

export function Button({
  type,
  ...props
}: {
  type: string;
  href?: string;
  disabled?: boolean;
  onClick?: () => void;
  children?: React.ReactNode;
  className?: string;
}) {
  const [mouseDown, setMouseDown] = useState(false);
  const router = useRouter();
  const handleMouseUp = () => {
    setMouseDown(false);
    // Handle interaction with the button
    if (props.onClick) props.onClick();
    if (props.href) router.push(props.href);
  };
  return (
    <Link
      className="w-full"
      href={props.href || "#"}
      onMouseDown={(event) => {
        event.preventDefault();
        setMouseDown(true);
      }}
      onMouseUp={handleMouseUp}
    >
      <button
        {...props}
        onClick={(event) => {
          event.preventDefault();
        }}
        disabled={props.disabled}
        className={clsx(
          "px-4 py-2 font-medium transition-all duration-200 flex flex-row items-center justify-center w-full text-center shadow-sm rounded-sm",
          !mouseDown && "hover:scale-105 hover:shadow-md",
          mouseDown && "scale-105 shadow-md",
          buttonTypes
            .filter((buttonType) => buttonType.name === type)
            .map((buttonType) => buttonType.color),
          buttonTypes
            .filter((buttonType) => buttonType.name === type)
            .map((buttonType) => buttonType.textColor),
          buttonTypes
            .filter((buttonType) => buttonType.name === type)
            .map((buttonType) => buttonType.hoverColor),
          buttonTypes
            .filter((buttonType) => buttonType.name === type)
            .map((buttonType) => buttonType.activeColor),
          props.className
        )}
      >
        {props.children}
      </button>
    </Link>
  );
}
