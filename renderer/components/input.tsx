import clsx from "clsx";

declare module "react" {
  interface InputHTMLAttributes<T> extends HTMLAttributes<T> {
    webkitdirectory?: string;
  }
}
export function Input({
  onChange,
  type,
  placeholder,
  invalid,
  directoryOnly,
  ...props
}: {
  onChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
  type: string;
  placeholder: string;
  invalid?: boolean;
  directoryOnly?: boolean;
  [key: string]: any;
}) {
  return (
    <input
      type={type}
      className={clsx(
        "w-full p-2 text-black focus:outline-none focus:ring-2 focus:ring-blue-600 rounded-sm border border-gray-300 focus:border-transparent",
        invalid && "border-red-500"
      )}
      onChange={onChange}
      placeholder={placeholder}
      {...(directoryOnly ? { webkitdirectory: "" } : {})}
      {...props}
    />
  );
}
