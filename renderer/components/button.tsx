export function Button({ ...props }) {
  return (
    <button
      {...props}
      className="rounded-lg bg-sky-500 px-5 py-2 text-white hover:bg-sky-600 active:bg-sky-700"
    >
      {props.children}
    </button>
  );
}
