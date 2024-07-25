export function Input({ onChange, type, placeholder, ...props }) {
  return (
    <input
      type={type}
      className="w-full border border-gray-300 p-2"
      onChange={onChange}
      placeholder={placeholder}
      {...props}
    />
  );
}
