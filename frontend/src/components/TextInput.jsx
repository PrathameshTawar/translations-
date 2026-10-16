import React from 'react';

export default function TextInput({ value, onChange, placeholder, maxLength }) {
  return (
    <textarea
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      maxLength={maxLength}
      className="w-full h-40 p-4 border rounded-md resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-800 bg-white shadow-sm"
    />
  );
}
