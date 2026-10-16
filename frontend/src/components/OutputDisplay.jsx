import React from 'react';

export default function OutputDisplay({ text }) {
  return (
    <div className="w-full h-40 p-4 border rounded-md bg-gray-50 text-gray-800 shadow-inner overflow-auto whitespace-pre-wrap">
      {text || <span className="text-gray-400 italic">Translation will appear here...</span>}
    </div>
  );
}
