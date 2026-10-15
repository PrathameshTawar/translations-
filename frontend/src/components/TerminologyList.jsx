import React from 'react';

export default function TerminologyList({ terms }) {
  if (!terms || terms.length === 0) return null;

  return (
    <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-md">
      <h4 className="text-sm font-semibold text-blue-900 mb-2">Context / References Used</h4>
      <ul className="space-y-1 text-sm text-blue-800">
        {terms.map((term, idx) => (
          <li key={idx} className="flex flex-col border-b border-blue-100 pb-1">
            <span className="font-medium">{term.term}</span>
            <span className="text-xs opacity-80">{term.context}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
