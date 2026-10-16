import React from 'react';

export default function QualityBadge({ score, confidence }) {
  const getBadgeColor = () => {
    if (confidence === 'high') return 'bg-green-100 text-green-800 border-green-200';
    if (confidence === 'medium') return 'bg-yellow-100 text-yellow-800 border-yellow-200';
    return 'bg-red-100 text-red-800 border-red-200';
  };

  return (
    <div className={`inline-flex items-center px-3 py-1 rounded-full border text-sm font-medium ${getBadgeColor()}`}>
      <span className="mr-2">Quality Score:</span>
      <strong>{Math.round(score * 100)}%</strong>
      <span className="ml-2 uppercase text-xs tracking-wider opacity-75">({confidence})</span>
    </div>
  );
}
