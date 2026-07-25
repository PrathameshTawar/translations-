import React from 'react';

export default function LanguageSelector({ value, onChange }) {
  const languages = [
    { code: 'english', name: 'English' },
    { code: 'hindi', name: 'हिंदी' },
    { code: 'marathi', name: 'मराठी' },
    { code: 'bengali', name: 'বাংলা' },
    { code: 'gujarati', name: 'ગુજરાતી' },
    { code: 'tamil', name: 'தமிழ்' },
    { code: 'telugu', name: 'తెలుగు' },
    { code: 'kannada', name: 'ಕನ್ನಡ' }
  ];

  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="border rounded p-2 text-gray-700 bg-white shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
    >
      {languages.map((lang) => (
        <option key={lang.code} value={lang.code}>
          {lang.name}
        </option>
      ))}
    </select>
  );
}
