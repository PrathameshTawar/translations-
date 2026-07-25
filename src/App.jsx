import React from 'react';
import TranslatorCard from './components/TranslatorCard';

function App() {
  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center py-10 px-4">
      <div className="w-full max-w-6xl mb-8 text-center">
        <h1 className="text-4xl font-extrabold text-blue-900 tracking-tight sm:text-5xl">
          Indian Translation
        </h1>
        <p className="mt-4 text-xl text-gray-600">
          RAG-Enhanced Online Indic Languages Translator
        </p>
      </div>

      <TranslatorCard />
      
      <footer className="mt-auto py-8 text-center text-sm text-gray-500">
        &copy; 2026 Indic RAG Research. Powered by NLLB-200 and fine-tuned embeddings.
      </footer>
    </div>
  );
}

export default App;
