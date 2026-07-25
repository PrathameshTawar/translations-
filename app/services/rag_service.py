import os
import json

try:
    import faiss
    import numpy as np
    import fasttext
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from langchain_groq import ChatGroq
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
except ImportError:
    print("Dependencies missing. Ensure langchain, sentence-transformers, faiss, and fasttext-wheel are installed.")

def download_fasttext_model():
    import urllib.request
    model_path = "lid.176.bin"
    if not os.path.exists(model_path):
        print("Downloading fasttext language ID model...")
        url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
        urllib.request.urlretrieve(url, model_path)
        print("Downloaded lid.176.bin")
    return model_path

class IndicRAGPipeline:
    def __init__(self):
        lid_model_path = download_fasttext_model()
        try:
            self.lid_model = fasttext.load_model(lid_model_path)
        except Exception as e:
            print(f"Error loading fasttext: {e}")
            self.lid_model = None
            
        model_name = "indicbert-finetuned-hindi"
        if not os.path.exists(model_name):
            print("Finetuned model not found, falling back to ai4bharat/indic-bert")
            model_name = "ai4bharat/indic-bert"
            
        print("Loading embedding model...")
        self.encoder = SentenceTransformer(model_name)
        
        print("Loading cross-encoder reranker...")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("Warning: GROQ_API_KEY environment variable not set. Generation will fail.")
            self.llm = None
        else:
            self.llm = ChatGroq(model="llama3-8b-8192", api_key=api_key)
            
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="Only answer using the provided context; say so explicitly if the answer isn't present.\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
        )
        if self.llm:
            self.qa_chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
            
        self.indices = {}
        self.mappings = {}

    def detect_language(self, query):
        if not self.lid_model:
            return 'hi' # default
        predictions = self.lid_model.predict(query, k=1)
        lang_code = predictions[0][0].replace('__label__', '')
        return lang_code

    def load_index(self, lang):
        if lang not in self.indices:
            index_path = f"{lang}_index.faiss"
            mapping_path = f"{lang}_mapping.json"
            if os.path.exists(index_path) and os.path.exists(mapping_path):
                self.indices[lang] = faiss.read_index(index_path)
                with open(mapping_path, 'r', encoding='utf-8') as f:
                    self.mappings[lang] = json.load(f)
            else:
                print(f"Index or mapping not found for language {lang}")
                return False
        return True

    def retrieve_and_answer(self, query, lang=None, top_k=5, rerank_k=3):
        if not lang:
            lang = self.detect_language(query)
            
        if lang not in ['hi', 'mr']:
            lang = 'hi' 
            
        if not self.load_index(lang):
            return {"answer": "Index not found.", "sources": []}
            
        query_emb = self.encoder.encode([query], convert_to_numpy=True)
        
        distances, indices = self.indices[lang].search(query_emb, top_k)
        
        retrieved_passages = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.mappings[lang]):
                retrieved_passages.append(self.mappings[lang][idx])
                
        if not retrieved_passages:
            return {"answer": "No relevant passages found.", "sources": []}
            
        cross_inp = [[query, passage] for passage in retrieved_passages]
        cross_scores = self.reranker.predict(cross_inp)
        
        ranked_passages = [x for _, x in sorted(zip(cross_scores, retrieved_passages), reverse=True)]
        top_passages = ranked_passages[:rerank_k]
        
        context = "\n".join(top_passages)
        if self.llm:
            try:
                answer = self.qa_chain.run(context=context, question=query)
            except Exception as e:
                answer = f"Error generating answer: {e}"
        else:
            answer = "LLM not initialized (missing API key)."
            
        return {
            "answer": answer,
            "sources": top_passages,
            "detected_lang": lang
        }

if __name__ == "__main__":
    print("RAG Pipeline loaded. You can import IndicRAGPipeline to use it.")
