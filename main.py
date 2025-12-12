import re
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
import pickle

class SanskritDocumentLoader:
    
    def __init__(self, encoding='utf-8'):
        self.encoding = encoding
    
    def load_document(self, filepath: str) -> str:
        with open(filepath, 'r', encoding=self.encoding) as f:
            return f.read()
    
    def extract_stories(self, text: str) -> List[Dict]:
        # Split by story markers (looking for ** headers **)
        pattern = r'\*\*([^*]+)\*\*'
        parts = re.split(pattern, text)
        
        stories = []
        for i in range(1, len(parts), 2):
            if i < len(parts):
                title = parts[i].strip()
                content = parts[i+1].strip() if i+1 < len(parts) else ""
                if content:
                    stories.append({
                        'title': title,
                        'content': content
                    })
        
        return stories

class SanskritPreprocessor:
    
    def __init__(self):
        self.devanagari_range = range(0x0900, 0x097F)
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def create_chunks(self, story: Dict, chunk_size: int = 200) -> List[Dict]:
        content = story['content']
        sentences = re.split(r'[।॥\.]', content)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
                
            sent_length = len(sent.split())
            
            if current_length + sent_length > chunk_size and current_chunk:
                chunks.append({
                    'title': story['title'],
                    'content': ' '.join(current_chunk),
                    'chunk_id': len(chunks)
                })
                current_chunk = [sent]
                current_length = sent_length
            else:
                current_chunk.append(sent)
                current_length += sent_length
        
        if current_chunk:
            chunks.append({
                'title': story['title'],
                'content': ' '.join(current_chunk),
                'chunk_id': len(chunks)
            })
        
        return chunks
    
    def extract_keywords(self, text: str) -> List[str]:
       
        # Remove punctuation and split
        words = re.findall(r'\w+', text.lower())
        # Filter short words
        keywords = [w for w in words if len(w) > 2]
        return keywords

class KeywordRetriever:
    
    
    def __init__(self):
        self.chunks = []
        self.vocab = {}
        self.idf_scores = {}
        self.chunk_vectors = []
    
    def build_index(self, chunks: List[Dict]):
        """Build inverted index for chunks"""
        self.chunks = chunks
        
        # Build vocabulary
        word_doc_count = Counter()
        all_words = set()
        
        for chunk in chunks:
            words = set(re.findall(r'\w+', chunk['content'].lower()))
            all_words.update(words)
            for word in words:
                word_doc_count[word] += 1
        
        # Calculate IDF
        n_docs = len(chunks)
        self.vocab = {word: idx for idx, word in enumerate(sorted(all_words))}
        
        for word, count in word_doc_count.items():
            self.idf_scores[word] = np.log((n_docs + 1) / (count + 1)) + 1
        
        # Create chunk vectors (TF-IDF)
        for chunk in chunks:
            vector = self._vectorize(chunk['content'])
            self.chunk_vectors.append(vector)
    
    def _vectorize(self, text: str) -> np.ndarray:
        
        words = re.findall(r'\w+', text.lower())
        word_count = Counter(words)
        
        vector = np.zeros(len(self.vocab))
        for word, count in word_count.items():
            if word in self.vocab:
                idx = self.vocab[word]
                tf = count / len(words) if words else 0
                idf = self.idf_scores.get(word, 1)
                vector[idx] = tf * idf
        
       
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        
        query_vector = self._vectorize(query)
        
        scores = []
        for idx, chunk_vector in enumerate(self.chunk_vectors):
        
            score = np.dot(query_vector, chunk_vector)
            scores.append((score, idx))

    
        scores.sort(reverse=True)
        
        # Return top-k with scores
        results = []
        for score, idx in scores[:top_k]:
            if score > 0:  # Only return if some relevance
                chunk = self.chunks[idx].copy()
                chunk['relevance_score'] = float(score)
                results.append(chunk)
        
        return results
    
    def save_index(self, filepath: str):
        """Save index to disk"""
        data = {
            'chunks': self.chunks,
            'vocab': self.vocab,
            'idf_scores': self.idf_scores,
            'chunk_vectors': self.chunk_vectors
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load_index(self, filepath: str):
        """Load index from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.chunks = data['chunks']
        self.vocab = data['vocab']
        self.idf_scores = data['idf_scores']
        self.chunk_vectors = data['chunk_vectors']

class RuleBasedGenerator:
    
    
    def __init__(self):
        self.patterns = {
            'who': r'\b(who|कः|का|के)\b',
            'what': r'\b(what|किम्|क्या)\b',
            'how': r'\b(how|कथम्|कैसे)\b',
            'why': r'\b(why|किमर्थम्|क्यों)\b',
            'moral': r'\b(moral|teaching|lesson|नीति|उपदेश)\b',
            'story': r'\b(story|tale|कथा|कहानी)\b'
        }
    
    def generate(self, query: str, chunks: List[Dict]) -> Dict:
        """Generate response based on retrieved chunks"""
        if not chunks:
            return {
                'answer': "क्षम्यताम्, प्रश्नस्य उत्तरम् न प्राप्तम्। (Sorry, no relevant information found.)",
                'confidence': 'low',
                'sources': []
            }
        
        query_lower = query.lower()
        response_type = self._classify_query(query_lower)
        
        if response_type == 'who':
            return self._answer_who(chunks)
        elif response_type == 'what':
            return self._answer_what(chunks)
        elif response_type == 'how':
            return self._answer_how(chunks)
        elif response_type == 'moral':
            return self._answer_moral(chunks)
        else:
            return self._answer_general(chunks)
    
    def _classify_query(self, query: str) -> str:
        
        for qtype, pattern in self.patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                return qtype
        return 'general'
    
    def _answer_who(self, chunks: List[Dict]) -> Dict:
        """Answer who questions"""
        chunk = chunks[0]
        answer = f"Based on '{chunk['title']}':\n\n{chunk['content'][:300]}..."
        
        return {
            'answer': answer,
            'confidence': 'high',
            'sources': [c['title'] for c in chunks[:2]]
        }
    
    def _answer_what(self, chunks: List[Dict]) -> Dict:
        """Answer what questions"""
        summaries = []
        for i, chunk in enumerate(chunks[:2], 1):
            summaries.append(f"{i}. {chunk['title']}: {chunk['content'][:200]}...")
        
        answer = "Relevant information:\n\n" + "\n\n".join(summaries)
        
        return {
            'answer': answer,
            'confidence': 'high',
            'sources': [c['title'] for c in chunks[:2]]
        }
    
    def _answer_how(self, chunks: List[Dict]) -> Dict:
        """Answer how questions"""
        chunk = chunks[0]
        answer = f"From '{chunk['title']}':\n\n{chunk['content']}"
        
        return {
            'answer': answer,
            'confidence': 'medium',
            'sources': [chunk['title']]
        }
    
    def _answer_moral(self, chunks: List[Dict]) -> Dict:
        
        # Look for verses (indicated by ॥)
        for chunk in chunks:
            if '॥' in chunk['content']:
                return {
                    'answer': f"Moral teaching:\n\n{chunk['content']}",
                    'confidence': 'high',
                    'sources': [chunk['title']]
                }
        
        return self._answer_general(chunks)
    
    def _answer_general(self, chunks: List[Dict]) -> Dict:
        """General answer"""
        chunk = chunks[0]
        answer = f"Most relevant content from '{chunk['title']}':\n\n{chunk['content'][:400]}..."
        
        return {
            'answer': answer,
            'confidence': 'medium',
            'sources': [c['title'] for c in chunks[:2]]
        }

class SanskritRAGSystem:
    
    
    def __init__(self):
        self.loader = SanskritDocumentLoader()
        self.preprocessor = SanskritPreprocessor()
        self.retriever = KeywordRetriever()
        self.generator = RuleBasedGenerator()
        self.indexed = False
    
    def ingest_document(self, filepath: str):
        
        print(f"Loading document: {filepath}")
        text = self.loader.load_document(filepath)
        
        print("Extracting stories...")
        stories = self.loader.extract_stories(text)
        print(f"Found {len(stories)} stories")
        
        print("Creating chunks...")
        all_chunks = []
        for story in stories:
            chunks = self.preprocessor.create_chunks(story)
            all_chunks.extend(chunks)
        print(f"Created {len(all_chunks)} chunks")
        
        print("Building retrieval index...")
        self.retriever.build_index(all_chunks)
        self.indexed = True
        print("Indexing complete!")
        
        return len(all_chunks)
    
    def query(self, question: str, top_k: int = 3) -> Dict:
        """Process a query and return answer"""
        if not self.indexed:
            return {
                'error': 'System not indexed. Please ingest documents first.'
            }
        
        print(f"\nQuery: {question}")
        
        # Retrieve relevant chunks
        print("Retrieving relevant chunks...")
        chunks = self.retriever.retrieve(question, top_k=top_k)
        print(f"Retrieved {len(chunks)} chunks")
        
        # Generate response
        print("Generating response...")
        response = self.generator.generate(question, chunks)
        response['retrieved_chunks'] = chunks
        
        return response
    
    def save_model(self, dirpath: str):
        """Save the indexed model"""
        Path(dirpath).mkdir(parents=True, exist_ok=True)
        self.retriever.save_index(f"{dirpath}/index.pkl")
        print(f"Model saved to {dirpath}")
    
    def load_model(self, dirpath: str):
        """Load a pre-indexed model"""
        self.retriever.load_index(f"{dirpath}/index.pkl")
        self.indexed = True
        print(f"Model loaded from {dirpath}")

def main():
    """Main execution function"""
    print("=" * 60)
    print("Sanskrit Document RAG System")
    print("CPU-Optimized Implementation")
    print("=" * 60)
    
    # Initialize system
    rag = SanskritRAGSystem()
    
    # Ingest document
    doc_path = "data/Rag-docs.txt"  # Adjust path as needed
    rag.ingest_document(doc_path)
    
    # Save index for future use
    rag.save_model("models")
    
    # Example queries
    queries = [
        "Tell me about the foolish servant",
        "What moral lesson is taught about divine help?",
        "How did Kalidasa show his cleverness?",
        "कालीदासस्य कथा वदतु",
        "What happened with the old woman and the bell?"
    ]
    
    print("\n" + "=" * 60)
    print("Running Example Queries")
    print("=" * 60)
    
    for query in queries:
        print("\n" + "-" * 60)
        result = rag.query(query)
        
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nConfidence: {result['confidence']}")
        print(f"Sources: {', '.join(result['sources'])}")
        print(f"Retrieved chunks: {len(result['retrieved_chunks'])}")
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("Interactive Mode (type 'quit' to exit)")
    print("=" * 60)
    
    while True:
        try:
            user_query = input("\nEnter your query: ").strip()
            if user_query.lower() in ['quit', 'exit', 'q']:
                break
            
            if user_query:
                result = rag.query(user_query)
                print(f"\n{result['answer']}")
                print(f"\n[Confidence: {result['confidence']}]")
        
        except KeyboardInterrupt:
            break
    
    print("\nThank you for using Sanskrit RAG System!")

if __name__ == "__main__":
    main()
