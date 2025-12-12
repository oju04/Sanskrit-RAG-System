import time
import json
from typing import List, Dict
from main import SanskritRAGSystem

class RAGEvaluator:
    
    def __init__(self, rag_system: SanskritRAGSystem):
        self.rag = rag_system
        self.results = []
    
    def create_test_set(self) -> List[Dict]:
        test_queries = [
            {
                'query': 'Tell me about the foolish servant',
                'expected_keywords': ['शंखनाद', 'servant', 'foolish', 'गोवर्धनदास'],
                'expected_title': 'मूर्खभृत्यस्य',
                'type': 'story'
            },
            {
                'query': 'What is the moral about servants?',
                'expected_keywords': ['moral', 'भृत्य', 'servant', 'नीति'],
                'expected_title': 'मूर्खभृत्यस्य',
                'type': 'moral'
            },
            {
                'query': 'How did Kalidasa help the poet?',
                'expected_keywords': ['कालीदास', 'Kalidasa', 'poet', 'clever'],
                'expected_title': 'कालीदासस्य',
                'type': 'how'
            },
            {
                'query': 'कालीदासस्य चातुर्यम् किम्?',
                'expected_keywords': ['कालीदास', 'चातुर्य', 'भोजराज'],
                'expected_title': 'कालीदासस्य',
                'type': 'sanskrit'
            },
            {
                'query': 'What happened with the old woman and bell?',
                'expected_keywords': ['old woman', 'वृद्धा', 'bell', 'घण्टा', 'monkey'],
                'expected_title': 'वृद्धायाः',
                'type': 'story'
            },
            {
                'query': 'What lesson is taught about divine help?',
                'expected_keywords': ['देव', 'god', 'effort', 'प्रयत्न', 'devotee'],
                'expected_title': 'देवभक्तः',
                'type': 'moral'
            },
            {
                'query': 'Who was Shankhanada?',
                'expected_keywords': ['शंखनाद', 'servant', 'foolish'],
                'expected_title': 'मूर्खभृत्यस्य',
                'type': 'who'
            },
            {
                'query': 'How did the scholar lose to Kalidasa?',
                'expected_keywords': ['scholar', 'grammar', 'शीतं', 'बाधते'],
                'expected_title': 'शीतं बाधते',
                'type': 'how'
            },
            {
                'query': 'What stories show cleverness?',
                'expected_keywords': ['clever', 'चातुर्य', 'कालीदास', 'वृद्धा'],
                'expected_title': 'multiple',
                'type': 'theme'
            },
            {
                'query': 'Tell me about King Bhoja',
                'expected_keywords': ['भोजराज', 'Bhoj', 'king', 'court'],
                'expected_title': 'कालीदासस्य',
                'type': 'who'
            },
            {
                'query': 'मूर्खभृत्यस्य नीतिः किम्?',
                'expected_keywords': ['नीति', 'moral', 'भृत्य', 'servant'],
                'expected_title': 'मूर्खभृत्यस्य',
                'type': 'sanskrit_moral'
            },
            {
                'query': 'What does the devotee learn?',
                'expected_keywords': ['devotee', 'भक्त', 'effort', 'उद्यम'],
                'expected_title': 'देवभक्तः',
                'type': 'lesson'
            },
            {
                'query': 'How was the demon problem solved?',
                'expected_keywords': ['demon', 'राक्षस', 'घण्टाकर्ण', 'solution'],
                'expected_title': 'वृद्धायाः',
                'type': 'how'
            },
            {
                'query': 'What mistake did the servant make with milk?',
                'expected_keywords': ['milk', 'दुग्ध', 'rope', 'दोरक', 'spill'],
                'expected_title': 'मूर्खभृत्यस्य',
                'type': 'story'
            },
            {
                'query': 'उद्यमः साहसम् धैर्यम् - explain this verse',
                'expected_keywords': ['उद्यम', 'effort', 'courage', 'god helps'],
                'expected_title': 'देवभक्तः',
                'type': 'verse'
            }
        ]
        
        return test_queries
    
    def evaluate_retrieval(self, query: str, retrieved_chunks: List[Dict], 
                          expected_keywords: List[str], expected_title: str) -> Dict:
        """Evaluate retrieval quality"""
        
        # Check if expected title in retrieved chunks
        titles = [chunk['title'] for chunk in retrieved_chunks]
        title_match = any(expected_title in title for title in titles)
        
        # Check keyword presence in retrieved content
        all_content = ' '.join([chunk['content'] for chunk in retrieved_chunks]).lower()
        keyword_matches = sum(1 for kw in expected_keywords if kw.lower() in all_content)
        keyword_score = keyword_matches / len(expected_keywords) if expected_keywords else 0
        
        # Relevance scores
        avg_score = sum(chunk.get('relevance_score', 0) for chunk in retrieved_chunks) / len(retrieved_chunks) if retrieved_chunks else 0
        
        return {
            'title_match': title_match,
            'keyword_score': keyword_score,
            'avg_relevance': avg_score,
            'num_retrieved': len(retrieved_chunks)
        }
    
    def evaluate_response(self, answer: str, expected_keywords: List[str]) -> Dict:
        """Evaluate response quality"""
        
        answer_lower = answer.lower()
        
        # Keyword coverage
        keyword_in_answer = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
        coverage = keyword_in_answer / len(expected_keywords) if expected_keywords else 0
        
        # Response length (proxy for completeness)
        length_score = min(len(answer.split()) / 50, 1.0)  # Normalize to 50 words
        
        # Check for error messages
        has_error = 'sorry' in answer_lower or 'cannot find' in answer_lower or 'क्षम्यताम्' in answer_lower
        
        return {
            'keyword_coverage': coverage,
            'length_score': length_score,
            'has_error': has_error,
            'length': len(answer.split())
        }
    
    def run_evaluation(self):
        """Run complete evaluation"""
        print("=" * 70)
        print("Sanskrit RAG System Evaluation")
        print("=" * 70)
        
        test_set = self.create_test_set()
        total_queries = len(test_set)
        
        # Metrics
        retrieval_metrics = {
            'title_matches': 0,
            'avg_keyword_score': 0,
            'avg_relevance': 0
        }
        
        response_metrics = {
            'avg_coverage': 0,
            'errors': 0,
            'avg_length': 0
        }
        
        timing_metrics = {
            'total_time': 0,
            'times': []
        }
        
        print(f"\nRunning {total_queries} test queries...\n")
        
        for i, test in enumerate(test_set, 1):
            print(f"[{i}/{total_queries}] Testing: {test['query'][:50]}...")
            
            # Time the query
            start_time = time.time()
            result = self.rag.query(test['query'])
            end_time = time.time()
            
            query_time = end_time - start_time
            timing_metrics['times'].append(query_time)
            timing_metrics['total_time'] += query_time
            
            # Evaluate retrieval
            retrieval_eval = self.evaluate_retrieval(
                test['query'],
                result.get('retrieved_chunks', []),
                test['expected_keywords'],
                test['expected_title']
            )
            
            # Evaluate response
            response_eval = self.evaluate_response(
                result.get('answer', ''),
                test['expected_keywords']
            )
            
            # Aggregate metrics
            if retrieval_eval['title_match']:
                retrieval_metrics['title_matches'] += 1
            retrieval_metrics['avg_keyword_score'] += retrieval_eval['keyword_score']
            retrieval_metrics['avg_relevance'] += retrieval_eval['avg_relevance']
            
            response_metrics['avg_coverage'] += response_eval['keyword_coverage']
            if response_eval['has_error']:
                response_metrics['errors'] += 1
            response_metrics['avg_length'] += response_eval['length']
            
            # Store individual result
            self.results.append({
                'query': test['query'],
                'type': test['type'],
                'retrieval': retrieval_eval,
                'response': response_eval,
                'time': query_time,
                'confidence': result.get('confidence', 'unknown')
            })
            
            print(f"  ✓ Time: {query_time*1000:.0f}ms | "
                  f"Title: {'✓' if retrieval_eval['title_match'] else '✗'} | "
                  f"Keywords: {retrieval_eval['keyword_score']:.2f}")
        
        # Calculate averages
        retrieval_metrics['avg_keyword_score'] /= total_queries
        retrieval_metrics['avg_relevance'] /= total_queries
        response_metrics['avg_coverage'] /= total_queries
        response_metrics['avg_length'] /= total_queries
        
        # Print results
        self._print_results(retrieval_metrics, response_metrics, timing_metrics, total_queries)
        
        return self.results
    
    def _print_results(self, retrieval_metrics, response_metrics, timing_metrics, total):
        """Print evaluation results"""
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        
        print("\n📊 RETRIEVAL METRICS:")
        print(f"  Title Match Rate:     {retrieval_metrics['title_matches']}/{total} "
              f"({retrieval_metrics['title_matches']/total*100:.1f}%)")
        print(f"  Avg Keyword Score:    {retrieval_metrics['avg_keyword_score']:.3f}")
        print(f"  Avg Relevance Score:  {retrieval_metrics['avg_relevance']:.3f}")
        
        print("\n📝 RESPONSE METRICS:")
        print(f"  Avg Keyword Coverage: {response_metrics['avg_coverage']:.3f}")
        print(f"  Error Rate:           {response_metrics['errors']}/{total} "
              f"({response_metrics['errors']/total*100:.1f}%)")
        print(f"  Avg Response Length:  {response_metrics['avg_length']:.0f} words")
        
        print("\n⚡ PERFORMANCE METRICS:")
        print(f"  Total Time:           {timing_metrics['total_time']:.2f}s")
        print(f"  Avg Query Time:       {timing_metrics['total_time']/total*1000:.0f}ms")
        print(f"  Min Query Time:       {min(timing_metrics['times'])*1000:.0f}ms")
        print(f"  Max Query Time:       {max(timing_metrics['times'])*1000:.0f}ms")
        
        # Overall score
        overall_score = (
            retrieval_metrics['title_matches']/total * 0.4 +
            retrieval_metrics['avg_keyword_score'] * 0.3 +
            response_metrics['avg_coverage'] * 0.3
        ) * 100
        
        print(f"\n🎯 OVERALL SCORE: {overall_score:.1f}%")
        print("=" * 70)
    
    def save_results(self, filepath: str):
        """Save detailed results to JSON"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {filepath}")
    
    def analyze_by_type(self):
        """Analyze performance by query type"""
        print("\n" + "=" * 70)
        print("ANALYSIS BY QUERY TYPE")
        print("=" * 70)
        
        types = {}
        for result in self.results:
            qtype = result['type']
            if qtype not in types:
                types[qtype] = []
            types[qtype].append(result)
        
        for qtype, results in sorted(types.items()):
            print(f"\n{qtype.upper()} ({len(results)} queries):")
            
            avg_time = sum(r['time'] for r in results) / len(results) * 1000
            title_matches = sum(1 for r in results if r['retrieval']['title_match'])
            avg_kw_score = sum(r['retrieval']['keyword_score'] for r in results) / len(results)
            
            print(f"  Avg Time:         {avg_time:.0f}ms")
            print(f"  Title Match:      {title_matches}/{len(results)}")
            print(f"  Keyword Score:    {avg_kw_score:.3f}")

def main():
    """Main evaluation function"""
    print("Initializing RAG system...")
    rag = SanskritRAGSystem()
    
    # Load pre-indexed model or index new
    try:
        rag.load_model("models")
    except:
        print("No existing index found. Indexing document...")
        rag.ingest_document("data/Rag-docs.txt")
        rag.save_model("models")
    
    # Run evaluation
    evaluator = RAGEvaluator(rag)
    results = evaluator.run_evaluation()
    
    # Analyze by type
    evaluator.analyze_by_type()
    
    # Save results
    evaluator.save_results("evaluation_results.json")
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()
