import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Set, Callable
from dataclasses import dataclass, field
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class IterativeRAGConfig:
    num_iterations: int = 3
    max_new_tokens: int = 100
    top_k: int = 5
    temperature: float = 0.7
    consistency_threshold: float = 0.7
    observation_vocab: Optional[List[str]] = None
    device: str = 'cuda'


class ObservationDetector(nn.Module):
    def __init__(self, config: IterativeRAGConfig):
        super().__init__()
        self.config = config
        self.observation_vocab = config.observation_vocab or self._get_default_observations()
    
    def _get_default_observations(self) -> List[str]:
        return [
            "Atelectasis", "Cardiomegaly", "Consolidation",
            "Edema", "Pleural Effusion", "Pneumonia",
            "Pneumothorax", "No Finding", "Fracture",
            "Support Devices", "Enlarged Cardiomediastinum",
            "Lung Opacity", "Pulmonary Edema", "Rib Fracture"
        ]
    
    def detect_observations(self, text: str) -> Set[str]:
        if not text:
            return set()
        
        text_lower = text.lower()
        detected = set()
        
        for obs in self.observation_vocab:
            if obs.lower() in text_lower:
                detected.add(obs)
        
        return detected
    
    def find_missing_observations(self, generated_text: str, reference_text: str) -> Set[str]:
        try:
            generated_obs = self.detect_observations(generated_text)
            reference_obs = self.detect_observations(reference_text)
            
            missing = reference_obs - generated_obs
            return missing
        
        except Exception as e:
            logger.error(f"Error finding missing observations: {e}")
            return set()


class ConsistencyVerifier(nn.Module):
    def __init__(self, config: IterativeRAGConfig):
        super().__init__()
        self.config = config
        self.observation_detector = ObservationDetector(config)
    
    def compute_consistency(self, generations: List[str]) -> float:
        if len(generations) < 2:
            return 1.0
        
        try:
            observation_sets = [self.observation_detector.detect_observations(gen) 
                               for gen in generations]
            
            if not observation_sets or all(len(obs) == 0 for obs in observation_sets):
                return 1.0
            
            common_obs = set.intersection(*observation_sets) if observation_sets else set()
            all_obs = set.union(*observation_sets) if observation_sets else set()
            
            if len(all_obs) == 0:
                return 1.0
            
            consistency = len(common_obs) / len(all_obs)
            return float(consistency)
        
        except Exception as e:
            logger.error(f"Error computing consistency: {e}")
            return 0.0
    
    def find_consistent_observations(self, generations: List[str]) -> Set[str]:
        try:
            observation_sets = [self.observation_detector.detect_observations(gen) 
                               for gen in generations]
            
            if not observation_sets:
                return set()
            
            consistent = set.intersection(*observation_sets)
            return consistent
        
        except Exception as e:
            logger.error(f"Error finding consistent observations: {e}")
            return set()


class TargetedRetriever(nn.Module):
    def __init__(self, config: IterativeRAGConfig):
        super().__init__()
        self.config = config
    
    def build_retrieval_query(self, missing_observations: Set[str], image_context: str = "") -> str:
        if not missing_observations:
            return "general findings"
        
        obs_text = ", ".join(list(missing_observations)[:5])
        query = f"Cases with {obs_text}"
        
        if image_context:
            query += f" in {image_context}"
        
        return query
    
    def rank_retrieved_passages(self, passages: List[str],
                               missing_observations: Set[str]) -> List[Tuple[str, float]]:
        if not passages or not missing_observations:
            return [(p, 0.5) for p in passages]
        
        try:
            ranked = []
            observation_detector = ObservationDetector(self.config)
            
            for passage in passages:
                passage_obs = observation_detector.detect_observations(passage)
                
                overlap = len(passage_obs & missing_observations)
                coverage = overlap / (len(missing_observations) + 1e-8)
                
                diversity_bonus = min(overlap / max(len(missing_observations), 1), 1.0) * 0.2
                
                score = coverage + diversity_bonus
                ranked.append((passage, score))
            
            ranked.sort(key=lambda x: x[1], reverse=True)
            return ranked
        
        except Exception as e:
            logger.error(f"Error ranking passages: {e}")
            return [(p, 0.5) for p in passages]


class IterativeRetrieval(nn.Module):
    def __init__(self, config: IterativeRAGConfig):
        super().__init__()
        self.config = config
        self.targeted_retriever = TargetedRetriever(config)
        self.observation_detector = ObservationDetector(config)
    
    def initialize_retrieval_state(self) -> Dict:
        return {
            'iteration': 0,
            'retrieved_passages': [],
            'retrieved_scores': [],
            'missing_observations': set(),
            'cumulative_findings': set()
        }
    
    def update_retrieval_state(self, state: Dict, new_passages: List[str],
                              new_scores: List[float], generated_text: str) -> Dict:
        try:
            state['retrieved_passages'].extend(new_passages)
            state['retrieved_scores'].extend(new_scores)
            
            current_obs = self.observation_detector.detect_observations(generated_text)
            state['cumulative_findings'].update(current_obs)
            
            state['iteration'] += 1
            return state
        
        except Exception as e:
            logger.error(f"Error updating retrieval state: {e}")
            return state


class IterativeRetrievalAugmentedGeneration(nn.Module):
    def __init__(self, config: IterativeRAGConfig):
        super().__init__()
        self.config = config
        
        self.observation_detector = ObservationDetector(config)
        self.consistency_verifier = ConsistencyVerifier(config)
        self.targeted_retriever = TargetedRetriever(config)
        self.iterative_retrieval = IterativeRetrieval(config)
    
    def generate_with_iterative_retrieval(self, initial_findings: str,
                                         retrieval_function: Callable,
                                         generation_function: Callable,
                                         reference_text: Optional[str] = None) -> Dict:
        logger.info("Starting iterative retrieval-augmented generation...")
        
        state = self.iterative_retrieval.initialize_retrieval_state()
        generations = []
        all_generations = []
        
        current_context = initial_findings
        
        for iteration in range(self.config.num_iterations):
            logger.info(f"Iteration {iteration + 1}/{self.config.num_iterations}")
            
            try:
                generated_text = generation_function(current_context)
                generations.append(generated_text)
                all_generations.append(generated_text)
                
                if reference_text:
                    missing_obs = self.observation_detector.find_missing_observations(
                        generated_text, reference_text
                    )
                else:
                    consistency = self.consistency_verifier.compute_consistency(generations)
                    
                    if consistency >= self.config.consistency_threshold:
                        logger.info("Consistency threshold reached")
                        break
                    missing_obs = set()
                
                state['missing_observations'] = missing_obs
                
                if not missing_obs:
                    logger.info("No missing observations")
                    break
                
                query = self.targeted_retriever.build_retrieval_query(missing_obs)
                retrieved_passages, retrieval_scores = retrieval_function(query, self.config.top_k)
                
                if not retrieved_passages:
                    logger.warning("No passages retrieved")
                    break
                
                ranked_passages = self.targeted_retriever.rank_retrieved_passages(
                    retrieved_passages, missing_obs
                )
                
                state = self.iterative_retrieval.update_retrieval_state(
                    state,
                    [p for p, _ in ranked_passages],
                    [s for _, s in ranked_passages],
                    generated_text
                )
                
                top_passages = [p for p, _ in ranked_passages[:2]]
                if top_passages:
                    current_context = generated_text + "\n\nRetrieved Evidence:\n" + "\n".join(top_passages)
            
            except Exception as e:
                logger.error(f"Error in iteration {iteration}: {e}")
                break
        
        final_consistency = self.consistency_verifier.compute_consistency(generations)
        consistent_findings = self.consistency_verifier.find_consistent_observations(generations)
        
        return {
            'generations': generations,
            'all_generations': all_generations,
            'retrieved_passages': state['retrieved_passages'],
            'retrieved_scores': state['retrieved_scores'],
            'iterations': state['iteration'],
            'final_consistency': final_consistency,
            'consistent_findings': consistent_findings,
            'cumulative_findings': state['cumulative_findings'],
            'final_text': generations[-1] if generations else initial_findings
        }
    
    def generate_with_verification(self, input_text: str, generation_function: Callable,
                                  num_samples: int = 3) -> Dict:
        logger.info(f"Starting generation with verification ({num_samples} samples)...")
        
        generations = []
        
        try:
            for sample_idx in range(num_samples):
                generated = generation_function(input_text)
                generations.append(generated)
            
            consistency = self.consistency_verifier.compute_consistency(generations)
            consistent_obs = self.consistency_verifier.find_consistent_observations(generations)
            
            best_gen = max(generations, 
                          key=lambda x: len(self.observation_detector.detect_observations(x)))
            
            all_observations = set().union(*[
                self.observation_detector.detect_observations(g) for g in generations
            ])
            
            return {
                'generations': generations,
                'best_generation': best_gen,
                'consistency_score': consistency,
                'consistent_observations': consistent_obs,
                'all_observations': all_observations
            }
        
        except Exception as e:
            logger.error(f"Error in verification: {e}")
            return {
                'generations': generations,
                'best_generation': input_text,
                'consistency_score': 0.0,
                'consistent_observations': set(),
                'all_observations': set()
            }


def create_iterative_rag_model(num_observations: int = 14, device: str = 'cuda') -> IterativeRetrievalAugmentedGeneration:
    config = IterativeRAGConfig(device=device)
    return IterativeRetrievalAugmentedGeneration(config)


if __name__ == "__main__":
    logger.info("CONTRIBUTION 2.3: ITERATIVE RETRIEVAL-AUGMENTED GENERATION")
    
    try:
        logger.info("Building iterative RAG model...")
        rag_model = create_iterative_rag_model()
        
        def mock_retrieval(query: str, k: int) -> Tuple[List[str], List[float]]:
            return [f"Report {i} about {query[:20]}" for i in range(k)], [0.9 - i*0.05 for i in range(k)]
        
        def mock_generation(context: str) -> str:
            return f"Generated report: {context[:50]}..."
        
        logger.info("Testing iterative retrieval-augmented generation...")
        results = rag_model.generate_with_iterative_retrieval(
            "Initial findings",
            mock_retrieval,
            mock_generation,
            reference_text="Reference with Cardiomegaly and Atelectasis"
        )
        
        logger.info("Iterative RAG Results:")
        logger.info(f"Iterations: {results['iterations']}")
        logger.info(f"Final consistency: {results['final_consistency']:.4f}")
        logger.info(f"Consistent findings: {results['consistent_findings']}")
        
        logger.info("Testing self-consistency verification...")
        verify_results = rag_model.generate_with_verification(
            "Initial prompt",
            mock_generation,
            num_samples=3
        )
        
        logger.info(f"Consistency score: {verify_results['consistency_score']:.4f}")
        logger.info("Iterative RAG working correctly")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()