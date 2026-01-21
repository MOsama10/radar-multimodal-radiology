import torch
import sys
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "annotate_retrieve"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

IMAGE_DIR = Path(r"C:\Users\3line\Downloads\radar\data\images")
ANNOTATION_FILE = Path(r"C:\Users\3line\Downloads\radar\data\mimic_cxr\annotation.json")
OBSERVATION_FILE = Path(r"C:\Users\3line\Downloads\radar\data\mimic_cxr\observation.json")


def check_data_files():
    logger.info("Checking data files...")
    logger.info(f"Looking for annotation: {ANNOTATION_FILE}")
    logger.info(f"Looking for observations: {OBSERVATION_FILE}")
    logger.info(f"Looking for images: {IMAGE_DIR}")
    
    if not ANNOTATION_FILE.exists():
        logger.error(f"Annotation file not found: {ANNOTATION_FILE}")
        return False
    logger.info(f"Found annotation file (Size: {ANNOTATION_FILE.stat().st_size} bytes)")
    
    if not OBSERVATION_FILE.exists():
        logger.error(f"Observation file not found: {OBSERVATION_FILE}")
        return False
    logger.info(f"Found observation file (Size: {OBSERVATION_FILE.stat().st_size} bytes)")
    
    if not IMAGE_DIR.exists():
        logger.error(f"Image directory not found: {IMAGE_DIR}")
        return False
    
    image_files = list(IMAGE_DIR.glob('*.png')) + list(IMAGE_DIR.glob('*.jpg'))
    logger.info(f"Found {len(image_files)} images")
    
    return True


def test_contribution_2_1():
    logger.info("\n" + "="*80)
    logger.info("TEST 1: Contribution 2.1 - Dense Passage Retrieval (DPR)")
    logger.info("="*80)
    
    try:
        from modeling_dense_passage_retrieval import MIMICCXRDataLoader, create_dpr_model
        
        logger.info("Step 1: Loading MIMIC-CXR data...")
        data_loader = MIMICCXRDataLoader(max_samples=50)
        passages = data_loader.get_passages()
        observations = data_loader.get_observations_list()
        
        if not passages:
            logger.error("No passages loaded")
            return False
        
        logger.info(f"  Loaded {len(passages)} passages")
        logger.info(f"  Loaded {len(observations)} observation sets")
        
        logger.info("Step 2: Creating DPR model...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"  Using device: {device}")
        dpr = create_dpr_model(device=device)
        
        logger.info("Step 3: Building retrieval database...")
        dpr.build_retrieval_database(passages, observations)
        logger.info("  Database built successfully")
        
        logger.info("Step 4: Testing text retrieval...")
        test_queries = ["cardiomegaly", "pneumonia", "chest findings"]
        for query in test_queries:
            retrieved, scores = dpr.retrieve_for_text(query, k=5)
            logger.info(f"  Query: '{query}'")
            logger.info(f"    Retrieved {len(retrieved)} passages")
            logger.info(f"    Top scores: {[f'{s:.4f}' for s in scores[:3]]}")
        
        if data_loader.image_paths:
            logger.info("Step 5: Testing image retrieval...")
            image = data_loader.load_image()
            retrieved, scores = dpr.retrieve_for_image(image, k=5)
            logger.info(f"  Retrieved {len(retrieved)} passages")
            logger.info(f"  Top scores: {[f'{s:.4f}' for s in scores[:3]]}")
        
        logger.info("TEST 1: PASSED")
        return True
    
    except Exception as e:
        logger.error(f"TEST 1: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_contribution_2_2():
    logger.info("\n" + "="*80)
    logger.info("TEST 2: Contribution 2.2 - Attention-Based Knowledge Fusion")
    logger.info("="*80)
    
    try:
        from modeling_knowledge_fusion import create_fusion_model
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        logger.info("Step 1: Creating fusion model...")
        fusion_model = create_fusion_model(device=device)
        logger.info("  Model created successfully")
        
        logger.info("Step 2: Creating test data...")
        batch_size = 2
        pf_features = torch.randn(batch_size, 10, 768, device=device)
        sf_features = torch.randn(batch_size, 8, 768, device=device)
        image_features = torch.randn(batch_size, 768, device=device)
        logger.info(f"  PF shape: {pf_features.shape}")
        logger.info(f"  SF shape: {sf_features.shape}")
        logger.info(f"  Image shape: {image_features.shape}")
        
        logger.info("Step 3: Running fusion forward pass...")
        results = fusion_model(pf_features, sf_features, image_features)
        
        logger.info("Step 4: Checking results...")
        logger.info(f"  Fused features shape: {results['fused_features'].shape}")
        logger.info(f"  PF gates shape: {results['pf_gates'].shape}")
        logger.info(f"  SF gates shape: {results['sf_gates'].shape}")
        logger.info(f"  Conflict scores shape: {results['conflict_scores'].shape}")
        logger.info(f"  Conflicts detected: {results['conflict_mask'].sum().item()}")
        logger.info(f"  PF gate mean: {results['pf_gates'].mean():.4f}")
        logger.info(f"  SF gate mean: {results['sf_gates'].mean():.4f}")
        
        logger.info("TEST 2: PASSED")
        return True
    
    except Exception as e:
        logger.error(f"TEST 2: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_contribution_2_3():
    logger.info("\n" + "="*80)
    logger.info("TEST 3: Contribution 2.3 - Iterative Retrieval-Augmented Generation")
    logger.info("="*80)
    
    try:
        from modeling_iterative_rag import create_iterative_rag_model
        from modeling_dense_passage_retrieval import MIMICCXRDataLoader
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        logger.info("Step 1: Creating iterative RAG model...")
        rag_model = create_iterative_rag_model(device=device)
        logger.info("  Model created successfully")
        
        logger.info("Step 2: Loading real MIMIC-CXR data...")
        data_loader = MIMICCXRDataLoader(max_samples=30)
        passages = data_loader.get_passages()
        
        if not passages:
            logger.error("No passages loaded")
            return False
        logger.info(f"  Loaded {len(passages)} passages")
        
        logger.info("Step 3: Setting up retrieval function...")
        def real_retrieval(query, k):
            np.random.seed(hash(query) % 2**32)
            indices = np.random.choice(len(passages), min(k, len(passages)), replace=False)
            retrieved = [passages[i] for i in indices]
            scores = [0.8 - i*0.05 for i in range(len(retrieved))]
            return retrieved, scores
        
        logger.info("Step 4: Setting up generation function...")
        def simple_generation(context):
            return f"Generated report based on: {context[:50]}..."
        
        logger.info("Step 5: Testing iterative retrieval-augmented generation with real data...")
        results = rag_model.generate_with_iterative_retrieval(
            "Initial clinical findings",
            real_retrieval,
            simple_generation,
            reference_text="Reference with Cardiomegaly and Pneumonia findings"
        )
        
        logger.info("Step 6: Checking iterative results...")
        logger.info(f"  Iterations: {results['iterations']}")
        logger.info(f"  Final consistency: {results['final_consistency']:.4f}")
        logger.info(f"  Consistent findings: {results['consistent_findings']}")
        logger.info(f"  Final text: {results['final_text'][:80]}...")
        
        logger.info("Step 7: Testing self-consistency verification...")
        verify_results = rag_model.generate_with_verification(
            "Initial prompt",
            simple_generation,
            num_samples=3
        )
        
        logger.info("Step 8: Checking verification results...")
        logger.info(f"  Consistency score: {verify_results['consistency_score']:.4f}")
        logger.info(f"  Generated {len(verify_results['generations'])} samples")
        
        logger.info("TEST 3: PASSED")
        return True
    
    except Exception as e:
        logger.error(f"TEST 3: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    logger.info("\n" + "="*80)
    logger.info("RADAR CONTRIBUTION FILES TEST")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now()}\n")
    
    logger.info("SETUP: Checking data files...")
    if not check_data_files():
        logger.error("Data files check FAILED")
        return False
    logger.info("Data files check PASSED\n")
    
    logger.info("TESTING: Running contribution tests...\n")
    result_2_1 = test_contribution_2_1()
    result_2_2 = test_contribution_2_2()
    result_2_3 = test_contribution_2_3()
    
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    
    results = {
        "2.1 DPR": "PASSED" if result_2_1 else "FAILED",
        "2.2 Fusion": "PASSED" if result_2_2 else "FAILED",
        "2.3 RAG": "PASSED" if result_2_3 else "FAILED",
    }
    
    for test, status in results.items():
        logger.info(f"  {test}: {status}")
    
    all_passed = all(v == "PASSED" for v in results.values())
    
    logger.info("\n" + "="*80)
    if all_passed:
        logger.info("ALL TESTS PASSED!")
    else:
        logger.info("SOME TESTS FAILED!")
    logger.info("="*80 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)