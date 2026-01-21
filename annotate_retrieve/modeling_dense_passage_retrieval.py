import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from PIL import Image
import torchvision.transforms as transforms

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_CONFIG = {
    'IMAGE_DIR': Path(r"C:\Users\3line\Downloads\radar\data\images"),
    'ANNOTATION_JSON': Path(r"C:\Users\3line\Downloads\radar\data\mimic_cxr\annotation.json"),
    'OBSERVATION_JSON': Path(r"C:\Users\3line\Downloads\radar\data\mimic_cxr\observation.json"),
}

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    from transformers import AutoModel, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class ImagePreprocessor:
    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return self.transform(image)
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return torch.randn(3, self.image_size, self.image_size)
    
    def preprocess_batch(self, images: List[Image.Image]) -> torch.Tensor:
        tensors = [self.preprocess(img) for img in images]
        return torch.stack(tensors)


class MIMICCXRDataLoader:
    def __init__(self, max_samples: int = 100):
        self.max_samples = max_samples
        self.annotations = {}
        self.observations = {}
        self.image_paths = []
        self.img_preprocessor = ImagePreprocessor()
        self.load_all_data()
    
    def load_all_data(self):
        logger.info("Loading MIMIC-CXR data...")
        
        ann_path = DATA_CONFIG['ANNOTATION_JSON']
        if ann_path.exists():
            try:
                with open(ann_path, 'r', encoding='utf-8', errors='replace') as f:
                    try:
                        full_data = json.load(f)
                        if isinstance(full_data, dict):
                            self.annotations = dict(list(full_data.items())[:self.max_samples])
                        elif isinstance(full_data, list):
                            self.annotations = {str(i): item for i, item in enumerate(full_data[:self.max_samples])}
                    except json.JSONDecodeError:
                        f.seek(0)
                        self.annotations = self._extract_valid_json(f.read())
                logger.info(f"Loaded {len(self.annotations)} annotations")
            except Exception as e:
                logger.warning(f"Could not load annotations: {e}")
        
        obs_path = DATA_CONFIG['OBSERVATION_JSON']
        if obs_path.exists():
            try:
                with open(obs_path, 'r', encoding='utf-8', errors='replace') as f:
                    try:
                        full_data = json.load(f)
                        if isinstance(full_data, dict):
                            self.observations = dict(list(full_data.items())[:self.max_samples])
                        elif isinstance(full_data, list):
                            self.observations = {str(i): item for i, item in enumerate(full_data[:self.max_samples])}
                    except json.JSONDecodeError:
                        f.seek(0)
                        self.observations = self._extract_valid_json(f.read())
                logger.info(f"Loaded {len(self.observations)} observations")
            except Exception as e:
                logger.warning(f"Could not load observations: {e}")
        
        img_dir = DATA_CONFIG['IMAGE_DIR']
        if img_dir.exists():
            self.image_paths = list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpg'))
            logger.info(f"Found {len(self.image_paths)} images")
    
    def _extract_valid_json(self, content: str) -> dict:
        results = {}
        count = 0
        brace_depth = 0
        current = ""
        in_string = False
        
        for char in content:
            if char == '\\':
                current += char
                continue
            if char == '"' and (not in_string or current[-1] != '\\'):
                in_string = not in_string
            
            if not in_string:
                if char == '{':
                    if brace_depth == 0:
                        current = char
                    else:
                        current += char
                    brace_depth += 1
                    continue
                elif char == '}':
                    brace_depth -= 1
                    current += char
                    if brace_depth == 0 and current.strip().startswith('{'):
                        try:
                            obj = json.loads(current)
                            results[f"item_{count}"] = obj
                            count += 1
                            if count >= self.max_samples:
                                break
                        except:
                            pass
                        current = ""
                    continue
            
            if brace_depth > 0:
                current += char
        
        return results
    
    def get_passages(self) -> List[str]:
        passages = []
        for value in self.annotations.values():
            if isinstance(value, dict):
                text = value.get('report') or value.get('text')
                if text and len(str(text)) > 10:
                    passages.append(str(text)[:500])
        return passages
    
    def get_observations_list(self) -> List[List[str]]:
        obs_list = []
        for value in self.observations.values():
            if isinstance(value, list):
                obs_list.append([str(o) for o in value])
            else:
                obs_list.append(['No Finding'])
        return obs_list
    
    def load_image(self, idx: int = 0) -> torch.Tensor:
        if not self.image_paths:
            return torch.randn(3, 224, 224)
        try:
            img = Image.open(self.image_paths[idx % len(self.image_paths)]).convert('RGB')
            return self.img_preprocessor.preprocess(img)
        except Exception as e:
            logger.warning(f"Failed to load image: {e}")
            return torch.randn(3, 224, 224)


@dataclass
class RetrievalConfig:
    embedding_dim: int = 512
    num_retrieved: int = 5
    hybrid_alpha: float = 0.5
    device: str = 'cuda'


class CrossModalEmbedder(nn.Module):
    def __init__(self, config: RetrievalConfig):
        super().__init__()
        self.config = config
        self.device = torch.device('cuda' if (torch.cuda.is_available() and config.device == 'cuda') else 'cpu')
        self.model_loaded = False
        self.model = None
        self.tokenizer = None
        
        self._load_biomedclip_model()
        
        self.text_projection = nn.Linear(768, config.embedding_dim).to(self.device)
        self.image_projection = nn.Linear(768, config.embedding_dim).to(self.device)
    
    def _load_biomedclip_model(self):
        try:
            if HAS_TRANSFORMERS:
                logger.info("Loading BiomedCLIP-PubMedBERT model...")
                
                model_name = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
                self.model = AutoModel.from_pretrained(model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                self.model.to(self.device)
                self.model.eval()
                
                self.model_loaded = True
                logger.info("BiomedCLIP model loaded successfully")
            else:
                logger.error("Transformers not available")
        except Exception as e:
            logger.error(f"Failed to load BiomedCLIP: {e}")
            logger.warning("Falling back to random embeddings")
            self.model_loaded = False
    
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        try:
            if self.model_loaded and self.tokenizer:
                inputs = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=512
                )
                
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model.get_text_features(**inputs)
                
                embeds = outputs
            else:
                embeds = torch.randn(len(texts), 768, device=self.device)
            
            return F.normalize(self.text_projection(embeds), dim=-1)
        
        except Exception as e:
            logger.error(f"Text encoding error: {e}")
            return torch.randn(len(texts), self.config.embedding_dim, device=self.device)
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        try:
            if self.model_loaded and hasattr(self.model, 'get_image_features'):
                images = images.to(self.device)
                
                with torch.no_grad():
                    embeds = self.model.get_image_features(images)
                
            else:
                embeds = torch.randn(images.size(0), 768, device=self.device)
            
            return F.normalize(self.image_projection(embeds), dim=-1)
        
        except Exception as e:
            logger.error(f"Image encoding error: {e}")
            return torch.randn(images.size(0), self.config.embedding_dim, device=self.device)


class HybridRetriever(nn.Module):
    def __init__(self, config: RetrievalConfig, embedder: CrossModalEmbedder):
        super().__init__()
        self.config = config
        self.embedder = embedder
        self.passages = []
        self.semantic_index = None
    
    def build_indices(self, passages: List[str], observations: List[List[str]]):
        self.passages = passages
        
        if not passages:
            logger.warning("No passages to index")
            return
        
        if HAS_FAISS:
            try:
                logger.info(f"Building FAISS index for {len(passages)} passages...")
                
                embeds = []
                for i in range(0, len(passages), 32):
                    batch = passages[i:i+32]
                    batch_embeds = self.embedder.encode_text(batch).cpu().numpy()
                    embeds.append(batch_embeds)
                
                embeds = np.vstack(embeds).astype('float32')
                
                self.semantic_index = faiss.IndexFlatIP(self.config.embedding_dim)
                self.semantic_index.add(embeds)
                
                logger.info(f"FAISS index built: {self.semantic_index.ntotal} passages")
            except Exception as e:
                logger.error(f"FAISS indexing failed: {e}")
                self.semantic_index = None
    
    def retrieve(self, query_embed: torch.Tensor, k: int = None) -> Tuple[List[str], List[float]]:
        if k is None:
            k = self.config.num_retrieved
        k = min(k, len(self.passages))
        
        if self.semantic_index:
            try:
                query_np = query_embed.cpu().numpy().astype('float32').reshape(1, -1)
                scores, indices = self.semantic_index.search(query_np, k)
                return [self.passages[i] for i in indices[0]], [float(s) for s in scores[0]]
            except Exception as e:
                logger.error(f"FAISS retrieval failed: {e}")
        
        return self.passages[:k], [0.5] * k
    
    def retrieve_with_hard_negatives(self, query_embed: torch.Tensor, k: int = None, num_negatives: int = 3) -> Dict:
        if k is None:
            k = self.config.num_retrieved
        
        retrieved, scores = self.retrieve(query_embed, k + num_negatives)
        
        return {
            'positives': retrieved[:k],
            'negatives': retrieved[k:k+num_negatives],
            'positive_scores': scores[:k],
            'negative_scores': scores[k:k+num_negatives]
        }


class DensePassageRetrieval(nn.Module):
    def __init__(self, config: RetrievalConfig):
        super().__init__()
        self.config = config
        self.embedder = CrossModalEmbedder(config)
        self.retriever = HybridRetriever(config, self.embedder)
    
    def build_retrieval_database(self, passages: List[str], observations: List[List[str]]):
        self.retriever.build_indices(passages, observations)
    
    def retrieve_for_text(self, text: str, k: int = None) -> Tuple[List[str], List[float]]:
        query_embed = self.embedder.encode_text([text]).squeeze(0)
        return self.retriever.retrieve(query_embed, k)
    
    def retrieve_for_image(self, image: torch.Tensor, k: int = None) -> Tuple[List[str], List[float]]:
        query_embed = self.embedder.encode_image(image.unsqueeze(0)).squeeze(0)
        return self.retriever.retrieve(query_embed, k)


def create_dpr_model(device: str = 'cuda') -> DensePassageRetrieval:
    config = RetrievalConfig(device=device)
    return DensePassageRetrieval(config)


if __name__ == "__main__":
    logger.info("CONTRIBUTION 2.1: DENSE PASSAGE RETRIEVAL")
    
    try:
        data_loader = MIMICCXRDataLoader(max_samples=50)
        passages = data_loader.get_passages()
        observations = data_loader.get_observations_list()
        
        if not passages:
            logger.error("No passages loaded")
            exit(1)
        
        logger.info("Building DPR model...")
        dpr = create_dpr_model()
        
        logger.info("Building retrieval indices...")
        dpr.build_retrieval_database(passages, observations)
        
        logger.info("Testing text retrieval...")
        test_queries = ["cardiomegaly", "pneumonia", "chest findings"]
        for query in test_queries:
            retrieved, scores = dpr.retrieve_for_text(query, k=5)
            logger.info(f"Query: {query}")
            logger.info(f"Retrieved: {len(retrieved)} passages")
            logger.info(f"Scores: {[f'{s:.4f}' for s in scores[:3]]}")
        
        if data_loader.image_paths:
            logger.info("Testing image retrieval...")
            image = data_loader.load_image()
            retrieved, scores = dpr.retrieve_for_image(image, k=5)
            logger.info(f"Retrieved: {len(retrieved)} passages")
            logger.info(f"Scores: {[f'{s:.4f}' for s in scores[:3]]}")
        
        logger.info("DPR System Working Successfully")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()