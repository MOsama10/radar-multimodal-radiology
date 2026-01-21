import json
import logging
from pathlib import Path
from typing import Dict, Tuple, List
import pandas as pd
from tqdm import tqdm
import shutil
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FindingsExtractor:
    """Extract medical findings from radiology reports"""
    
    # Common chest X-ray findings
    FINDINGS_KEYWORDS = {
        # Lungs
        "atelectasis": ["atelectasis", "atelect", "collapse"],
        "pneumonia": ["pneumonia", "pneumonic", "infiltrate"],
        "pneumothorax": ["pneumothorax", "ptx"],
        "pleural_effusion": ["pleural effusion", "effusion", "fluid"],
        "pulmonary_edema": ["pulmonary edema", "edema"],
        "consolidation": ["consolidation", "consolidate"],
        "emphysema": ["emphysema"],
        "fibrosis": ["fibrosis", "fibroid"],
        "nodule": ["nodule", "nodular"],
        "mass": ["mass", "lesion"],
        "opacity": ["opacity", "opacit"],
        
        # Heart
        "cardiomegaly": ["cardiomegaly", "cardiac enlargement", "enlarged heart"],
        "normal_cardiac": ["normal cardiac", "normal heart size", "normal silhouette"],
        "pericardial_effusion": ["pericardial effusion"],
        
        # Bones
        "fracture": ["fracture", "fractured"],
        "osteoporosis": ["osteoporosis"],
        "degenerative": ["degenerative", "spondylosis"],
        
        # Devices
        "central_line": ["central line", "central venous"],
        "chest_tube": ["chest tube", "chest drain"],
        "pacemaker": ["pacemaker"],
        "defibrillator": ["icd", "defibrillator"],
        "endotracheal_tube": ["endotracheal", "etube", "intubat"],
        "nasogastric_tube": ["nasogastric", "ng tube"],
        "foley_catheter": ["foley", "catheter"],
        "ijv_catheter": ["internal jugular", "ijv"],
        "picc_line": ["picc line", "picc"],
        "port": ["port", "implanted port"],
        
        # Other
        "surgical_clips": ["surgical clip", "clip", "suture"],
        "post_surgical": ["post surgical", "postoperative", "post op"],
        "hyperinflation": ["hyperinflation"],
        "low_lung_volume": ["low lung volume", "low volume"],
        "normal": ["no acute", "no significant", "no evidence", "normal exam"],
    }

    @staticmethod
    def extract_findings(report: str) -> List[str]:
        """
        Extract findings from medical report text
        Returns list of finding labels found in the report
        """
        if not report or pd.isna(report):
            return []
        
        report_lower = report.lower()
        findings = []
        
        for finding_label, keywords in FindingsExtractor.FINDINGS_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in report_lower:
                    if finding_label not in findings:
                        findings.append(finding_label)
                    break
        
        return sorted(findings)


class MIMICDataLoader:
    CSV_PATH = r"C:\Users\3line\Downloads\radar\radar-multimodal-radiology\scripts\Cxr_df.csv"
    MIMIC_IMAGES_SOURCE = r"C:\Users\3line\Downloads\mimic_dset"  
    OUTPUT_DIR = Path("data/mimic_cxr")

    def __init__(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.OUTPUT_DIR / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.extractor = FindingsExtractor()

    def inspect_csv(self) -> Tuple[pd.DataFrame, Dict]:
        """Load and inspect the CSV file"""
        if not Path(self.CSV_PATH).exists():
            raise FileNotFoundError(f"CSV not found: {self.CSV_PATH}")

        df = pd.read_csv(self.CSV_PATH)

        logger.info(f"Total rows: {len(df)}")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info("\nFirst 3 rows:")
        logger.info(df.head(3).to_string())
        logger.info(f"\nMissing values:\n{df.isnull().sum()}")

        return df, {
            "columns": list(df.columns),
            "missing": df.isnull().sum().to_dict(),
            "shape": df.shape
        }

    def identify_columns(self, df: pd.DataFrame) -> Dict:
        """Auto-detect important columns in CSV"""
        mapping = {}

        id_candidates = ['id', 'image_id', 'study_id', 'subject_id']
        image_candidates = ['image', 'path', 'file', 'filename']
        report_candidates = ['report', 'text', 'finding', 'impression']

        for col in df.columns:
            col_lower = col.lower()
            if any(c in col_lower for c in id_candidates):
                mapping['id'] = col
                break

        for col in df.columns:
            col_lower = col.lower()
            if any(c in col_lower for c in image_candidates):
                mapping['image'] = col
                break

        for col in df.columns:
            col_lower = col.lower()
            if any(c in col_lower for c in report_candidates):
                mapping['report'] = col
                break

        if 'id' not in mapping:
            mapping['id'] = df.columns[0]

        logger.info(f"\nDetected columns mapping: {mapping}")
        return mapping

    def create_annotation(self, df: pd.DataFrame, mapping: Dict):
        """
        Create annotation.json with findings extracted from reports
        """
        out_path = self.OUTPUT_DIR / "annotation.json"
        annotation = {}

        logger.info("\nCreating annotations with extracted findings...")
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
            try:
                sid = str(row[mapping['id']])
                report_text = str(row[mapping['report']]).strip() if mapping.get('report') and pd.notna(row[mapping['report']]) else ""
                
                # Extract findings from report
                findings = self.extractor.extract_findings(report_text)
                
                # Create relative path for images
                img_path = f"images/{sid}.jpg"
                
                annotation[sid] = {
                    "image_id": sid,
                    "image_path": img_path,
                    "report": report_text,
                    "findings": findings  # Now populated with extracted findings!
                }
            except Exception as e:
                logger.warning(f"Error processing row: {e}")
                continue

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(annotation, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ Annotation file created: {out_path}")
        logger.info(f"✓ Total annotated samples: {len(annotation)}")
        
        return annotation

    

    def create_splits(self, annotation: Dict, train_ratio=0.8, val_ratio=0.1):
        """
        Create train/val/test splits
        """
        import random
        
        ids = list(annotation.keys())
        random.shuffle(ids)
        
        n = len(ids)
        train_n = int(n * train_ratio)
        val_n = int(n * val_ratio)
        
        splits = {
            "train": ids[:train_n],
            "val": ids[train_n:train_n + val_n],
            "test": ids[train_n + val_n:]
        }
        
        splits_path = self.OUTPUT_DIR / "splits.json"
        with open(splits_path, "w", encoding="utf-8") as f:
            json.dump(splits, f, indent=2)
        
        logger.info(f"\n✓ Data splits created:")
        logger.info(f"  - Train: {len(splits['train'])} ({train_ratio*100:.0f}%)")
        logger.info(f"  - Val: {len(splits['val'])} ({val_ratio*100:.0f}%)")
        logger.info(f"  - Test: {len(splits['test'])} ({(1-train_ratio-val_ratio)*100:.0f}%)")
        
        return splits

    def generate_findings_stats(self, annotation: Dict):
        """
        Generate statistics about findings distribution
        """
        findings_count = {}
        
        for item in annotation.values():
            for finding in item.get("findings", []):
                findings_count[finding] = findings_count.get(finding, 0) + 1
        
        stats_path = self.OUTPUT_DIR / "findings_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(findings_count, f, indent=2)
        
        logger.info(f"\n✓ Findings statistics:")
        sorted_findings = sorted(findings_count.items(), key=lambda x: x[1], reverse=True)
        for finding, count in sorted_findings[:15]:  # Top 15
            logger.info(f"  - {finding}: {count}")
        
        return findings_count


def main():
    """Main execution"""
    loader = MIMICDataLoader()
    
    # Step 1: Inspect CSV
    logger.info("=" * 60)
    logger.info("STEP 1: Inspecting CSV file")
    logger.info("=" * 60)
    df, csv_info = loader.inspect_csv()
    
    # Step 2: Identify columns
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Identifying columns")
    logger.info("=" * 60)
    mapping = loader.identify_columns(df)
    
    # Step 3: Create annotation JSON with findings extraction
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Creating annotation.json with extracted findings")
    logger.info("=" * 60)
    annotation = loader.create_annotation(df, mapping)

    
    # Step 5: Create train/val/test splits
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Creating data splits")
    logger.info("=" * 60)
    splits = loader.create_splits(annotation)
    
    # Step 6: Generate findings statistics
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: Generating findings statistics")
    logger.info("=" * 60)
    stats = loader.generate_findings_stats(annotation)
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ SETUP COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"\n data structure:")
    logger.info(f"data/mimic_cxr/")
    logger.info(f"  annotation.json     ")
    logger.info(f"  splits.json     ")

if __name__ == "__main__":
    main()