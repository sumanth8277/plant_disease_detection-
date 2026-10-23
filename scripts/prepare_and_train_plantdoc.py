"""
PlantDoc Ingestion, Training & Agronomic Recovery Pipeline (scripts/prepare_and_train_plantdoc.py)
=================================================================================================
Ingests the real-world field-condition PlantDoc dataset from GitHub, maps its 28 classes
to project standards, trains a MobileNetV2 candidate model, evaluates against existing models,
and generates full pesticide and fertilizer recovery recommendations for all PlantDoc diseases.
"""

import csv
import hashlib
import json
import logging
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Configure Windows stdout encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from services.image_preprocessing import load_and_preprocess_image

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("plantdoc_pipeline")

PLANTDOC_ZIP_URL = "https://codeload.github.com/pratikkayal/PlantDoc-Dataset/zip/refs/heads/master"
PLANTDOC_DIR = BASE_DIR / "dataset" / "plantdoc"
CANDIDATES_DIR = BASE_DIR / "models" / "candidates"
REPORTS_DIR = BASE_DIR / "reports" / "candidates"

RANDOM_SEED = 42

# Official mapping: PlantDoc directory name -> 38-class project name
PLANTDOC_CLASS_MAPPING = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Peach leaf": "Peach___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato leaf": "Tomato___healthy",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
    "grape leaf black rot": "Grape___Black_rot",
    "grape leaf": "Grape___healthy",
}

# Curated Agronomic Recovery Guide for PlantDoc Diseases
PLANTDOC_AGRONOMIC_GUIDE = {
    "Apple___Apple_scab": {
        "disease_name": "Apple Scab (Venturia inaequalis)",
        "symptoms": "Olive-green to black velvety spots on leaf surface, leading to leaf distortion and premature yellowing.",
        "chemical_pesticides": [
            {"name": "Difenoconazole (Score 25 EC)", "dosage": "0.5 ml/L", "phi_days": 14, "action": "Systemic curative fungicide"},
            {"name": "Mancozeb (Dithane M-45)", "dosage": "2.5 g/L", "phi_days": 21, "action": "Broad-spectrum contact protectant"},
        ],
        "organic_pesticides": [
            {"name": "Copper Hydroxide (Kocide)", "dosage": "2.0 g/L", "phi_days": 7, "action": "Bactericide/Fungicide protectant"},
            {"name": "Potassium Bicarbonate", "dosage": "3.0 g/L", "phi_days": 1, "action": "pH disruption of fungal spores"},
        ],
        "fertilizer_recovery": (
            "Avoid excessive early-season Nitrogen (N) which produces soft, susceptible foliage. "
            "Apply Potassium Sulfate (0:0:50) at 5 g/L to strengthen epidermal cell walls. Supplement with zinc and boron foliar sprays."
        ),
        "ipm_recovery_practices": "Rake and compost or destroy fallen overwintered leaves. Prune tree canopy to maximize air movement and sunlight penetration.",
    },
    "Apple___Cedar_apple_rust": {
        "disease_name": "Cedar Apple Rust (Gymnosporangium juniperi-virginianae)",
        "symptoms": "Bright yellow-orange or reddish spots on upper leaf surface, later forming cylindrical tube structures on leaf undersides.",
        "chemical_pesticides": [
            {"name": "Myclobutanil (Systhane 20 EW)", "dosage": "0.4 ml/L", "phi_days": 14, "action": "Sterol inhibitor systemic fungicide"},
            {"name": "Mancozeb (75% WP)", "dosage": "2.0 g/L", "phi_days": 21, "action": "Foliar protectant before rain events"},
        ],
        "organic_pesticides": [
            {"name": "Sulfur Wettable Powder", "dosage": "3.0 g/L", "phi_days": 3, "action": "Multi-site fungal spore inhibitor"},
        ],
        "fertilizer_recovery": "Apply balanced 10-10-10 with humic acid to relieve metabolic rust stress. Avoid foliage-wetting irrigation.",
        "ipm_recovery_practices": "Eradicate nearby eastern red cedar / juniper trees within a 500m radius if possible to break the alternating spore cycle.",
    },
    "Pepper,_bell___Bacterial_spot": {
        "disease_name": "Bell Pepper Bacterial Spot (Xanthomonas campestris)",
        "symptoms": "Small, water-soaked dark lesions becoming circular with pale brown centers and yellow chlorotic margins.",
        "chemical_pesticides": [
            {"name": "Copper Oxychloride + Streptomycin Sulfate (Plantomycin)", "dosage": "2.5 g + 0.1 g/L", "phi_days": 7, "action": "Contact bactericidal spray"},
        ],
        "organic_pesticides": [
            {"name": "Bacillus subtilis (Bio-fungicide/bactericide)", "dosage": "5.0 ml/L", "phi_days": 0, "action": "Competitive exclusion of pathogen"},
            {"name": "Copper Octanoate (Copper Soap)", "dosage": "3.0 ml/L", "phi_days": 1, "action": "Gentle organic copper barrier"},
        ],
        "fertilizer_recovery": "Foliar spray Calcium Chloride (2 g/L) + Magnesium Sulfate (3 g/L) to prevent blossom-end rot and fortify leaf tissue.",
        "ipm_recovery_practices": "Do not enter fields when plants are wet. Sanitize pruning tools with 10% sodium hypochlorite between rows.",
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "disease_name": "Corn Gray Leaf Spot (Cercospora zeae-maydis)",
        "symptoms": "Narrow, rectangular lesions restricted by leaf veins; lesions turn grayish-tan with distinct parallel borders.",
        "chemical_pesticides": [
            {"name": "Azoxystrobin + Difenoconazole (Amistar Top)", "dosage": "1.0 ml/L", "phi_days": 14, "action": "QoI and DMI dual-action systemic"},
            {"name": "Pyraclostrobin (Headline)", "dosage": "0.8 ml/L", "phi_days": 21, "action": "Strobilurin broad-spectrum fungicide"},
        ],
        "organic_pesticides": [
            {"name": "Trichoderma viride Bio-formulation", "dosage": "5.0 g/L", "phi_days": 0, "action": "Microbial bio-antagonist"},
        ],
        "fertilizer_recovery": "Ensure adequate soil Potassium (K). Low potassium increases susceptibility to stalk and leaf blights.",
        "ipm_recovery_practices": "Incorporate crop residues into soil post-harvest. Practice 2-year rotation with non-host crops such as soybean.",
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "disease_name": "Corn Northern Leaf Blight (Exserohilum turcicum)",
        "symptoms": "Long, elliptical cigar-shaped grayish-green lesions (2.5 to 15 cm long) on lower leaves progressing upwards.",
        "chemical_pesticides": [
            {"name": "Propiconazole (Tilt 25 EC)", "dosage": "1.0 ml/L", "phi_days": 14, "action": "Systemic triazole protectant/curative"},
            {"name": "Mancozeb (75% WP)", "dosage": "2.5 g/L", "phi_days": 21, "action": "Contact foliar protectant"},
        ],
        "organic_pesticides": [
            {"name": "Neem Seed Kernel Extract (NSKE 5%)", "dosage": "50 ml/L", "phi_days": 1, "action": "Botanical antifungal extract"},
        ],
        "fertilizer_recovery": "Apply balanced N-P-K (120:60:40 kg/ha). Split nitrogen applications to avoid excessive lush vegetative growth.",
        "ipm_recovery_practices": "Plant blight-resistant hybrids with Ht genes. Plow under infected crop debris after harvest.",
    },
    "Corn_(maize)___Common_rust_": {
        "disease_name": "Corn Common Rust (Puccinia sorghi)",
        "symptoms": "Small, circular to elongate golden-brown to cinnamon pustules rupturing both upper and lower leaf surfaces.",
        "chemical_pesticides": [
            {"name": "Tebuconazole (Folicur 250 EC)", "dosage": "1.0 ml/L", "phi_days": 14, "action": "Curative systemic fungicide"},
            {"name": "Mancozeb (75% WP)", "dosage": "2.0 g/L", "phi_days": 14, "action": "Preventive contact protectant"},
        ],
        "organic_pesticides": [
            {"name": "Wettable Sulfur (80% WDG)", "dosage": "2.5 g/L", "phi_days": 3, "action": "Contact protective fungicide"},
        ],
        "fertilizer_recovery": "Boost Potassium (MOP or SOP @ 50 kg/ha) to accelerate tissue healing. Maintain balanced soil fertility.",
        "ipm_recovery_practices": "Early planting helps crops mature before high rust spore densities arrive from southern regions.",
    },
    "Potato___Early_blight": {
        "disease_name": "Potato Early Blight (Alternaria solani)",
        "symptoms": "Target-like circular brown lesions with concentric rings, surrounded by yellow chlorotic margins on older leaves.",
        "chemical_pesticides": [
            {"name": "Chlorothalonil (Kavach 75 WP)", "dosage": "2.0 g/L", "phi_days": 7, "action": "Broad-spectrum contact protectant"},
            {"name": "Azoxystrobin (Amistar 25 SC)", "dosage": "0.5 ml/L", "phi_days": 14, "action": "Preventive and curative systemic"},
        ],
        "organic_pesticides": [
            {"name": "Copper Oxychloride (50% WP)", "dosage": "2.5 g/L", "phi_days": 5, "action": "Protective copper barrier"},
            {"name": "Neem Oil Extract (1500 ppm)", "dosage": "5.0 ml/L", "phi_days": 0, "action": "Botanical fungal spore suppressant"},
        ],
        "fertilizer_recovery": "Maintain adequate Nitrogen and Potassium. Stressed or under-fertilized potato vines are significantly more prone to early blight.",
        "ipm_recovery_practices": "Ensure minimum 3-year crop rotation. Use drip irrigation to prevent free moisture on leaves.",
    },
    "Potato___Late_blight": {
        "disease_name": "Potato Late Blight (Phytophthora infestans)",
        "symptoms": "Dark, water-soaked necrotic lesions on leaf margins with white fungal mildew visible on leaf undersides in humid conditions.",
        "chemical_pesticides": [
            {"name": "Metalaxyl + Mancozeb (Ridomil Gold 68 WG)", "dosage": "2.5 g/L", "phi_days": 14, "action": "Systemic and contact oomycete killer"},
            {"name": "Dimethomorph (Acrobat 50 WP)", "dosage": "1.0 g/L", "phi_days": 7, "action": "Cell wall disruptor systemic"},
        ],
        "organic_pesticides": [
            {"name": "Bordeaux Mixture (1%)", "dosage": "10 g/L", "phi_days": 7, "action": "Classic copper sulfate + lime protective wash"},
        ],
        "fertilizer_recovery": "Do not apply late-season nitrogen. Apply Potassium Phosphite (3 ml/L) as a systemic defense elicitor.",
        "ipm_recovery_practices": "Destroy cull piles. Harvest tubers only after vines have been dead for 2 weeks to prevent tuber infection.",
    },
    "Squash___Powdery_mildew": {
        "disease_name": "Squash Powdery Mildew (Podosphaera xanthii)",
        "symptoms": "White, talcum powder-like fungal patches on both upper and lower leaf surfaces, causing leaves to yellow and brown.",
        "chemical_pesticides": [
            {"name": "Penconazole (Topas 100 EC)", "dosage": "0.5 ml/L", "phi_days": 7, "action": "Systemic sterol demethylation inhibitor"},
            {"name": "Hexaconazole (Contaf 5 EC)", "dosage": "1.0 ml/L", "phi_days": 14, "action": "Broad-spectrum triazole fungicide"},
        ],
        "organic_pesticides": [
            {"name": "Potassium Bicarbonate (MilStop)", "dosage": "3.0 g/L", "phi_days": 0, "action": "Cell wall dehydrator and spore killer"},
            {"name": "Horticultural Mineral Oil", "dosage": "5.0 ml/L", "phi_days": 1, "action": "Smothers fungal mycelium and prevents spore germination"},
        ],
        "fertilizer_recovery": "Apply Silicon fertilizer (Potassium Silicate @ 2 ml/L) to reinforce leaf cuticle against fungal penetration.",
        "ipm_recovery_practices": "Provide ample row spacing for airflow. Remove severely infected older leaves from bottom of vines.",
    },
    "Tomato___Early_blight": {
        "disease_name": "Tomato Early Blight (Alternaria solani)",
        "symptoms": "Dark brown target-spot concentric rings on lower leaves, causing progressive yellowing and defoliation from ground up.",
        "chemical_pesticides": [
            {"name": "Mancozeb (Dithane M-45)", "dosage": "2.5 g/L", "phi_days": 7, "action": "Contact multi-site protectant"},
            {"name": "Difenoconazole (Score 25 EC)", "dosage": "0.5 ml/L", "phi_days": 7, "action": "Systemic curative triazole"},
        ],
        "organic_pesticides": [
            {"name": "Cold-pressed Neem Oil (10000 ppm)", "dosage": "4.0 ml/L", "phi_days": 0, "action": "Bio-fungicide and insect repellent"},
            {"name": "Trichoderma harzianum", "dosage": "5.0 g/L", "phi_days": 0, "action": "Beneficial microbial antagonist"},
        ],
        "fertilizer_recovery": "Apply Calcium Nitrate (3 g/L) foliar spray to strengthen leaf cell membranes against enzymatic fungal breakdown.",
        "ipm_recovery_practices": "Mulch soil with clean straw to prevent soil-splash onto lower leaves. Prune lower 30cm of foliage.",
    },
    "Tomato___Late_blight": {
        "disease_name": "Tomato Late Blight (Phytophthora infestans)",
        "symptoms": "Greasy, dark water-soaked patches on leaves rapidly expanding under wet weather, with white mildew under leaves.",
        "chemical_pesticides": [
            {"name": "Cymoxanil + Mancozeb (Curzate M8)", "dosage": "2.0 g/L", "phi_days": 7, "action": "Penetrating curative and protective mix"},
            {"name": "Famoxadone + Cymoxanil (Equation Pro)", "dosage": "0.8 g/L", "phi_days": 3, "action": "High-potency sporulation inhibitor"},
        ],
        "organic_pesticides": [
            {"name": "Copper Hydroxide (Kocide 3000)", "dosage": "2.0 g/L", "phi_days": 1, "action": "Bio-protective copper wash"},
        ],
        "fertilizer_recovery": "Stop overhead watering immediately. Spray Mono-Potassium Phosphate (0:52:34 @ 4 g/L) to induce systemic acquired resistance (SAR).",
        "ipm_recovery_practices": "Rogue out infected plants immediately and bag them to prevent airborne spore dissemination.",
    },
    "Tomato___Septoria_leaf_spot": {
        "disease_name": "Tomato Septoria Leaf Spot (Septoria lycopersici)",
        "symptoms": "Numerous small circular spots (2-3 mm) with dark brown borders and pale gray or tan centers speckled with tiny black dots (pycnidia).",
        "chemical_pesticides": [
            {"name": "Chlorothalonil (Bravo 720)", "dosage": "2.0 ml/L", "phi_days": 7, "action": "Multi-site broad-spectrum barrier"},
            {"name": "Azoxystrobin (Heritage)", "dosage": "0.5 g/L", "phi_days": 1, "action": "Preventive strobilurin"},
        ],
        "organic_pesticides": [
            {"name": "Bio-Copper Soap", "dosage": "3.0 ml/L", "phi_days": 0, "action": "Gentle non-phytotoxic organic protectant"},
        ],
        "fertilizer_recovery": "Maintain balanced micronutrient feeding (Iron, Manganese, Boron) to stimulate vigor without forcing soft succulent stems.",
        "ipm_recovery_practices": "Eliminate solanaceous weeds (such as horsenettle and nightshade) surrounding the plot.",
    },
    "Tomato___Bacterial_spot": {
        "disease_name": "Tomato Bacterial Spot (Xanthomonas spp.)",
        "symptoms": "Small greasy water-soaked circular spots becoming angular, dark brown, and scabby on leaves and fruits.",
        "chemical_pesticides": [
            {"name": "Copper Oxychloride + Kasugamycin", "dosage": "2.0 g/L", "phi_days": 7, "action": "Antibiotic-copper synergy"},
        ],
        "organic_pesticides": [
            {"name": "Bacillus amyloliquefaciens", "dosage": "3.0 ml/L", "phi_days": 0, "action": "Colonizing bio-bactericide"},
        ],
        "fertilizer_recovery": "Avoid high ammonium nitrogen. Ensure high calcium nutrition to strengthen pectins in the cell middle lamella.",
        "ipm_recovery_practices": "Use certified pathogen-free seeds. Never work in wet tomato fields.",
    },
    "Tomato___Leaf_Mold": {
        "disease_name": "Tomato Leaf Mold (Passalora fulva / Cladosporium fulvum)",
        "symptoms": "Pale green or yellowish spots on upper leaf surfaces matching olive-green to brown velvety fungal patches on lower surface.",
        "chemical_pesticides": [
            {"name": "Difenoconazole", "dosage": "0.5 ml/L", "phi_days": 7, "action": "Triazole systemic curative"},
        ],
        "organic_pesticides": [
            {"name": "Potassium Bicarbonate", "dosage": "3.0 g/L", "phi_days": 0, "action": "Spore desiccant"},
            {"name": "Sulfur Dust", "dosage": "2.0 g/L", "phi_days": 1, "action": "Protective multi-site"},
        ],
        "fertilizer_recovery": "Apply Potassium Sulfate (0:0:50 @ 4 g/L). Maintain soil moisture consistency with drip irrigation.",
        "ipm_recovery_practices": "Heavily ventilate greenhouse / polyhouse structures. Keep relative humidity below 85%.",
    },
    "Tomato___Tomato_mosaic_virus": {
        "disease_name": "Tomato Mosaic Virus (ToMV)",
        "symptoms": "Mottling of leaves with alternating light and dark green areas, leaf curling, blistering, and 'fern-leaf' distortion.",
        "chemical_pesticides": [
            {"name": "No chemical viricide exists", "dosage": "N/A", "phi_days": 0, "action": "Viruses cannot be cured chemically"},
        ],
        "organic_pesticides": [
            {"name": "Skim Milk spray (20% solution)", "dosage": "200 ml/L", "phi_days": 0, "action": "Milk proteins inactivate virus particles during handling"},
        ],
        "fertilizer_recovery": "Apply Zinc-EDTA (1 g/L) + Seaweed Extract (2 ml/L) to support plant immune vitality and metabolic resilience.",
        "ipm_recovery_practices": "Wash hands with soap and water after handling tobacco. Disinfect all stakes and trellises with 20% trisodium phosphate (TSP).",
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "disease_name": "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "symptoms": "Severe plant stunting, upward cupping of leaf margins, prominent interveinal chlorosis (yellowing), and blossom drop.",
        "chemical_pesticides": [
            {"name": "Imidacloprid 17.8 SL (Targets vector Whiteflies)", "dosage": "0.5 ml/L", "phi_days": 14, "action": "Systemic neonicotinoid vector control"},
            {"name": "Spiromesifen (Oberon 240 SC)", "dosage": "1.0 ml/L", "phi_days": 7, "action": "Lipid biosynthesis inhibitor vector control"},
        ],
        "organic_pesticides": [
            {"name": "Yellow Sticky Traps", "dosage": "1 trap per 100 sq ft", "phi_days": 0, "action": "Mechanical vector capture"},
            {"name": "Beauveria bassiana (Entomopathogenic fungus)", "dosage": "5.0 g/L", "phi_days": 0, "action": "Bio-control of whiteflies"},
            {"name": "Neem Oil 10,000 ppm", "dosage": "4.0 ml/L", "phi_days": 0, "action": "Whitefly oviposition deterrent"},
        ],
        "fertilizer_recovery": "Apply balanced foliar feed with Magnesium Sulfate (2 g/L) and Potassium Nitrate (3 g/L) to support remaining foliage.",
        "ipm_recovery_practices": "Install 50-mesh insect-proof netting in greenhouses. Destroy infected plants immediately upon early symptom detection.",
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "disease_name": "Two-Spotted Spider Mite (Tetranychus urticae)",
        "symptoms": "Fine white/yellow stippling on leaf surface, leading to bronzing, drying, and characteristic fine silken webbing under leaves.",
        "chemical_pesticides": [
            {"name": "Abamectin (Vertimec 1.8 EC)", "dosage": "0.5 ml/L", "phi_days": 7, "action": "Translaminar avermectin miticide"},
            {"name": "Fenazaquin (Magister 10 EC)", "dosage": "1.5 ml/L", "phi_days": 14, "action": "Mitochondrial electron transport inhibitor"},
        ],
        "organic_pesticides": [
            {"name": "Insecticidal Soap (Potassium salts of fatty acids)", "dosage": "10 ml/L", "phi_days": 0, "action": "Dissolves mite cuticles on contact"},
            {"name": "Predatory Mites (Phytoseiulus persimilis)", "dosage": "2-5 mites per plant", "phi_days": 0, "action": "Biological predation"},
        ],
        "fertilizer_recovery": "Avoid water stress. Dusty, dry conditions with high nitrogen encourage explosive spider mite reproduction.",
        "ipm_recovery_practices": "Wash foliage with strong water jets to disrupt webs and dislodge mite colonies.",
    },
    "Grape___Black_rot": {
        "disease_name": "Grape Black Rot (Guignardia bidwellii)",
        "symptoms": "Reddish-brown circular spots on leaves with tiny black fruiting bodies (pycnidia) arranged in rings inside lesions.",
        "chemical_pesticides": [
            {"name": "Mancozeb (75 WP)", "dosage": "2.5 g/L", "phi_days": 28, "action": "Early season protectant"},
            {"name": "Tebuconazole (Folicur)", "dosage": "0.75 ml/L", "phi_days": 14, "action": "Curative triazole"},
        ],
        "organic_pesticides": [
            {"name": "Copper Sulfate + Lime (Bordeaux Mixture)", "dosage": "10 g/L", "phi_days": 14, "action": "Traditional protective wash"},
        ],
        "fertilizer_recovery": "Foliar spray Boron (1 g/L) and Zinc (2 g/L) during cluster pre-bloom. Avoid lush overgrown canopies.",
        "ipm_recovery_practices": "Remove and burn mummified fruit clusters hanging on vines or fallen on soil.",
    },
}


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def sanitize_filename(name: str) -> str:
    """Removes or replaces characters forbidden in Windows filenames."""
    for ch in ['?', '*', ':', '"', '<', '>', '|', '\n', '\r', '\t']:
        name = name.replace(ch, '_')
    return name.strip()


def step_1_download_and_extract() -> Tuple[Path, Path]:
    """Downloads PlantDoc master archive and extracts train/test splits with Windows path sanitization."""
    PLANTDOC_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = PLANTDOC_DIR / "plantdoc_archive.zip"
    extracted_root = PLANTDOC_DIR / "raw"

    train_dir = extracted_root / "train"
    test_dir = extracted_root / "test"

    if train_dir.exists() and test_dir.exists():
        train_count = sum(len(files) for _, _, files in os.walk(train_dir))
        test_count = sum(len(files) for _, _, files in os.walk(test_dir))
        if train_count > 2000 and test_count > 200:
            logger.info("PlantDoc dataset already extracted (%d train, %d test images). Skipping download.", train_count, test_count)
            return train_dir, test_dir

    if not (zip_path.exists() and zip_path.stat().st_size > 900 * 1024 * 1024):
        logger.info("Downloading PlantDoc Dataset archive from %s...", PLANTDOC_ZIP_URL)
        req = urllib.request.Request(PLANTDOC_ZIP_URL, headers={"User-Agent": "Mozilla/5.0"})

        t0 = time.time()
        with urllib.request.urlopen(req) as resp, open(zip_path, "wb") as out_file:
            total_size = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 4 * 1024 * 1024  # 4MB chunks
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    speed = (downloaded / (1024 * 1024)) / max(1, time.time() - t0)
                    if downloaded % (20 * 1024 * 1024) < chunk_size:
                        logger.info("Download progress: %.1f MB / %.1f MB (%.1f%%) - Speed: %.2f MB/s", downloaded / (1024*1024), total_size / (1024*1024), pct, speed)
        logger.info("Download finished in %.1f seconds.", time.time() - t0)
    else:
        logger.info("Found existing downloaded archive: %s (%.1f MB). Skipping re-download.", zip_path, zip_path.stat().st_size / (1024*1024))

    logger.info("Extracting train and test folders with Windows-safe filenames...")
    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.namelist():
            parts = member.split("/")
            if len(parts) > 2 and parts[1] in ("train", "test"):
                # Clean filename components for Windows safety
                clean_parts = [parts[1], parts[2]] + [sanitize_filename(p) for p in parts[3:]]
                dest = extracted_root / Path(*clean_parts)
                if member.endswith("/"):
                    dest.mkdir(parents=True, exist_ok=True)
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        with z.open(member) as src, open(dest, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                    except Exception as err:
                        logger.warning("Could not extract %s: %s", member, err)

    if zip_path.exists():
        try:
            zip_path.unlink()
            logger.info("Removed temporary zip archive to conserve disk space.")
        except Exception as e:
            logger.warning("Could not delete zip archive: %s", e)

    logger.info("PlantDoc extraction complete.")
    return train_dir, test_dir


def step_2_build_manifest(train_dir: Path, test_dir: Path) -> Dict[str, Any]:
    """Indexes PlantDoc images, maps to project classes, and creates stratified train/val/test splits."""
    logger.info("Step 2: Indexing and mapping PlantDoc dataset to 38 project classes...")
    with open(BASE_DIR / "models" / "class_names.json", "r", encoding="utf-8") as f:
        project_classes = json.load(f)
    class_to_idx = {c: i for i, c in enumerate(project_classes)}

    records = []
    skipped = 0
    rng = np.random.RandomState(RANDOM_SEED)

    # 1. Process Train folder
    for plantdoc_cls, mapped_cls in PLANTDOC_CLASS_MAPPING.items():
        src_folder = train_dir / plantdoc_cls
        if not src_folder.exists():
            continue
        files = [f for f in src_folder.iterdir() if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        rng.shuffle(files)

        # Split 85% train, 15% validation from the train directory
        n_val = max(1, int(len(files) * 0.15))
        for i, file_p in enumerate(files):
            split = "val" if i < n_val else "train"
            records.append({
                "filepath": str(file_p.resolve()),
                "filename": file_p.name,
                "plantdoc_class": plantdoc_cls,
                "class_name": mapped_cls,
                "class_index": class_to_idx[mapped_cls],
                "split": split,
                "source": "PlantDoc-RealWorld",
            })

    # 2. Process Official Test folder
    for plantdoc_cls, mapped_cls in PLANTDOC_CLASS_MAPPING.items():
        src_folder = test_dir / plantdoc_cls
        if not src_folder.exists():
            continue
        files = [f for f in src_folder.iterdir() if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        for file_p in files:
            records.append({
                "filepath": str(file_p.resolve()),
                "filename": file_p.name,
                "plantdoc_class": plantdoc_cls,
                "class_name": mapped_cls,
                "class_index": class_to_idx[mapped_cls],
                "split": "test",
                "source": "PlantDoc-RealWorld",
            })

    split_counts = {
        "train": sum(1 for r in records if r["split"] == "train"),
        "val": sum(1 for r in records if r["split"] == "val"),
        "test": sum(1 for r in records if r["split"] == "test"),
    }
    logger.info("Indexed %d PlantDoc images: Train=%d, Val=%d, Test=%d across %d classes.",
                len(records), split_counts["train"], split_counts["val"], split_counts["test"], len(PLANTDOC_CLASS_MAPPING))

    manifest_data = {
        "dataset_name": "PlantDoc-Dataset-RealWorld",
        "total_images": len(records),
        "split_counts": split_counts,
        "classes_mapped": len(PLANTDOC_CLASS_MAPPING),
        "class_indices": class_to_idx,
        "classes": project_classes,
        "records": records,
    }

    manifest_path = PLANTDOC_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    logger.info("Saved PlantDoc manifest to: %s", manifest_path)

    return manifest_data


def step_3_train_model(manifest_data: Dict[str, Any]) -> Dict[str, Any]:
    """Trains a MobileNetV2 candidate model on PlantDoc real-world images."""
    import tensorflow as tf
    from models.disease_classifier import build_disease_classifier, unfreeze_for_fine_tuning
    from services.image_preprocessing import get_augmentation_pipeline, load_and_preprocess_image

    logger.info("Step 3: Training MobileNetV2 candidate model on real-world PlantDoc data...")
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    records = manifest_data["records"]
    train_recs = [r for r in records if r["split"] == "train"]
    val_recs = [r for r in records if r["split"] == "val"]
    num_classes = len(manifest_data["classes"])

    batch_size = 32
    image_size = (224, 224)

    def _build_ds(recs, is_train=False):
        paths = [r["filepath"] for r in recs]
        labels = [int(r["class_index"]) for r in recs]
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if is_train:
            ds = ds.shuffle(buffer_size=len(recs), seed=RANDOM_SEED)

        def _loader(path_t, label_t):
            def _py_read(p):
                return load_and_preprocess_image(p.decode("utf-8"), target_size=image_size)
            img = tf.numpy_function(_py_read, [path_t], tf.float32)
            img.set_shape([224, 224, 3])
            return img, label_t

        ds = ds.map(_loader, num_parallel_calls=tf.data.AUTOTUNE)
        if is_train:
            aug = get_augmentation_pipeline()
            ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds = _build_ds(train_recs, is_train=True)
    val_ds = _build_ds(val_recs, is_train=False)

    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    candidate_model_path = CANDIDATES_DIR / "plantdoc_candidate.keras"

    # Transfer Learning from MobileNetV2 ImageNet
    model = build_disease_classifier(
        num_classes=num_classes,
        input_shape=(224, 224, 3),
        weights="imagenet",
        dropout_rate=0.3,
        initial_learning_rate=1e-3,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(candidate_model_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    t0 = time.time()
    logger.info("--- Stage 1: Training Classification Head on PlantDoc (5 Epochs) ---")
    h1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5,
        callbacks=callbacks,
        verbose=1,
    )

    logger.info("--- Stage 2: Fine-Tuning Top 30 Layers on PlantDoc (3 Epochs) ---")
    model = unfreeze_for_fine_tuning(model, num_layers_to_unfreeze=30, fine_tune_learning_rate=1e-5)
    h2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=3,
        callbacks=callbacks,
        verbose=1,
    )
    training_duration = time.time() - t0

    model.save(str(candidate_model_path))
    logger.info("Saved PlantDoc candidate model to: %s", candidate_model_path)

    combined_history = {}
    for k in h1.history.keys():
        combined_history[k] = h1.history[k] + h2.history.get(k, [])

    history_csv_path = REPORTS_DIR / "plantdoc_candidate_history.csv"
    with open(history_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + list(combined_history.keys()))
        for ep in range(len(combined_history["loss"])):
            writer.writerow([ep + 1] + [combined_history[k][ep] for k in combined_history.keys()])

    metadata = {
        "candidate_model_path": str(candidate_model_path.resolve()),
        "architecture": "MobileNetV2",
        "num_classes": num_classes,
        "input_shape": [224, 224, 3],
        "preprocessing": "mobilenet_v2.preprocess_input",
        "training_time_seconds": round(training_duration, 2),
        "sha256": compute_sha256(candidate_model_path),
        "dataset_source": "PlantDoc (Real-World In-The-Wild Imagery)",
        "history_summary": {
            "final_train_accuracy": float(combined_history["accuracy"][-1]),
            "final_val_accuracy": float(combined_history["val_accuracy"][-1]),
            "final_train_loss": float(combined_history["loss"][-1]),
            "final_val_loss": float(combined_history["val_loss"][-1]),
        },
    }
    with open(CANDIDATES_DIR / "plantdoc_candidate_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def step_4_evaluate(manifest_data: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluates the PlantDoc candidate vs. production models on the official held-out test split."""
    import tensorflow as tf

    logger.info("Step 4: Evaluating models on PlantDoc held-out test set...")
    test_recs = [r for r in manifest_data["records"] if r["split"] == "test"]
    class_names = manifest_data["classes"]
    candidate_path = CANDIDATES_DIR / "plantdoc_candidate.keras"
    production_path = BASE_DIR / "models" / "best_model.keras"

    logger.info("Testing on %d real-world test images across 28 represented classes.", len(test_recs))

    test_x, test_y = [], []
    for r in test_recs:
        try:
            img = load_and_preprocess_image(r["filepath"], target_size=(224, 224))
            test_x.append(img)
            test_y.append(r["class_index"])
        except Exception as e:
            logger.warning("Could not read test image %s: %s", r["filepath"], e)

    test_x = np.array(test_x, dtype=np.float32)
    test_y = np.array(test_y, dtype=np.int32)

    # 1. Candidate Model
    cand_model = tf.keras.models.load_model(str(candidate_path), compile=False)
    cand_preds = cand_model.predict(test_x, batch_size=32, verbose=1)
    cand_pred_labels = np.argmax(cand_preds, axis=1)

    cand_acc = float(np.mean(cand_pred_labels == test_y))
    cand_report = classification_report(
        test_y,
        cand_pred_labels,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cand_cm = confusion_matrix(test_y, cand_pred_labels, labels=list(range(len(class_names))))

    # 2. Production Model
    prod_model = tf.keras.models.load_model(str(production_path), compile=False)
    prod_preds = prod_model.predict(test_x, batch_size=32, verbose=1)
    prod_pred_labels = np.argmax(prod_preds, axis=1)

    prod_acc = float(np.mean(prod_pred_labels == test_y))
    prod_report = classification_report(
        test_y,
        prod_pred_labels,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    # Plot Confusion Matrix
    plt.figure(figsize=(18, 15))
    plt.imshow(cand_cm, interpolation="nearest", cmap="Purples")
    plt.title(f"PlantDoc Candidate Model Confusion Matrix (Real-World Test Acc: {cand_acc*100:.2f}%)", fontsize=14, fontweight="bold")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xticks(np.arange(len(class_names)), class_names, rotation=90, fontsize=8)
    plt.yticks(np.arange(len(class_names)), class_names, fontsize=8)
    plt.xlabel("Predicted Label", fontweight="bold")
    plt.ylabel("True Label", fontweight="bold")
    plt.tight_layout()
    cm_path = REPORTS_DIR / "plantdoc_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()

    eval_results = {
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "test_dataset": "PlantDoc-RealWorld (Field Conditions)",
        "test_samples_total": len(test_y),
        "evaluated_classes_count": len(class_names),
        "classes_in_plantdoc": len(PLANTDOC_CLASS_MAPPING),
        "candidate_model": {
            "path": str(candidate_path.resolve()),
            "accuracy_percent": round(cand_acc * 100, 2),
            "macro_f1": round(cand_report["macro avg"]["f1-score"], 4),
            "weighted_f1": round(cand_report["weighted avg"]["f1-score"], 4),
            "correct_samples": int(np.sum(cand_pred_labels == test_y)),
        },
        "production_model": {
            "path": str(production_path.resolve()),
            "accuracy_percent": round(prod_acc * 100, 2),
            "macro_f1": round(prod_report["macro avg"]["f1-score"], 4),
            "weighted_f1": round(prod_report["weighted avg"]["f1-score"], 4),
            "correct_samples": int(np.sum(prod_pred_labels == test_y)),
        },
        "per_class_candidate_metrics": {
            cls: {
                "precision": round(cand_report[cls]["precision"], 4),
                "recall": round(cand_report[cls]["recall"], 4),
                "f1_score": round(cand_report[cls]["f1-score"], 4),
                "support": cand_report[cls]["support"],
            }
            for cls in class_names
        },
    }

    eval_json_path = REPORTS_DIR / "plantdoc_eval.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)

    # Save Agronomic Recovery Guide JSON
    guide_path = REPORTS_DIR / "plantdoc_agronomic_recovery_guide.json"
    with open(guide_path, "w", encoding="utf-8") as f:
        json.dump(PLANTDOC_AGRONOMIC_GUIDE, f, indent=2)

    logger.info("Saved evaluation JSON to: %s", eval_json_path)
    logger.info("Saved Agronomic Recovery Guide to: %s", guide_path)
    return eval_results


def main():
    print("=" * 75)
    print("PlantDoc Dataset Ingestion, Candidate Training & Agronomic Recovery")
    print("=" * 75)

    prod_model_path = BASE_DIR / "models" / "best_model.keras"
    initial_prod_hash = compute_sha256(prod_model_path)
    print(f"Verified Production Model Initial SHA-256: {initial_prod_hash}")

    # Step 1: Download & Extract
    train_dir, test_dir = step_1_download_and_extract()

    # Step 2: Index & Split
    manifest_data = step_2_build_manifest(train_dir, test_dir)

    # Step 3: Train Candidate Model
    train_metadata = step_3_train_model(manifest_data)

    # Step 4: Evaluate
    eval_results = step_4_evaluate(manifest_data)

    # Step 5: Verify Production Model Safety
    final_prod_hash = compute_sha256(prod_model_path)
    if initial_prod_hash != final_prod_hash:
        logger.critical("SAFETY VIOLATION: models/best_model.keras was modified during PlantDoc run!")
        sys.exit(1)
    else:
        print(f"\n[SAFETY VERIFIED] models/best_model.keras remained 100% UNTOUCHED.")

    print("\n" + "=" * 75)
    print("PLANTDOC PIPELINE COMPLETE")
    print(f"Total Images:              {manifest_data['total_images']}")
    print(f"PlantDoc Classes Mapped:   {manifest_data['classes_mapped']} / 28")
    print(f"Candidate Model Path:      {train_metadata['candidate_model_path']}")
    print(f"Candidate Test Acc:        {eval_results['candidate_model']['accuracy_percent']}% (Macro F1: {eval_results['candidate_model']['macro_f1']})")
    print(f"Production Test Acc:       {eval_results['production_model']['accuracy_percent']}% (Macro F1: {eval_results['production_model']['macro_f1']})")
    print("=" * 75)


if __name__ == "__main__":
    main()
