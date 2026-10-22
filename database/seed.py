"""
Database Seeding Script
Populates the database with demonstration records for plant disease recommendations
and fertilizer guidelines.

Usage:
    python -m database.seed
    or
    python database/seed.py
"""

import sys
from pathlib import Path

# Add project root to path if script is run directly
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from database.db import check_db_connection, get_db_cursor
from config import Config

# ==============================================================================
# Sample Demonstration Data
# Educational disclaimer: These entries represent sample agronomic recommendations
# for demonstration and testing purposes. They do not substitute for certified
# local agricultural extension diagnostics or regional chemical safety guidelines.
# ==============================================================================

SAMPLE_DISEASE_RECOMMENDATIONS = [
    {
        "disease_code": "TOMATO_EARLY_BLIGHT",
        "crop_name": "Tomato",
        "disease_name": "Early Blight (Alternaria solani)",
        "symptoms": (
            "Concentric dark brown to black target-like spots appearing initially on lower, "
            "older leaves. Yellow halo surrounds necrotic spots, progressing to defoliation."
        ),
        "treatment_guidance": (
            "[Educational Sample] Prune affected lower foliage to curb spread. If severe, apply "
            "registered copper-based bio-fungicides or chlorothalonil following regional extension advisory."
        ),
        "prevention_guidance": (
            "Practice 3-year crop rotation with non-solanaceous crops. Employ drip irrigation to keep foliage "
            "dry and maintain adequate plant spacing (60 cm) for air circulation."
        ),
        "source_reference": "ICAR-IIHR Plant Pathology Advisory Series (Demo Data)",
        "review_status": "educational_only",
    },
    {
        "disease_code": "TOMATO_LATE_BLIGHT",
        "crop_name": "Tomato",
        "disease_name": "Late Blight (Phytophthora infestans)",
        "symptoms": (
            "Water-soaked irregular pale green lesions on leaves turning dark brown/purplish-black. "
            "Under humid conditions, delicate white fungal-like growth develops on leaf undersides."
        ),
        "treatment_guidance": (
            "[Educational Sample] Immediately rogue out and destroy severely infected plants. Apply preventive "
            "mancozeb or systemic cymoxanil/metalaxyl formulations as approved by local agricultural departments."
        ),
        "prevention_guidance": (
            "Avoid overhead sprinkler irrigation. Choose certified blight-resistant tomato varieties. "
            "Destroy volunteer potato and tomato cull piles before planting."
        ),
        "source_reference": "FAO Integrated Pest Management Guidelines (Demo Data)",
        "review_status": "educational_only",
    },
    {
        "disease_code": "POTATO_EARLY_BLIGHT",
        "crop_name": "Potato",
        "disease_name": "Early Blight (Alternaria solani)",
        "symptoms": (
            "Small circular dark brown necrotic spots with concentric ridges (target pattern) on mature foliage. "
            "Infected leaves turn leathery, curl, and drop prematurely."
        ),
        "treatment_guidance": (
            "[Educational Sample] Apply protective contact fungicides upon first sign of lesions on lower canopy. "
            "Rotate chemical fungicide modes of action to prevent fungal resistance."
        ),
        "prevention_guidance": (
            "Maintain optimal crop vigor with balanced soil nutrition (adequate potassium and nitrogen). "
            "Harvest tubers only after vine desiccation to reduce skinning and tuber rot."
        ),
        "source_reference": "USDA-ARS Potato Disease Identification Handbook (Demo Data)",
        "review_status": "educational_only",
    },
    {
        "disease_code": "CORN_COMMON_RUST",
        "crop_name": "Corn (Maize)",
        "disease_name": "Common Rust (Puccinia sorghi)",
        "symptoms": (
            "Small, powdery cinnamon-brown pustules (uredinia) on both upper and lower leaf surfaces. "
            "Pustules rupture the epidermis, releasing powdery rust-colored spores."
        ),
        "treatment_guidance": (
            "[Educational Sample] Chemical control is generally warranted only when high rust pressure occurs "
            "on susceptible inbred lines before tasseling. Consult state agronomy extension for economic thresholds."
        ),
        "prevention_guidance": (
            "Plant resistant or tolerant commercial hybrid varieties. Plant early in the season to allow plants "
            "to mature before peak airborne spore influx occurs."
        ),
        "source_reference": "CIMMYT Maize Agronomy & IPM Guide (Demo Data)",
        "review_status": "educational_only",
    },
    {
        "disease_code": "APPLE_APPLE_SCAB",
        "crop_name": "Apple",
        "disease_name": "Apple Scab (Venturia inaequalis)",
        "symptoms": (
            "Olive-green to brown velvety spots on leaves, becoming raised, dark, and corky. "
            "Severe infection leads to early leaf chlorosis and premature defoliation."
        ),
        "treatment_guidance": (
            "[Educational Sample] Prune orchard canopies to encourage rapid foliage drying after rainfall. "
            "Apply protective captan or mancozeb sprays at green-tip stage according to degree-day scab models."
        ),
        "prevention_guidance": (
            "Rake or shred fallen leaves in autumn and apply 5% urea to accelerate leaf decomposition, "
            "drastically reducing overwintering ascospore inoculum."
        ),
        "source_reference": "Cornell Cooperative Extension Tree Fruit IPM Guidelines (Demo Data)",
        "review_status": "educational_only",
    },
    {
        "disease_code": "PEPPER_BELL_BACTERIAL_SPOT",
        "crop_name": "Bell Pepper",
        "disease_name": "Bacterial Spot (Xanthomonas vesicatoria)",
        "symptoms": (
            "Small circular water-soaked lesions developing into brown angular spots with pale halos. "
            "Severely spotted leaves turn yellow and drop, exposing developing peppers to sunscald."
        ),
        "treatment_guidance": (
            "[Educational Sample] Apply preventative fixed-copper bactericides mixed with mancozeb to overcome "
            "copper-tolerant bacterial strains, adhering strictly to pre-harvest intervals (PHI)."
        ),
        "prevention_guidance": (
            "Use only certified hot-water treated or pathogen-free seeds and resistant cultivars (XR 1-5). "
            "Avoid field operations when canopy foliage is wet from rain or dew."
        ),
        "source_reference": "University of Florida IFAS Extension Plant Pathology (Demo Data)",
        "review_status": "educational_only",
    },
    {
        "disease_code": "GRAPE_BLACK_ROT",
        "crop_name": "Grape",
        "disease_name": "Black Rot (Guignardia bidwellii)",
        "symptoms": (
            "Circular reddish-brown necrotic spots on upper leaf surfaces bounded by tiny black pycnidia. "
            "Berries shrivel into hard, black, wrinkled mummies that remain attached to the vine."
        ),
        "treatment_guidance": (
            "[Educational Sample] Remove and burn or bury all dried fruit mummies during winter pruning. "
            "Apply protective myclobutanil or mancozeb sprays starting at early shoot growth (1-3 inches)."
        ),
        "prevention_guidance": (
            "Maintain trellis architecture to ensure sunlight penetration and rapid air drying across canopies. "
            "Mow vineyard ground cover to reduce relative humidity beneath vines."
        ),
        "source_reference": "Penn State Extension Grape Disease Management Guide (Demo Data)",
        "review_status": "educational_only",
    },
]

SAMPLE_FERTILIZER_RECOMMENDATIONS = [
    {
        "crop_name": "Apple",
        "growth_stage": "Bud Break to Petal Fall (Spring)",
        "soil_condition": "Well-drained orchard soil, pH 6.0 - 6.5",
        "recommendation_text": (
            "[Educational Sample] Apply foliar boron and moderate nitrogen to support strong bloom and cell division. "
            "Avoid excessive post-bloom nitrogen to safeguard fruit storage firmness and coloration."
        ),
        "source_reference": "WSU Tree Fruit Nutrition Guide",
        "review_status": "educational_only",
    },
    {
        "crop_name": "Bell Pepper",
        "growth_stage": "Vegetative to First Flower",
        "soil_condition": "Loamy soil, high organic matter",
        "recommendation_text": (
            "[Educational Sample] Side-dress balanced NPK (15-15-15) supplemented with magnesium sulfate (Epsom salt). "
            "Maintain steady calcium availability to prevent blossom-end rot during early fruit enlargement."
        ),
        "source_reference": "Texas A&M AgriLife Extension Vegetable Practices",
        "review_status": "educational_only",
    },
    {
        "crop_name": "Grape",
        "growth_stage": "Post-Harvest to Dormancy",
        "soil_condition": "Gravelly or rocky hillside soils",
        "recommendation_text": (
            "[Educational Sample] Apply light potassium and organic compost to restore carbohydrate reserves in roots and trunks "
            "before winter dormancy, without inducing late autumn vegetative flushes."
        ),
        "source_reference": "UC Davis Viticulture Nutrient Management",
        "review_status": "educational_only",
    },
    {
        "crop_name": "Tomato",
        "growth_stage": "Early Vegetative (Weeks 2-4)",
        "soil_condition": "Standard sandy-loam, pH 6.0 - 6.8",
        "recommendation_text": (
            "[Educational Sample] Apply balanced starter fertilizer (e.g., NPK 10-10-10 or 19-19-19) "
            "combined with well-composted organic matter to promote strong root architecture and vegetative canopy."
        ),
        "source_reference": "ICAR Package of Practices for Vegetable Crops",
        "review_status": "educational_only",
    },
    {
        "crop_name": "Tomato",
        "growth_stage": "Flowering & Fruit Set (Weeks 5-8)",
        "soil_condition": "Well-drained soil, monitor calcium levels",
        "recommendation_text": (
            "[Educational Sample] Shift toward higher potassium and phosphorus (e.g., NPK 5-10-10 or 13-0-45). "
            "Apply foliar calcium nitrate to prevent blossom-end rot and promote firm fruit development."
        ),
        "source_reference": "FAO Integrated Plant Nutrition Guidelines",
        "review_status": "educational_only",
    },
    {
        "crop_name": "Potato",
        "growth_stage": "Tuber Initiation (Weeks 4-6)",
        "soil_condition": "Slightly acidic loam (pH 5.5 - 6.2)",
        "recommendation_text": (
            "[Educational Sample] Side-dress with potassium sulfate (SOP) and moderate phosphorus. "
            "Avoid excess nitrogen after tuber initiation to prevent excessive foliage growth over tuber bulking."
        ),
        "source_reference": "USDA-ARS Potato Production Advisory",
        "review_status": "educational_only",
    },
    {
        "crop_name": "Corn (Maize)",
        "growth_stage": "Knee-High Stage (V6-V8)",
        "soil_condition": "Alluvial / Loamy soils with nitrogen demand",
        "recommendation_text": (
            "[Educational Sample] Side-dress urea or ammonium nitrate (approx. 40-50 kg N/ha equivalent) "
            "immediately prior to rapid stem elongation to maximize grain ear development."
        ),
        "source_reference": "CIMMYT Maize Production Handbook",
        "review_status": "educational_only",
    },
]


def seed_database() -> dict[str, int]:
    """
    Seeds demonstration records into disease_recommendations and fertilizer_recommendations.
    Uses parameterized SQL and idempotent updates (ON DUPLICATE KEY UPDATE) to prevent duplicates.
    Never seeds fake users or insecure default credentials.
    Returns a dictionary of counts: {'disease_upserted': X, 'fertilizer_upserted': Y}.
    """
    print("=" * 65)
    print("Plant Disease Advisory & Recommendation System - Seeder")
    print("=" * 65)

    # 1. Connectivity Check
    is_connected, msg = check_db_connection()
    if not is_connected:
        print(f"\n[ERROR] Database connection failed: {msg}")
        print("\nPlease ensure that:")
        print("1. MySQL Server is installed and running on the target host/port.")
        print(f"2. Target database '{Config.DB_NAME}' has been created via database/schema.sql.")
        print("3. Valid credentials are configured in your local .env file.")
        sys.exit(1)

    print(f"[OK] Database connection verified ({Config.DB_HOST}:{Config.DB_PORT}/{Config.DB_NAME}).")

    disease_count = 0
    fertilizer_count = 0

    # 2. Seed Disease Recommendations (Idempotent upsert by disease_code)
    disease_sql = """
        INSERT INTO disease_recommendations (
            disease_code, crop_name, disease_name, symptoms,
            treatment_guidance, prevention_guidance, source_reference, review_status
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            crop_name = VALUES(crop_name),
            disease_name = VALUES(disease_name),
            symptoms = VALUES(symptoms),
            treatment_guidance = VALUES(treatment_guidance),
            prevention_guidance = VALUES(prevention_guidance),
            source_reference = VALUES(source_reference),
            review_status = VALUES(review_status);
    """

    with get_db_cursor(commit=True) as cursor:
        for item in SAMPLE_DISEASE_RECOMMENDATIONS:
            cursor.execute(
                disease_sql,
                (
                    item["disease_code"],
                    item["crop_name"],
                    item["disease_name"],
                    item["symptoms"],
                    item["treatment_guidance"],
                    item["prevention_guidance"],
                    item["source_reference"],
                    item["review_status"],
                ),
            )
            disease_count += 1

    print(f"[SUCCESS] Upserted {disease_count} disease recommendation records.")

    # 3. Seed Fertilizer Recommendations (Idempotent by crop_name + growth_stage match)
    check_fert_sql = """
        SELECT id FROM fertilizer_recommendations
        WHERE crop_name = %s AND growth_stage = %s
        LIMIT 1;
    """
    insert_fert_sql = """
        INSERT INTO fertilizer_recommendations (
            crop_name, growth_stage, soil_condition,
            recommendation_text, source_reference, review_status
        ) VALUES (%s, %s, %s, %s, %s, %s);
    """
    update_fert_sql = """
        UPDATE fertilizer_recommendations
        SET soil_condition = %s,
            recommendation_text = %s,
            source_reference = %s,
            review_status = %s
        WHERE id = %s;
    """

    with get_db_cursor(commit=True) as cursor:
        for item in SAMPLE_FERTILIZER_RECOMMENDATIONS:
            cursor.execute(check_fert_sql, (item["crop_name"], item["growth_stage"]))
            existing = cursor.fetchone()
            if existing:
                existing_id = existing["id"] if isinstance(existing, dict) else existing[0]
                cursor.execute(
                    update_fert_sql,
                    (
                        item["soil_condition"],
                        item["recommendation_text"],
                        item["source_reference"],
                        item["review_status"],
                        existing_id,
                    ),
                )
            else:
                cursor.execute(
                    insert_fert_sql,
                    (
                        item["crop_name"],
                        item["growth_stage"],
                        item["soil_condition"],
                        item["recommendation_text"],
                        item["source_reference"],
                        item["review_status"],
                    ),
                )
            fertilizer_count += 1

    print(f"[SUCCESS] Upserted {fertilizer_count} fertilizer recommendation records.")
    print("=" * 65)
    print("Database seeding completed successfully.")
    print("Note: Educational sample data populated. No fake user accounts were created.")
    print("=" * 65)

    return {"disease_upserted": disease_count, "fertilizer_upserted": fertilizer_count}


if __name__ == "__main__":
    seed_database()
