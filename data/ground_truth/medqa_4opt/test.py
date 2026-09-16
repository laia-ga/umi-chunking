import json

ALLERGY_TERMS = [
    # Alergia, hipersensibilidad y reacciones alérgicas generales
    "allergy", "allergic reaction", "allergic reactions", "hypersensitivity", "hypersensitivity reaction", "hypersensitivity reactions", "immediate hypersensitivity", "immediate hypersensitivity reaction", "delayed hypersensitivity", "allergic disease", "allergic disorder", "atopy", "atopic", "allergen", "allergens", "allergen exposure", "allergic symptoms", "systemic allergic reaction", "systemic reaction", "systemic reactions", "severe allergic reaction",

    # Anafilaxia
    "anaphylaxis", "anaphylactic reaction", "anaphylactic reactions", "anaphylactic shock", "anaphylactoid reaction", "severe anaphylaxis", "refractory anaphylaxis", "biphasic anaphylaxis", "biphasic reaction", "biphasic reactions", "systemic anaphylaxis", "acute anaphylaxis", "anaphylactic symptoms", "anaphylactic episode", "anaphylactic episodes", "history of anaphylaxis", "risk of anaphylaxis", "fatal anaphylaxis", "food-induced anaphylaxis", "drug-induced anaphylaxis", "insect sting anaphylaxis", "venom-induced anaphylaxis",

    # Manifestaciones típicas de anafilaxia/alergia
    "urticaria", "generalized urticaria", "generalised urticaria", "acute urticaria", "hives", "angioedema", "facial angioedema", "laryngeal edema", "laryngeal oedema", "tongue swelling", "lip swelling", "facial swelling", "throat swelling", "throat tightness", "wheezing and urticaria", "urticaria and angioedema", "difficulty breathing after", "shortness of breath after", "hypotension after", "collapse after", "syncope after", "allergic shock",

    # Adrenalina / epinefrina y tratamiento de anafilaxia
    "epinephrine", "adrenaline", "intramuscular epinephrine", "intramuscular adrenaline", "IM epinephrine", "IM adrenaline", "epinephrine injection", "adrenaline injection", "epinephrine autoinjector", "epinephrine auto-injector", "adrenaline autoinjector", "adrenaline auto-injector", "epinephrine pen", "adrenaline pen", "injectable epinephrine", "injectable adrenaline", "epinephrine administration", "adrenaline administration", "epinephrine dose", "adrenaline dose", "epinephrine treatment", "adrenaline treatment", "first-line treatment of anaphylaxis", "treatment of anaphylaxis", "management of anaphylaxis", "emergency treatment of anaphylaxis",

    # Autoinyectores/Jext y uso de adrenalina
    "Jext", "Jext 300", "300 micrograms adrenaline", "300 micrograms epinephrine", "pre-filled pen", "prefilled pen", "autoinjector", "auto-injector", "adrenaline autoinjector", "epinephrine autoinjector", "self-injectable epinephrine", "self-injectable adrenaline", "emergency epinephrine", "emergency adrenaline", "second dose epinephrine", "second dose adrenaline", "repeat epinephrine", "repeat adrenaline", "epinephrine device", "adrenaline device",

    # Adrenalina en colegios
    "stock epinephrine", "stock adrenaline", "school epinephrine", "school adrenaline", "epinephrine in schools", "adrenaline in schools", "school anaphylaxis", "anaphylaxis at school", "school nurse", "epinephrine access", "adrenaline access", "emergency medication at school", "epinephrine availability", "adrenaline availability",

    # Alergia alimentaria
    "food allergy", "food allergies", "food allergic", "food hypersensitivity", "food-induced allergic reaction", "food-induced anaphylaxis", "food allergen", "food allergens", "IgE-mediated food allergy", "IgE mediated food allergy", "oral food challenge", "food challenge", "allergen avoidance", "food avoidance", "elimination diet", "allergen reintroduction", "food reintroduction",

    # Alergia al huevo
    "egg allergy", "egg allergic", "egg hypersensitivity", "hen egg allergy", "hen's egg allergy", "hens egg allergy", "egg protein allergy", "egg white allergy", "egg white", "egg yolk", "ovalbumin", "ovomucoid", "Gal d 1", "Gal d 2", "baked egg", "cooked egg", "raw egg", "egg avoidance", "egg reintroduction", "egg challenge", "oral egg challenge", "egg-specific IgE", "egg specific IgE", "egg sensitization", "egg sensitisation",

    # Diagnóstico de alergia alimentaria
    "skin prick test", "skin prick testing", "allergy skin test", "allergy testing", "specific IgE", "specific immunoglobulin E", "serum specific IgE", "allergen-specific IgE", "allergen specific IgE", "IgE antibodies", "IgE sensitization", "IgE sensitisation", "oral challenge test", "oral food challenge", "food challenge test",

    # Huevo y vacunas
    "egg allergy vaccine", "egg allergic vaccine", "influenza vaccine egg allergy", "influenza vaccination egg allergy", "MMR vaccine egg allergy", "yellow fever vaccine egg allergy", "egg-containing vaccine", "egg-containing vaccines",

    # Rinitis alérgica
    "allergic rhinitis", "seasonal allergic rhinitis", "perennial allergic rhinitis", "seasonal allergies", "perennial allergies", "hay fever", "rhinoconjunctivitis", "allergic rhinoconjunctivitis", "nasal allergy", "nasal allergies", "nasal allergic symptoms", "nasal hypersensitivity", "pollen allergy", "pollen allergies", "pollen-induced rhinitis", "house dust mite allergy", "dust mite allergy",

    # Síntomas característicos de rinitis
    "nasal congestion", "nasal obstruction", "nasal itching", "itchy nose", "nasal pruritus", "rhinorrhea", "rhinorrhoea", "watery rhinorrhea", "watery rhinorrhoea", "sneezing", "sneezing attacks", "allergic nasal symptoms", "seasonal nasal symptoms",

    # Tratamiento intranasal/rinitis
    "intranasal corticosteroid", "intranasal corticosteroids", "nasal corticosteroid", "nasal corticosteroids", "intranasal steroid", "intranasal steroids", "nasal steroid", "nasal steroids", "nasal spray", "intranasal treatment", "intranasal therapy", "intranasal antihistamine", "intranasal antihistamines", "oral antihistamine", "oral antihistamines", "allergic rhinitis treatment", "allergic rhinitis management",

    # Mometasona / Nasonex
    "mometasone", "mometasone furoate", "mometasone nasal spray", "Nasonex", "nasal mometasone", "intranasal mometasone", "corticosteroid nasal spray",

    # Pólipos nasales, también cubiertos por Nasonex
    "nasal polyps", "nasal polyp", "nasal polyposis", "polyposis nasi", "nasal polypectomy",

    # Antihistamínicos / bilastina / Bilaxten
    "antihistamine", "antihistamines", "H1 antihistamine", "H1 antihistamines", "H1 receptor antagonist", "histamine H1 antagonist", "second-generation antihistamine", "second generation antihistamine", "nonsedating antihistamine", "non-sedating antihistamine", "bilastine", "Bilaxten", "oral antihistamine therapy", "antihistamine treatment", "antihistamine therapy", "treatment of urticaria", "chronic urticaria", "chronic spontaneous urticaria", "allergic rhinoconjunctivitis",

    # Asma y montelukast
    "montelukast", "Montelukast Cinfa", "leukotriene receptor antagonist", "leukotriene receptor antagonists", "cysteinyl leukotriene receptor antagonist", "CysLT1 receptor", "leukotriene modifier", "asthma prophylaxis", "asthma prevention", "exercise-induced bronchoconstriction", "exercise induced bronchoconstriction", "exercise-induced asthma", "exercise induced asthma", "bronchoconstriction induced by exercise", "aspirin-sensitive asthma", "aspirin sensitive asthma", "asthma with allergic rhinitis", "allergic asthma", "persistent asthma", "asthma maintenance treatment", "asthma controller",

    # Alergia ocular / conjuntivitis alérgica
    "allergic conjunctivitis", "ocular allergy", "ocular allergies", "ocular allergic disease", "ocular hypersensitivity", "allergic eye disease", "seasonal allergic conjunctivitis", "perennial allergic conjunctivitis", "SAC", "PAC", "vernal keratoconjunctivitis", "VKC", "atopic keratoconjunctivitis", "AKC", "giant papillary conjunctivitis", "contact blepharoconjunctivitis", "allergic rhinoconjunctivitis",

    # Síntomas oculares alérgicos
    "ocular itching", "eye itching", "itchy eyes", "ocular pruritus", "conjunctival itching", "conjunctival hyperemia", "conjunctival hyperaemia", "conjunctival redness", "eye redness", "red eyes", "watery eyes", "tearing", "lacrimation", "ocular irritation", "conjunctival edema", "conjunctival oedema", "chemosis", "eyelid swelling", "papillary conjunctivitis",

    # Tratamiento de conjuntivitis alérgica
    "ophthalmic antihistamine", "ophthalmic antihistamines", "topical antihistamine", "topical ocular antihistamine", "antihistamine eye drops", "antiallergic eye drops", "anti-allergic eye drops", "ophthalmic solution", "ophthalmic solutions", "mast cell stabilizer", "mast cell stabilizers", "mast cell stabiliser", "mast cell stabilisers", "dual-action antihistamine", "dual action antihistamine", "topical ocular treatment", "topical ocular therapy",

    # Olopatadina
    "olopatadine", "olopatadine hydrochloride", "olopatadine eye drops", "olopatadine ophthalmic", "ophthalmic olopatadine", "Olopatadina", "Olopatadina Abamed",

    # Medios de contraste
    "contrast media", "contrast medium", "contrast agent", "contrast agents", "radiographic contrast media", "radiocontrast media", "iodinated contrast", "iodinated contrast media", "iodinated contrast medium", "iodinated contrast agent", "iodine contrast", "radiocontrast agent", "gadolinium", "gadolinium contrast", "gadolinium-based contrast agent", "gadolinium-based contrast agents", "GBCA",

    # Hipersensibilidad a contraste
    "contrast media hypersensitivity", "contrast medium hypersensitivity", "contrast hypersensitivity", "contrast allergy", "contrast media allergy", "iodinated contrast allergy", "iodinated contrast hypersensitivity", "gadolinium hypersensitivity", "gadolinium allergy", "reaction to contrast", "contrast reaction", "contrast reactions", "immediate reaction to contrast", "delayed reaction to contrast", "previous contrast reaction", "history of contrast reaction",

    # Prevención/diagnóstico de reacciones a contraste
    "contrast premedication", "premedication for contrast", "contrast pretreatment", "corticosteroid premedication", "antihistamine premedication", "diphenhydramine premedication", "contrast skin testing", "contrast allergy testing", "alternative contrast agent",

    # Himenópteros
    "Hymenoptera", "Hymenoptera allergy", "Hymenoptera venom", "Hymenoptera venom allergy", "venom allergy", "insect venom allergy", "insect sting allergy", "sting allergy", "allergic reaction to insect sting", "anaphylaxis after insect sting", "venom-induced anaphylaxis",

    # Abejas, avispas, etc.
    "bee sting", "bee stings", "honeybee sting", "honeybee venom", "bee venom", "wasp sting", "wasp stings", "wasp venom", "yellow jacket sting", "yellow jacket venom", "hornet sting", "hornet venom", "vespid venom", "insect sting", "insect stings", "stung by a bee", "stung by a wasp",

    # Reacciones a picadura
    "large local reaction", "large local reactions", "local reaction to sting", "systemic reaction to sting", "systemic sting reaction", "systemic reaction after sting", "generalized reaction after sting", "generalised reaction after sting",

    # Diagnóstico de alergia a veneno
    "venom-specific IgE", "venom specific IgE", "bee venom specific IgE", "wasp venom specific IgE", "venom skin test", "venom skin testing", "intradermal venom test", "venom allergy diagnosis", "venom allergy testing", "component-resolved diagnosis", "component-resolved diagnostics", "component resolved diagnosis", "molecular allergy diagnosis", "molecular allergy diagnostics", "cross-reactive carbohydrate determinants", "cross-reactive carbohydrate determinant", "CCD", "double sensitization", "double sensitisation", "double positivity",

    # Alérgenos de veneno relativamente específicos
    "Api m 1", "Api m 2", "Api m 3", "Api m 5", "Api m 10", "Ves v 1", "Ves v 5", "phospholipase A2", "hyaluronidase", "antigen 5",

    # Inmunoterapia con veneno
    "venom immunotherapy", "venom-specific immunotherapy", "venom specific immunotherapy", "VIT", "bee venom immunotherapy", "wasp venom immunotherapy", "insect venom immunotherapy",

    # Inmunología/alergología clínica y manejo
    "allergy immunology", "allergy and immunology", "allergy specialist", "allergist", "allergy clinic", "allergy management", "allergy diagnosis", "allergy treatment", "allergy care", "allergy practice", "immunotherapy", "allergen immunotherapy"
]

## CONTAR METAMAP_PHRASES UNICAS EN UN DOCUMENTO
def get_unique_metamap_phrases(file_path):
    """
    Devuelve todas las frases de 'metamap_phrases'
    del archivo JSONL sin duplicados.
    """

    unique_phrases = set()

    with open(file_path, "r", encoding="utf-8") as file:

        for line in file:
            data = json.loads(line)

            phrases = data.get("metamap_phrases", [])

            unique_phrases.update(phrases)

    return list(unique_phrases)

phrases = get_unique_metamap_phrases(
    "data/medqa_4opt/phrases_no_exclude_test.jsonl"
)

# print(phrases)
# print(f"Número de frases únicas: {len(phrases)}")

# FILTRAR LAS PREGUNTAS DE UN DOCUMENTO POR METAMAP_PHRASES ESPECIFICAS
def filter_by_metamap_phrases(file_path, concepts):
    """
    Devuelve las preguntas cuyo 'metamap_phrases' contiene
    alguno de los conceptos buscados.

    La búsqueda:
    - ignora mayúsculas/minúsculas
    - permite coincidencias parciales
    """

    concepts_lower = [
        concept.lower()
        for concept in concepts
    ]

    filtered_lines = []

    with open(file_path, "r", encoding="utf-8") as file:

        for line in file:

            data = json.loads(line)

            metamap_phrases = data.get(
                "metamap_phrases",
                []
            )

            # Comprobar si algún concepto aparece
            # dentro de alguna frase de MetaMap
            found = any(
                concept in phrase.lower()
                for phrase in metamap_phrases
                for concept in concepts_lower
            )

            if found:
                filtered_lines.append(data)

    return filtered_lines

filtered_questions = filter_by_metamap_phrases(
    "data/medqa_4opt/phrases_no_exclude_test.jsonl",
    ALLERGY_TERMS
)

print(
    f"Preguntas encontradas: "
    f"{len(filtered_questions)}"
)