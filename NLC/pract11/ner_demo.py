"""
Practical 11: Named Entity Recognition (NER) - Demonstration Script
Author: NAROLA SARTH DHARMESHBHAI
Roll No: 23BCE194
Subject: NATURAL LANGUAGE PROCESSING
"""

import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("NAMED ENTITY RECOGNITION (NER) - PRACTICAL DEMONSTRATION")
print("="*80)
print()

# =============================================================================
# Part 1: NER using spaCy
# =============================================================================
print("\n" + "="*80)
print("PART 1: Named Entity Recognition using spaCy")
print("="*80)

try:
    import spacy
    
    # Load model
    nlp = spacy.load("en_core_web_sm")
    
    # Sample text
    text = """
    Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976. 
    The company is headquartered in Cupertino, California. In 2023, Tim Cook serves as the CEO. 
    Apple's products include the iPhone, iPad, and MacBook. The company's market value exceeded 
    $3 trillion in January 2022. Microsoft and Google are major competitors in the tech industry.
    """
    
    # Process text
    doc = nlp(text)
    
    print("\nNamed Entities found:\n")
    print(f"{'Entity':<30} {'Label':<15} {'Description'}")
    print("-" * 70)
    
    for ent in doc.ents:
        print(f"{ent.text:<30} {ent.label_:<15} {spacy.explain(ent.label_)}")
    
    # Group by type
    print("\n\nEntities grouped by type:\n")
    entities_dict = {}
    for ent in doc.ents:
        if ent.label_ not in entities_dict:
            entities_dict[ent.label_] = []
        entities_dict[ent.label_].append(ent.text)
    
    for entity_type, entity_list in entities_dict.items():
        print(f"{entity_type}: {', '.join(set(entity_list))}")
    
    print("\n✓ spaCy NER completed successfully!")
    
except Exception as e:
    print(f"✗ Error with spaCy: {e}")
    print("  Install with: pip install spacy")
    print("  Download model: python -m spacy download en_core_web_sm")


# =============================================================================
# Part 2: NER using NLTK
# =============================================================================
print("\n" + "="*80)
print("PART 2: Named Entity Recognition using NLTK")
print("="*80)

try:
    import nltk
    from nltk import word_tokenize, pos_tag, ne_chunk
    
    # Sample text
    text = "Barack Obama was born in Hawaii. He served as the 44th President of the United States from 2009 to 2017."
    
    # Tokenize and POS tag
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    
    # Named Entity Recognition
    named_entities = ne_chunk(pos_tags)
    
    print("\nNamed Entities (NLTK):\n")
    for chunk in named_entities:
        if hasattr(chunk, 'label'):
            entity = ' '.join(c[0] for c in chunk)
            entity_type = chunk.label()
            print(f"{entity:<30} -> {entity_type}")
    
    # Test with another example
    test_text = "Microsoft was founded by Bill Gates and Paul Allen in Seattle, Washington."
    tokens = word_tokenize(test_text)
    pos_tags = pos_tag(tokens)
    chunks = ne_chunk(pos_tags)
    
    print(f"\n\nTest Text: {test_text}\n")
    print("Extracted Entities:")
    for chunk in chunks:
        if hasattr(chunk, 'label'):
            entity_text = ' '.join(c[0] for c in chunk)
            entity_type = chunk.label()
            print(f"  {entity_text} -> {entity_type}")
    
    print("\n✓ NLTK NER completed successfully!")
    
except Exception as e:
    print(f"✗ Error with NLTK: {e}")
    print("  Install with: pip install nltk")
    print("  Download data: nltk.download('punkt'), nltk.download('averaged_perceptron_tagger')")
    print("                 nltk.download('maxent_ne_chunker'), nltk.download('words')")


# =============================================================================
# Part 3: NER using Transformers (BERT)
# =============================================================================
print("\n" + "="*80)
print("PART 3: Named Entity Recognition using Transformers (BERT)")
print("="*80)

try:
    from transformers import pipeline
    
    # Load pre-trained NER pipeline
    print("\nLoading BERT-based NER model...")
    ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    print("✓ Model loaded!")
    
    # Sample text
    text = """
    Elon Musk is the CEO of Tesla and SpaceX. He was born in Pretoria, South Africa. 
    Tesla's headquarters is in Austin, Texas. SpaceX successfully launched Starship in 2023.
    """
    
    # Perform NER
    results = ner_pipeline(text)
    
    print("\nNamed Entities (BERT):\n")
    print(f"{'Entity':<30} {'Type':<15} {'Confidence'}")
    print("-" * 60)
    
    for entity in results:
        print(f"{entity['word']:<30} {entity['entity_group']:<15} {entity['score']:.4f}")
    
    # Group by type
    print("\n\nGrouped Entities (confidence >= 0.85):\n")
    entities_by_type = {}
    for entity in results:
        if entity['score'] >= 0.85:
            entity_type = entity['entity_group']
            entity_text = entity['word']
            
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append(entity_text)
    
    for entity_type, entity_list in entities_by_type.items():
        print(f"{entity_type}: {', '.join(entity_list)}")
    
    print("\n✓ Transformer NER completed successfully!")
    
except Exception as e:
    print(f"✗ Error with Transformers: {e}")
    print("  Install with: pip install transformers torch")


# =============================================================================
# Part 4: Comparison of All Methods
# =============================================================================
print("\n" + "="*80)
print("PART 4: COMPARISON OF ALL NER METHODS")
print("="*80)

comparison_text = "Elon Musk founded Tesla in California and SpaceX in Texas."

print(f"\nTest Text: {comparison_text}\n")

# Try spaCy
try:
    print("\n1. spaCy Results:")
    print("-" * 70)
    doc_spacy = nlp(comparison_text)
    for ent in doc_spacy.ents:
        print(f"  {ent.text:<20} -> {ent.label_}")
except:
    print("  (spaCy not available)")

# Try NLTK
try:
    print("\n2. NLTK Results:")
    print("-" * 70)
    tokens = word_tokenize(comparison_text)
    pos_tags = pos_tag(tokens)
    chunks = ne_chunk(pos_tags)
    for chunk in chunks:
        if hasattr(chunk, 'label'):
            entity = ' '.join(c[0] for c in chunk)
            entity_type = chunk.label()
            print(f"  {entity:<20} -> {entity_type}")
except:
    print("  (NLTK not available)")

# Try Transformers
try:
    print("\n3. BERT Results:")
    print("-" * 70)
    results_bert = ner_pipeline(comparison_text)
    for entity in results_bert:
        print(f"  {entity['word']:<20} -> {entity['entity_group']} (score: {entity['score']:.3f})")
except:
    print("  (Transformers not available)")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print("""
Methods Demonstrated:
1. spaCy       - Fast, accurate, production-ready
2. NLTK        - Educational, basic NER capabilities
3. Transformers - State-of-the-art accuracy (BERT-based)
4. Custom      - BiLSTM-CRF (see notebook for implementation)

Common Entity Types:
- PER (Person)        - Names of people
- ORG (Organization)  - Companies, institutions
- LOC (Location)      - Cities, countries, regions
- DATE (Date)         - Dates and time expressions
- MONEY (Money)       - Monetary values
- GPE (Geo-Political) - Countries, cities, states

Applications:
- Information extraction from documents
- Question answering systems
- Content classification and tagging
- Knowledge graph construction
- Resume parsing
- Customer feedback analysis

Installation Commands:
  pip install spacy transformers torch nltk
  python -m spacy download en_core_web_sm
  
  # In Python:
  import nltk
  nltk.download('punkt')
  nltk.download('averaged_perceptron_tagger')
  nltk.download('maxent_ne_chunker')
  nltk.download('words')
""")

print("="*80)
print("END OF DEMONSTRATION")
print("="*80)
