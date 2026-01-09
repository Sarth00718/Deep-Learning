# Practical 11: Named Entity Recognition (NER)

**Author:** NAROLA SARTH DHARMESHBHAI  
**Roll No:** 23BCE194  
**Subject:** NATURAL LANGUAGE PROCESSING

## Overview

This practical demonstrates various approaches to Named Entity Recognition (NER), a fundamental NLP task that identifies and classifies named entities in text into predefined categories.

## Contents

1. **named_entity_recognition.ipynb** - Complete Jupyter notebook with all implementations
2. **ner_demo.py** - Standalone Python script for quick demonstration
3. **README.md** - This file

## Methods Covered

### 1. spaCy (Pre-trained)
- Fast and accurate
- Production-ready
- Easy to use
- Supports multiple languages

### 2. NLTK
- Educational purposes
- Basic NER capabilities
- Good for understanding fundamentals
- Rule-based approach

### 3. Transformers (BERT-based)
- State-of-the-art accuracy
- Deep learning approach
- Pre-trained on large corpora
- Higher computational requirements

### 4. Custom BiLSTM-CRF
- Full control over architecture
- Requires training data
- Can be fine-tuned for specific domains
- PyTorch implementation

## Entity Types

Common entity types recognized:

- **PER** (Person) - Names of people
- **ORG** (Organization) - Companies, institutions, agencies
- **LOC** (Location) - Cities, countries, regions
- **GPE** (Geo-Political Entity) - Countries, cities, states
- **DATE** - Dates and time expressions
- **MONEY** - Monetary values
- **PERCENT** - Percentages
- **PRODUCT** - Products and objects
- **EVENT** - Named events

## Installation

### Required Packages

```bash
# Core packages
pip install spacy transformers torch nltk

# Download spaCy model
python -m spacy download en_core_web_sm

# Download NLTK data (in Python)
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
```

## Usage

### Running the Jupyter Notebook

```bash
jupyter notebook named_entity_recognition.ipynb
```

### Running the Demo Script

```bash
python ner_demo.py
```

## Example Output

```
Text: "Apple Inc. was founded by Steve Jobs in Cupertino, California."

Entities:
  Apple Inc.     -> ORG (Organization)
  Steve Jobs     -> PER (Person)
  Cupertino      -> LOC (Location)
  California     -> LOC (Location)
```

## Applications

1. **Information Extraction**
   - Extract structured data from unstructured text
   - Build knowledge bases

2. **Question Answering**
   - Identify entities in questions and documents
   - Improve answer accuracy

3. **Content Classification**
   - Categorize documents based on entities
   - Tag and organize content

4. **Knowledge Graph Construction**
   - Identify entities and relationships
   - Build semantic networks

5. **Resume Parsing**
   - Extract names, companies, locations
   - Automate candidate screening

6. **Customer Feedback Analysis**
   - Identify products, features mentioned
   - Track brand mentions

## Performance Comparison

| Method        | Speed | Accuracy | Ease of Use | Customization |
|---------------|-------|----------|-------------|---------------|
| spaCy         | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐    | ⭐⭐⭐⭐⭐      | ⭐⭐⭐         |
| NLTK          | ⭐⭐⭐⭐  | ⭐⭐⭐     | ⭐⭐⭐⭐       | ⭐⭐          |
| Transformers  | ⭐⭐⭐   | ⭐⭐⭐⭐⭐   | ⭐⭐⭐⭐       | ⭐⭐⭐⭐        |
| Custom BiLSTM | ⭐⭐⭐   | ⭐⭐⭐⭐    | ⭐⭐          | ⭐⭐⭐⭐⭐       |

## Key Concepts

### BIO Tagging Scheme
- **B-** (Beginning) - First token of an entity
- **I-** (Inside) - Continuation of an entity
- **O** (Outside) - Not part of any entity

Example:
```
Words: [Apple, Inc, is, in, Cupertino]
Tags:  [B-ORG, I-ORG, O, O, B-LOC]
```

### Evaluation Metrics
- **Precision**: Percentage of predicted entities that are correct
- **Recall**: Percentage of actual entities that were found
- **F1-Score**: Harmonic mean of precision and recall

## Advanced Topics

1. **Domain-Specific NER**
   - Medical entities (diseases, drugs)
   - Legal entities (laws, cases)
   - Financial entities (stocks, currencies)

2. **Multilingual NER**
   - Cross-lingual transfer
   - Language-specific models

3. **Few-Shot NER**
   - Learning from limited examples
   - Meta-learning approaches

4. **Nested NER**
   - Entities within entities
   - Hierarchical structures

## Troubleshooting

### Common Issues

1. **spaCy model not found**
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **NLTK data missing**
   ```python
   import nltk
   nltk.download('all')  # Download all data
   ```

3. **Transformers slow on CPU**
   - Use smaller models
   - Consider GPU acceleration
   - Batch processing

## References

- [spaCy Documentation](https://spacy.io/)
- [NLTK Book](https://www.nltk.org/book/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [BiLSTM-CRF Paper](https://arxiv.org/abs/1508.01991)

## Future Enhancements

- [ ] Add more pre-trained models
- [ ] Implement entity linking
- [ ] Add relation extraction
- [ ] Create custom training pipeline
- [ ] Add visualization tools
- [ ] Support more languages

## License

Educational use only.

---

**Note:** This practical is designed for educational purposes to understand various NER approaches and their trade-offs.
