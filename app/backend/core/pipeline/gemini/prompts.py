from __future__ import annotations
from typing import Any

GEMINI_ENTITY_EXTRACTION_PROMPT = """
---Goal---
You are an expert knowledge graph builder tasked with extracting a structured representation of entities and relationships from the provided text. The extracted information will be used to construct a knowledge graph with two distinct levels.

---Instructions---
Analyze the provided text and identify:

1. All significant entities with these attributes:
   - entity_name: Unique identifier for the entity (use the entity's name)
   - entity_type: A short name for the entity type (e.g., MEDICATION, CONDITION, SYMPTOM, etc.)
   - entity_description: A description of the entity using information from the text

2. All meaningful relationships between the identified entities:
   - source_entity: Name of the source entity (must match an entity_name you identified)
   - target_entity: Name of the target entity (must match an entity_name you identified) 
   - relationship_type: Type of relationship (e.g., TREATS, CAUSES, PART_OF, RELATED_TO, etc.)
   - relationship_description: Clear description of how the entities are related

---Important Notes---
- Be comprehensive but precise - identify all significant entities and meaningful relationships
- Ensure all relationship endpoints refer to entities that exist in your entities list
- Do not create relationships between entities not present in the text
- Maintain consistent naming of entities across your response

___Input Text___
{input}

"""


GRAPH_FIELD_SEP = "<SEP>"

PROMPTS: dict[str, Any] = {}

PROMPTS["DEFAULT_LANGUAGE"] = "English"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

PROMPTS["DEFAULT_ENTITY_TYPES"] = [
    "organization", "person", "geo", "event", "category"]

# PROMPTS["entity_extraction"] = """---Goal---
# Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.
# Use {language} as output language.

# ---Steps---
# 1. Identify all entities. For each identified entity, extract the following information:
# - entity_name: Name of the entity, use same language as input text. If English, capitalized the name.
# - entity_type: One of the following types: [{entity_types}]
# - entity_description: Comprehensive description of the entity's attributes and activities
# Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

# 2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
# For each pair of related entities, extract the following information:
# - source_entity: name of the source entity, as identified in step 1
# - target_entity: name of the target entity, as identified in step 1
# - relationship_description: explanation as to why you think the source entity and the target entity are related to each other
# - relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
# - relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details
# Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

# 3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire text. These should capture the overarching ideas present in the document.
# Format the content-level key words as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

# 4. Translate output in {language} as a single list of all the entities and relationships identified in steps 1 and 2, ensure all entiy and relationship in {language}.


# Use **{record_delimiter}** as the list delimiter.
# ######################
# ---Examples---
# ######################
# {examples}

# #############################
# ---Output Format---
# #############################
# Json:
#     entities: List of entity
#     relationships: List of relationship

# Entity:
#     entity_name: str
#     entity_type: str
#     entity_description: str


# Relationship:
#     source_entity: str
#     target_entity: str
#     description: str
#     strength: float
#     keywords: List[str]


# ---Real Data---
# ######################
# Entity_types: [{entity_types}]

# ######################
# INPUT Text:
# {input_text}
# """

PROMPTS["personal_info_extraction"] = """
Extract all personal information from the following message.

User message: "{query}"

IMPORTANT: Return ONLY a valid JSON object with these fields (include only fields that are mentioned):
- name: User's full name
- age: User's age (as a number)
- gender: User's gender
- contact: Contact information like phone or email
- health_conditions: List of any mentioned health conditions
- diabetes_type: Type of diabetes if mentioned (Type 1, Type 2, etc.)
- medications: List of any medications mentioned

DO NOT include any introduction text, explanation, or conclusion.
DO NOT wrap the JSON in code blocks.
ONLY output the JSON object itself.
"""


PROMPTS["entity_extraction_low_high"] = """
Create structured entities from these keywords related to diabetes:

High-level keywords: {high_level}
Low-level keywords: {low_level}

For each keyword, create an entity with:
1. entity_name: The keyword itself (capitalized if a proper noun)
2. entity_type: One of: CONDITION, MEDICATION, TREATMENT, COMPLICATION, SYMPTOM, DIAGNOSTIC_TEST, RISK_FACTOR, LIFESTYLE_FACTOR, SPECIALIST, ORGANIZATION, RESEARCH, CONCEPT
3. entity_description: A comprehensive description of what this entity is, focusing on diabetes relevance
4. level: Either "high" or "low" based on the source keyword list

Return only a JSON object with an "entities" array containing these entity objects.

Example format:
{{
  "entities": [
    {{
      "entity_name": "Blood Glucose Monitoring",
      "entity_type": "DIAGNOSTIC_TEST",
      "entity_description": "The process of testing blood sugar levels regularly to manage diabetes.",
      "level": "low"
    }},
    {{
      "entity_name": "Diabetic Complications",
      "entity_type": "CONCEPT",
      "entity_description": "A general term for health problems that can develop as a result of poorly managed diabetes over time.",
      "level": "high"
    }}
  ]
}}
"""


PROMPTS["entity_extraction"] = """
Create structured entities from these keywords related to diabetes:

{input_text}

For each entity, provide:
1. entity_name: The name of the entity
2. entity_type: The type from this list: {entity_types}
3. entity_description: A comprehensive description

Return ONLY a JSON object with an "entities" array.

Example format:
{{
    "entities": [
    {{
        "entity_name": "Blood Glucose Monitoring",
        "entity_type": "DIAGNOSTIC_TEST",
        "entity_description": "The process of testing blood sugar levels regularly to manage diabetes."
    }},
    {{
        "entity_name": "Insulin",
        "entity_type": "MEDICATION",
        "entity_description": "A hormone that regulates blood sugar levels, often used as medication for diabetes treatment."
    }}
    ]
}}

######################
---Examples---
######################
{examples}

#############################
---Output Format---
#############################
Json:
    entities: List of entity objects
    relationships: List of relationship objects

Entity:
    entity_name: str (in {language})
    entity_type: str (from provided types)
    entity_description: str (in {language})

Relationship:
    source_entity: str (in {language})
    target_entity: str (in {language})
    description: str (in {language})
    strength: float
    keywords: List[str] (all in {language})

---Important Note---
Ensure that ALL text in the response (entity names, descriptions, relationship details, keywords) is in {language}, regardless of the input language.

---Real Data---
######################
Entity_types: [{entity_types}]
Input Language: {language}

Text:
{input_text}
######################
"""


PROMPTS["entity_extraction_examples"] = """
Example 1:

High-level keywords: ["Diabetes management", "Blood sugar control"]
Low-level keywords: ["Insulin", "Exercise", "Diet"]

Output:
{
    "entities": [
        {
            "entity_name": "Insulin",
            "entity_type": "MEDICATION",
            "entity_description": "A hormone used to regulate blood sugar levels, commonly used in diabetes treatment.",
            "level": "low"
        },
        {
            "entity_name": "Exercise",
            "entity_type": "LIFESTYLE_FACTOR",
            "entity_description": "Physical activity that helps improve insulin sensitivity and manage blood sugar levels.",
            "level": "low"
        },
        {
            "entity_name": "Diet",
            "entity_type": "LIFESTYLE_FACTOR",
            "entity_description": "A controlled eating plan to manage blood sugar levels and overall health in diabetes management.",
            "level": "low"
        },
        {
            "entity_name": "Diabetes management",
            "entity_type": "CONCEPT",
            "entity_description": "The overall process of monitoring and controlling diabetes through medication, lifestyle changes, and regular check-ups.",
            "level": "high"
        },
        {
            "entity_name": "Blood sugar control",
            "entity_type": "CONCEPT",
            "entity_description": "The practice of maintaining blood glucose levels within a target range to prevent complications.",
            "level": "high"
        }
    ]
}

Example 2:

High-level keywords: ["Diabetes complications", "Chronic disease management"]
Low-level keywords: ["Neuropathy", "Retinopathy", "Nephropathy"]

Output:
{
    "entities": [
        {
            "entity_name": "Neuropathy",
            "entity_type": "COMPLICATION",
            "entity_description": "Nerve damage caused by high blood sugar levels, often leading to pain or loss of sensation.",
            "level": "low"
        },
        {
            "entity_name": "Retinopathy",
            "entity_type": "COMPLICATION",
            "entity_description": "Damage to the blood vessels in the retina due to prolonged high blood sugar levels, potentially leading to vision loss.",
            "level": "low"
        },
        {
            "entity_name": "Nephropathy",
            "entity_type": "COMPLICATION",
            "entity_description": "Kidney damage resulting from long-term diabetes, which can lead to kidney failure if untreated.",
            "level": "low"
        },
        {
            "entity_name": "Diabetes complications",
            "entity_type": "CONCEPT",
            "entity_description": "Health problems that arise as a result of poorly managed diabetes over time.",
            "level": "high"
        },
        {
            "entity_name": "Chronic disease management",
            "entity_type": "CONCEPT",
            "entity_description": "The ongoing process of managing long-term health conditions like diabetes to improve quality of life.",
            "level": "high"
        }
    ]
}
"""

PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.
Use {language} as output language.

#######
Output Format:
JSON:
    sum_description: str

#######
---Data---
Entities: {entity_name}
Description List: {description_list}
"""

PROMPTS["entity_continue_extraction"] = """
MANY entities and relationships were missed in the last extraction.

---Remember Steps---

1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text. If English, capitalized the name.
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
- relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire text. These should capture the overarching ideas present in the document.
Format the content-level key words as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. Return output in {language} as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

5. When finished, output {completion_delimiter}

---Output---

Add them below using the same format:\n
""".strip()

PROMPTS["entity_if_loop_extraction"] = """
---Goal---'

It appears some entities may have still been missed.

---Output---

Answer ONLY by `YES` OR `NO` if there are still entities that need to be added.
""".strip()

PROMPTS["fail_response"] = (
    "Sorry, I'm not able to provide an answer to that question.[no-context]"
)

PROMPTS["rag_response"] = """---Role---

You are a helpful assistant responding to user query about Knowledge Base provided below.


---Goal---

Generate a concise response based on Knowledge Base and follow Response Rules, considering both the conversation history and the current query. Summarize all information in the provided Knowledge Base, and incorporating general knowledge relevant to the Knowledge Base. Do not include information not provided by Knowledge Base.

When handling relationships with timestamps:
1. Each relationship has a "created_at" timestamp indicating when we acquired this knowledge
2. When encountering conflicting relationships, consider both the semantic content and the timestamp
3. Don't automatically prefer the most recently created relationships - use judgment based on the context
4. For time-specific queries, prioritize temporal information in the content before considering creation timestamps

---Conversation History---
{history}

---Knowledge Base---
{context_data}

---Response Rules---

- Target format and length: {response_type}
- Use markdown formatting with appropriate section headings
- Please respond in the same language as the user's question.
- Ensure the response maintains continuity with the conversation history.
- List up to 5 most important reference sources at the end under "References" section. Clearly indicating whether each source is from Knowledge Graph (KG) or Vector Data (DC), and include the file path if available, in the following format: [KG/DC] Source content (File: file_path)
- If you don't know the answer, just say so.
- Do not make anything up. Do not include information not provided by the Knowledge Base."""

PROMPTS["keywords_extraction"] = """---Role---

You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query and conversation history.

---Goal---

Given the query and conversation history, list both high-level and low-level keywords. High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.

---Instructions---

- Consider both the current query and relevant conversation history when extracting keywords
- Output the keywords in JSON format, it will be parsed by a JSON parser, do not add any extra content in output
- Return ONLY a JSON object with two arrays:
  - "high_level_keywords" for overarching concepts or themes
  - "low_level_keywords" for specific entities or details


Conversation History:
{history}

Current Query: {query}


"""

PROMPTS["keywords_extraction_examples"] = [
    """Example 1:

Query: "What are the long-term complications of Type 2 diabetes?"
################
Output:
{
  "high_level_keywords": ["Diabetes complications", "Long-term effects", "Type 2 diabetes", "Chronic disease management"],
  "low_level_keywords": ["Neuropathy", "Retinopathy", "Nephropathy", "Cardiovascular disease", "Foot problems"]
}
#############################""",
    """Example 2:

Query: "How does insulin resistance develop and what can I do to prevent it?"
################
Output:
{
  "high_level_keywords": ["Insulin resistance", "Prevention", "Metabolic health", "Diabetes risk factors"],
  "low_level_keywords": ["Diet", "Exercise", "Body weight", "Glucose metabolism", "Insulin sensitivity"]
}
#############################""",
    """Example 3:

Query: "What's the relationship between blood sugar levels and A1C test results?"
################
Output:
{
  "high_level_keywords": ["Blood glucose monitoring", "Glycemic control", "Diabetes diagnosis"],
  "low_level_keywords": ["A1C test", "Blood sugar levels", "Glucose measurements", "Testing frequency", "Glycated hemoglobin"]
}
#############################""",
    """Example 4:

Query: "Can diabetic neuropathy be reversed or is it permanent?"
################
Output:
{
  "high_level_keywords": ["Diabetic neuropathy", "Nerve damage", "Diabetes complications", "Treatment outcomes"],
  "low_level_keywords": ["Nerve pain", "Symptom reversal", "Progression", "Peripheral neuropathy", "Sensation loss"]
}
#############################""",
    """Example 5:

Query: "What are the best foods to eat to manage blood sugar spikes after meals?"
################
Output:
{
  "high_level_keywords": ["Diabetes diet", "Blood sugar management", "Nutritional therapy", "Glycemic control"],
  "low_level_keywords": ["Postprandial glucose", "Fiber", "Low glycemic index", "Carbohydrates", "Meal planning"]
}
#############################"""
]


PROMPTS["naive_rag_response"] = """---Role---

You are a helpful assistant responding to user query about Document Chunks provided below.

---Goal---

Generate a concise response based on Document Chunks and follow Response Rules, considering both the conversation history and the current query. Summarize all information in the provided Document Chunks, and incorporating general knowledge relevant to the Document Chunks. Do not include information not provided by Document Chunks.

When handling content with timestamps:
1. Each piece of content has a "created_at" timestamp indicating when we acquired this knowledge
2. When encountering conflicting information, consider both the content and the timestamp
3. Don't automatically prefer the most recent content - use judgment based on the context
4. For time-specific queries, prioritize temporal information in the content before considering creation timestamps

---Conversation History---
{history}

---Document Chunks---
{content_data}

---Response Rules---

- Target format and length: {response_type}
- Use markdown formatting with appropriate section headings
- Please respond in the same language as the user's question.
- Ensure the response maintains continuity with the conversation history.
- List up to 5 most important reference sources at the end under "References" section. Clearly indicating whether each source is from Knowledge Graph (KG) or Vector Data (DC), and include the file path if available, in the following format: [KG/DC] Source content (File: file_path)
- If you don't know the answer, just say so.
- Do not include information not provided by the Document Chunks."""


PROMPTS[
    "similarity_check"
] = """Please analyze the similarity between these two questions:

Question 1: {original_prompt}
Question 2: {cached_prompt}

Please evaluate whether these two questions are semantically similar, and whether the answer to Question 2 can be used to answer Question 1, provide a similarity score between 0 and 1 directly.

Similarity score criteria:
0: Completely unrelated or answer cannot be reused, including but not limited to:
   - The questions have different topics
   - The locations mentioned in the questions are different
   - The times mentioned in the questions are different
   - The specific individuals mentioned in the questions are different
   - The specific events mentioned in the questions are different
   - The background information in the questions is different
   - The key conditions in the questions are different
1: Identical and answer can be directly reused
0.5: Partially related and answer needs modification to be used
Return only a number between 0-1, without any additional content.
"""

PROMPTS["mix_rag_response"] = """---Role---

You are a helpful assistant responding to user query about Data Sources provided below.


---Goal---

Generate a concise response based on Data Sources and follow Response Rules, considering both the conversation history and the current query. Data sources contain two parts: Knowledge Graph(KG) and Document Chunks(DC). Summarize all information in the provided Data Sources, and incorporating general knowledge relevant to the Data Sources. Do not include information not provided by Data Sources.

When handling information with timestamps:
1. Each piece of information (both relationships and content) has a "created_at" timestamp indicating when we acquired this knowledge
2. When encountering conflicting information, consider both the content/relationship and the timestamp
3. Don't automatically prefer the most recent information - use judgment based on the context
4. For time-specific queries, prioritize temporal information in the content before considering creation timestamps

---Conversation History---
{history}

---Data Sources---

1. From Knowledge Graph(KG):
{kg_context}

2. From Document Chunks(DC):
{vector_context}

---Response Rules---

- Target format and length: {response_type}
- Use markdown formatting with appropriate section headings
- Please respond in the same language as the user's question.
- Ensure the response maintains continuity with the conversation history.
- Organize answer in sections focusing on one main point or aspect of the answer
- Use clear and descriptive section titles that reflect the content
- List up to 5 most important reference sources at the end under "References" section. Clearly indicating whether each source is from Knowledge Graph (KG) or Vector Data (DC), and include the file path if available, in the following format: [KG/DC] Source content (File: file_path)
- If you don't know the answer, just say so. Do not make anything up.
- Do not include information not provided by the Data Sources."""


"""
Custom prompts for knowledge graph query processing.

This module contains updated prompt templates that prevent reference fabrication
and ensure responses only cite information from the knowledge graph.
"""

# This prompt should be added to your existing prompts.py file
CUSTOM_PROMPTS = {
    "rag_response": """---Role---

You are a helpful assistant responding to a user query about diabetes, using only the Knowledge Base provided below.

---Goal---

Generate a concise response based on Knowledge Base and follow Response Rules, considering both the conversation history and the current query. Summarize information provided in the Knowledge Base, and do not include any information not present in the Knowledge Base.

---Conversation History---
{history}

---Knowledge Base---
{context_data}

---Response Rules---

- Target format and length: {response_type}
- Use markdown formatting with appropriate section headings
- Please respond in the same language as the user's question
- Ensure the response maintains continuity with the conversation history
- IMPORTANT: Only reference source IDs that appear in the Knowledge Base with the format "Source ID: [identifier]"
- DO NOT create, invent, or fabricate any external sources, websites, publications, or organizations
- If you don't have enough information to answer the question, simply state what you do know and acknowledge the limitations
- DO NOT make anything up. Do not include information not provided by the Knowledge Base.""",

    "mix_rag_response": """---Role---

You are a helpful assistant responding to user query about diabetes, using only the Data Sources provided below.

---Goal---

Generate a concise response based on Data Sources and follow Response Rules, considering both the conversation history and the current query. Data sources contain general concepts (Level 1) and specific details (Level 2). Summarize information in the provided Data Sources, and do not include any information not present in these sources.

---Conversation History---
{history}

---Data Sources---

{context_data}

---Response Rules---

- Target format and length: {response_type}
- Use markdown formatting with appropriate section headings
- Please respond in the same language as the user's question
- Ensure the response maintains continuity with the conversation history
- Organize answer in sections focusing on one main point or aspect of the answer
- IMPORTANT: Only reference source IDs that appear in the Knowledge Base with the format "Source ID: [identifier]"
- DO NOT create, invent, or fabricate any external sources, websites, publications, or organizations
- If you don't have enough information to answer the question, simply state what you do know and acknowledge the limitations
- DO NOT make anything up. Do not include information not provided by the Data Sources."""
}

# Add a function to get the updated prompts


def get_updated_prompts(original_prompts):
    """
    Update the original prompts with the custom prompts.

    Args:
        original_prompts: Original prompts dictionary

    Returns:
        Updated prompts dictionary
    """
    # Create a copy of the original prompts
    updated_prompts = dict(original_prompts)

    # Update with custom prompts
    for key, value in CUSTOM_PROMPTS.items():
        updated_prompts[key] = value

    return updated_prompts


########################################################
# Score entity candidates prompt
PROMPTS["score_entity_candidates"] = """Please score the entities' contribution to the question on a scale from 0 to 1 (the sum of the scores of all entities is 1).
Q: What medications are commonly prescribed for Type 2 diabetes?
Relation: disease.medications
Entities: Metformin; Insulin; Sulfonylureas; DPP-4 inhibitors; GLP-1 receptor agonists; SGLT2 inhibitors
Score: 0.3, 0.2, 0.2, 0.1, 0.1, 0.1
Metformin is the most commonly prescribed first-line medication for Type 2 diabetes, followed by Insulin and Sulfonylureas. Therefore, Metformin gets the highest score.

Q: {query}
Relation: {relation}
Entities: {entities}"""

# Evaluate information sufficiency prompt
PROMPTS["evaluate_information"] = """Given a question and the associated retrieved knowledge graph triplets (entity, relation, entity), you are asked to answer whether it's sufficient for you to answer the question with these triplets and your knowledge (Yes or No).

Q: What are the recommended dietary changes for managing Type 2 diabetes?
Knowledge Triplets: 
Type 2 Diabetes, disease.dietary_recommendations, Reduce carbohydrate intake
Type 2 Diabetes, disease.dietary_recommendations, Increase fiber consumption
Type 2 Diabetes, disease.dietary_recommendations, Limit processed foods
A: Yes. The given knowledge triplets provide sufficient information about the recommended dietary changes for managing Type 2 diabetes, including reducing carbohydrate intake, increasing fiber consumption, and limiting processed foods.

Q: What are the long-term complications of Type 2 diabetes?
Knowledge Triplets: 
Type 2 Diabetes, disease.complications, Heart disease
Type 2 Diabetes, disease.complications, Kidney damage
A: No. While the given knowledge triplets mention some complications (heart disease and kidney damage), they don't provide a comprehensive list of all possible long-term complications of Type 2 diabetes, such as nerve damage, eye problems, and foot complications.

Q: {query}
Knowledge Triplets: {triplets}"""


# Validator prompt
PROMPTS["validator"] = """You are a medical information validator. Evaluate whether the provided information is accurate and sufficient for the given diabetes-related question.

Question: {query}
Query Intent: {intent}
Retrieved Information: {retrieved_info}
Ground Truth: {ground_truth}
Task: Is the retrieved information accurate and sufficient? Answer with Yes or No, followed by a brief explanation."""

# Commentor prompt
PROMPTS["commentor"] = """You are a specialized diabetes information assistant. Review and provide feedback on the extracted information from the question.

Question: {query}
Query Intent: {intent}
High Level Keywords: {high_level_keywords}
Low Level Keywords: {low_level_keywords}
Retrieved Level 1 Nodes: {level1_nodes}
Retrieved Level 2 Nodes: {level2_nodes}

Task: Please identify any issues with:
1. Query Intent Classification: Is it correctly classified?
2. High Level Keywords: Are they correctly identified and relevant?
3. Low Level Keywords: Are they correctly identified and specific enough?
4. Level 1 Nodes: Are they relevant to the question?
5. Level 2 Nodes: Do they provide sufficient detail?

Provide specific feedback for each aspect that needs improvement."""

# Generator prompt
PROMPTS["generator"] = """You are a specialized diabetes information assistant. Generate a comprehensive, evidence-based response to the user's diabetes-related query.

Question: {query}
Query Intent: {intent}
Retrieved Information: {retrieved_info}
Knowledge Graph Context: {kg_context}
Latest Research: {latest_research}

Instructions:
1. Synthesize information from all available sources
2. Prioritize evidence-based medical information
3. Include relevant clinical guidelines when applicable
4. Note any limitations or uncertainties
5. Format the response clearly with appropriate medical terminology
6. Include relevant statistics or research findings when available

Generate a well-structured, informative response."""

# Evaluator prompt
PROMPTS["evaluator"] = """You are a medical information evaluator. Assess the quality and accuracy of the generated response for a diabetes-related question.

Question: {query}
Query Intent: {intent}
Generated Response: {response}
Ground Truth: {ground_truth}

Evaluation Criteria:
1. Medical Accuracy (0-1)
2. Completeness (0-1)
3. Clarity (0-1)
4. Evidence-based (0-1)
5. Patient-friendly (0-1)

Task: Evaluate the response based on these criteria and provide a brief explanation for each score."""
