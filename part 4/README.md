PART 4 – LLM-POWERED FEATURE

Track Chosen

Track C – Model Prediction Explanation Pipeline

Overview

This part extends the machine learning pipeline developed in Parts 2 and
3 by integrating a Large Language Model (LLM) to generate structured
explanations for model predictions. The trained Random Forest pipeline
(best_model.pkl) predicts loan default risk, while the LLM converts the
prediction into an easy-to-understand JSON explanation. A PII guardrail
prevents sensitive information from being sent to the LLM.

Task 1 – LLM API Connection

-   API key stored securely in an environment variable (LLM_API_KEY).
-   Implemented reusable call_llm() using the requests library.
-   HTTP POST request contains:
    -   model
    -   messages
    -   temperature
    -   max_tokens
-   Response status validated before parsing.
-   Test prompt:
    -   Prompt: “Reply with only the word: hello”
    -   Output: hello

Task 2 – Prompt Design

System Prompt

You are an AI assistant that explains loan default predictions. Return
ONLY valid JSON with: - prediction_label - confidence_level -
top_reason - second_reason - next_step

Do not include markdown or additional text.

User Prompt Template

Feature Values: {feature_values}

Predicted Class: {predicted_class}

Predicted Probability: {predicted_probability}

Explain the prediction as valid JSON.

Temperature Choice

Temperature=0 was selected to produce deterministic, repeatable JSON
outputs suitable for validation. Temperature=0.7 produces more varied
wording and less predictable responses.

Temperature Comparison

  -----------------------------------------------------------------------
  Input             Temp=0            Temp=0.7          Difference
  ----------------- ----------------- ----------------- -----------------
  Sample 1          Consistent JSON   More varied       Higher creativity
                                      wording           

  Sample 2          Consistent JSON   Slight wording    Less
                                      changes           deterministic

  Sample 3          Consistent JSON   Alternative       More diversity
                                      explanations      
  -----------------------------------------------------------------------

Temperature 0 always selects the highest-probability token, making
outputs reproducible. Temperature 0.7 samples from a broader probability
distribution, increasing linguistic variation.

Task 3 – Structured Output Validation

Implemented JSON schema validation using jsonschema.

Required fields: - prediction_label - confidence_level - top_reason -
second_reason - next_step

Validation pipeline: 1. Receive LLM response. 2. Strip whitespace. 3.
Parse using json.loads(). 4. Validate using jsonschema.validate(). 5. On
failure return fallback JSON.

All three prediction examples successfully passed schema validation.

  Input      Predicted Class   Probability   Validation
  ---------- ----------------- ------------- ------------
  Record 1   0                 0.065         PASS
  Record 2   0                 0.290         PASS
  Record 3   1                 0.790         PASS

Task 4 – Guardrails

Implemented regex-based PII detection before every LLM call.

Detected: - Email addresses - Phone numbers

Behaviour: - PII detected → Block request - No PII → Continue to LLM

Two demonstration tests: 1. Input containing email → Blocked 2. Input
without PII → Allowed

Task 5 – End-to-End Demonstration

Three hand-crafted feature vectors were processed.

Pipeline: Input Features → Model Prediction → Prediction Probability →
Guardrail Check → LLM Explanation → JSON Validation

Results:

  Input      LLM Output                 Valid JSON   Guardrail
  ---------- -------------------------- ------------ -----------
  Record 1   Approved explanation       PASS         PASS
  Record 2   Low-risk explanation       PASS         PASS
  Record 3   Default-risk explanation   PASS         PASS

Project Structure

part4/ ├── README.md ├── app.py ├── predictor.py ├── llm.py ├──
guardrails.py ├── schema.py ├── requirements.txt ├── .env.example └──
best_model.pkl

Streamlit Application

The application allows users to: - Enter loan applicant details -
Predict default risk - View prediction probability - Generate LLM
explanation - Validate JSON output - Download explanation -
Automatically block PII

Conclusion

The integrated system combines machine learning with LLM-generated
explanations to provide transparent loan risk predictions. JSON schema
validation guarantees structured responses, while regex-based guardrails
prevent accidental transmission of sensitive information. The Streamlit
application demonstrates an end-to-end, production-style AI workflow.
