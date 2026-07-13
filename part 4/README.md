PART 4 – LLM-POWERED FEATURE

Track Chosen

Track C – Model Prediction Explanation Pipeline

Part 4 — Selected Feature Track

For Part 4, I selected Track C: Model Prediction Explanation Pipeline.

The best-performing machine-learning Pipeline from Part 3 is loaded from best_model.pkl and used to generate loan-risk predictions for three hand-crafted applicant feature inputs. The predicted class and probability, together with the applicant's feature values, are passed to an LLM API.

The LLM generates a structured JSON explanation containing the prediction label, confidence level, primary reason, secondary reason, and recommended next step. Each LLM response is parsed and validated against a predefined JSON schema.

A PII guardrail is applied before every LLM API call to prevent inputs containing email addresses or phone numbers from being transmitted to the external LLM service.

This track was selected because it directly integrates the trained credit-risk model with an LLM-powered explanation layer, aligning with the project's objective of developing a Loan Risk Prediction and Financial Advisor system.

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



Loading the Trained Model

The best-performing machine learning pipeline developed in Part 3 was loaded using `joblib.load('best_model.pkl')`. The loaded model expects 17 encoded input features, consisting of numerical variables together with one-hot encoded categorical variables. Before prediction, every applicant record is transformed into the same feature representation used during model training through the `encode_record()` function.

For each applicant record, the pipeline performs the following sequence:

1. Encode the raw feature values into the trained feature space.
2. Predict the loan default class using `predict()`.
3. Predict the probability of default using `predict_proba()`.
4. Construct a structured prompt containing the applicant features, predicted class, and prediction probability.
5. Send the prompt to the LLM for explanation generation.

---

JSON Schema Validation

A JSON schema was defined using the `jsonschema` library to guarantee that every explanation returned by the LLM follows a consistent structure. The schema requires the following five scalar fields:

* `prediction_label`
* `confidence_level`
* `top_reason`
* `second_reason`
* `next_step`

After each API call, the returned text is stripped of whitespace using `response.strip()` and parsed using `json.loads()`. The parsed object is then validated using `jsonschema.validate()`.

Two levels of exception handling are implemented:

* `json.JSONDecodeError` handles malformed or invalid JSON responses.
* `jsonschema.ValidationError` handles responses that do not satisfy the required schema.

If either exception occurs, the pipeline returns a fallback dictionary in which all required fields are assigned `null`. This prevents invalid LLM responses from propagating into the application.

---

Validation Results

Three different applicant records were processed through the complete prediction and explanation pipeline.

| Feature Input | Predicted Class | Probability | Explanation JSON                               | Validation Status |
| ------------- | --------------- | ----------- | ---------------------------------------------- | ----------------- |
| Applicant 1   | 0 (Low Risk)    | 0.0650      | Valid JSON containing all five required fields | PASS              |
| Applicant 2   | 0 (Low Risk)    | 0.2900      | Valid JSON containing all five required fields | PASS              |
| Applicant 3   | 1 (High Risk)   | 0.7900      | Valid JSON containing all five required fields | PASS              |

All three responses were successfully parsed as valid JSON and satisfied the predefined schema without any validation errors.

---

Interpretation

The results demonstrate that the complete prediction explanation pipeline functions reliably. The machine learning model first generated a prediction and probability for each applicant, after which the LLM produced a structured explanation that accurately reflected the model output.

The low-risk applicants (Applicants 1 and 2) received explanations highlighting positive financial indicators such as higher income, lower loan burden, or longer credit history. The high-risk applicant (Applicant 3) received an explanation identifying important risk factors including a high loan-to-income ratio, shorter employment history, and previous credit default.

Because every response passed schema validation successfully, the pipeline proved capable of generating deterministic, machine-readable explanations suitable for downstream processing. The use of schema validation and fallback handling also improves the robustness of the application by preventing malformed LLM outputs from affecting later stages of the workflow.


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

End-to-End Demonstration

The complete prediction explanation pipeline was executed using three manually created loan application records. For each record, the following workflow was performed:

1. The applicant's feature values were encoded using the same preprocessing pipeline employed during model training.
2. The trained Random Forest model generated a prediction (`predict()`) and a probability score (`predict_proba()`).
3. A PII guardrail inspected the prompt using regular expressions. Since none of the inputs contained email addresses or phone numbers, all three requests passed the guardrail and were forwarded to the LLM.
4. The LLM generated a structured JSON explanation describing the prediction.
5. The JSON response was validated against the predefined schema using `jsonschema.validate()`. All three responses successfully passed schema validation.

End-to-End Results

| Input                                                                     | LLM Output                                                                                           | Valid JSON | Pass/Block |
| ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- | ---------- | ---------- |
| Applicant 1 (High income, Grade A, Low loan percentage)                   | Loan approved with low default risk; stable employment and low loan burden identified as key reasons |  Pass     |  Pass     |
| Applicant 2 (Moderate income, Grade C, Medical loan)                      | Loan approved with moderate confidence; acceptable credit history and manageable risk identified     |  Pass     |  Pass     |
| Applicant 3 (Low income, Grade E, Previous default, High loan percentage) | High default risk predicted; large loan burden and previous default highlighted as major factors     |  Pass     |  Pass     |



Conclusion

The integrated system combines machine learning with LLM-generated
explanations to provide transparent loan risk predictions. JSON schema
validation guarantees structured responses, while regex-based guardrails
prevent accidental transmission of sensitive information. The Streamlit
application demonstrates an end-to-end, production-style AI workflow.
