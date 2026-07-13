import os
import re
import json
import requests

from dotenv import load_dotenv
from jsonschema import validate
from jsonschema.exceptions import ValidationError

# ---------------------------------------------------------
# Load Environment Variables
# ---------------------------------------------------------

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

MODEL_NAME = "openai/gpt-4.1-mini"

# ---------------------------------------------------------
# JSON Schema
# ---------------------------------------------------------

OUTPUT_SCHEMA = {

    "type": "object",

    "properties": {

        "prediction_label": {"type": "string"},
        "confidence_level": {"type": "string"},
        "top_reason": {"type": "string"},
        "second_reason": {"type": "string"},
        "next_step": {"type": "string"}

    },

    "required": [

        "prediction_label",
        "confidence_level",
        "top_reason",
        "second_reason",
        "next_step"

    ]

}

# ---------------------------------------------------------
# PII Detection
# ---------------------------------------------------------

def has_pii(text):

    email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"

    phone_pattern = r"\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b"

    return bool(

        re.search(email_pattern, text)

        or

        re.search(phone_pattern, text)

    )

# ---------------------------------------------------------
# System Prompt
# ---------------------------------------------------------

SYSTEM_PROMPT = """
You are a loan prediction explanation assistant.

Always return ONLY valid JSON.

Do not return markdown.

Do not return additional text.

Return exactly:

{
"prediction_label":"",
"confidence_level":"",
"top_reason":"",
"second_reason":"",
"next_step":""
}
"""

# ---------------------------------------------------------
# LLM Call
# ---------------------------------------------------------

def call_llm(user_prompt,
             temperature=0):

    headers = {

        "Authorization": f"Bearer {API_KEY}",

        "Content-Type": "application/json"

    }

    payload = {

        "model": MODEL_NAME,

        "messages": [

            {

                "role":"system",

                "content":SYSTEM_PROMPT

            },

            {

                "role":"user",

                "content":user_prompt

            }

        ],

        "temperature":temperature,

        "max_tokens":300

    }

    response = requests.post(

        OPENROUTER_URL,

        headers=headers,

        json=payload

    )

    if response.status_code != 200:

        print(response.text)

        return None

    return response.json()["choices"][0]["message"]["content"]

# ---------------------------------------------------------
# JSON Validation
# ---------------------------------------------------------

def validate_json(response):

    try:

        data = json.loads(response.strip())

        validate(

            instance=data,

            schema=OUTPUT_SCHEMA

        )

        return data, True

    except ValidationError as e:

        print(e)

    except json.JSONDecodeError as e:

        print(e)

    fallback = {

        "prediction_label":None,

        "confidence_level":None,

        "top_reason":None,

        "second_reason":None,

        "next_step":None

    }

    return fallback, False