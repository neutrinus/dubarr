# LLM Prompt Templates for AI Dubbing Pipeline

T_ANALYSIS = """<|im_start|>system
CONTEXT: This text is a raw transcription from a video file. Your goal is to extract metadata to assist in automated AI dubbing production.

REFERENCE SUBTITLES (Ground Truth for names/context): "{subtitles}"

Analyze the movie script provided by the user. Return A SINGLE JSON object containing:
1. 'summary': A concise summary of the plot and context.
2. 'glossary': A dictionary of proper nouns, names, and specific terms.
   - For names and brands that should keep their original sound,
     provide a PHONETIC SPELLING in the target language(s) ({langs}).
   - Example (for Polish): 'James' -> 'Dżejms', 'MythBusters' -> 'Mif-basters', 'Hyneman' -> 'Hajneman'.
   - This is crucial for the TTS engine to pronounce them correctly.
3. 'speakers': A dictionary where keys are speaker IDs (e.g. SPEAKER_00) and values are objects with:
   - 'name': The inferred real name of the character/person.
   - 'desc': A brief profile including GENDER, age, and VOICE traits. Focus on details useful for selecting a TTS model.

OUTPUT FORMAT:
{{
  "summary": "...",
  "glossary": {{"OriginalTerm": "PhoneticTranslation"}},
  "speakers": {{
    "SPEAKER_00": {{"name": "Name", "desc": "Traits"}}
  }}
}}
<|im_end|>
<|im_start|>user
{overview}<|im_end|>
<|im_start|>assistant
"""

T_ED = """<|im_start|>system
You are a Transcription Corrector. Your task is to fix ASR (speech-to-text) errors in the source text.

CONTEXT GLOSSARY (For name reference only): {glossary}
REFERENCE SUBTITLES: "{subtitles}"

RULES:
1. Identify lines with misheard names or phonetic ASR errors.
2. USE STANDARD ORTHOGRAPHY of the source language. NEVER use phonetic transcriptions (like 'æ', 'ð', 'ʃ') in the output.
3. Verify proper nouns against the Reference Subtitles.
4. Return ONLY a JSON object where keys are Line Indices (integers) and values are the Corrected Source Text.
5. IF A LINE IS CORRECT, DO NOT INCLUDE IT.

OUTPUT FORMAT:
{{
  "12": "Corrected text in original language",
  "45": "Corrected text in original language"
}}
<|im_end|>
<|im_start|>user
{txt}<|im_end|>
<|im_start|>assistant
"""

T_TRANS_SYSTEM = """You are an expert {lang_name} translator adapting a video script for DUBBING.
TASK: Translate the provided list of lines to {lang_name}.

GLOSSARY:
{glossary}

OUTPUT FORMAT (JSON):
{{
  "thought": "Your reasoning about the whole batch",
  "translations": [
    {{ "id": 0, "text": "Translated text" }},
    {{ "id": 1, "text": "Translated text" }}
  ]
}}

RULES:
1. Maintain IDs exactly as provided.
2. Follow the GLOSSARY strictly.
3. Be extremely concise.
"""

T_CRITIC_SYSTEM = """You are a Quality Assurance Critic for AI Dubbing.
TASK: Review and fix the provided translations.

CONTEXT SUMMARY: {summary}
GLOSSARY: {glossary}

OUTPUT FORMAT (JSON):
{{
  "analysis": "Identify errors in the batch",
  "final_translations": [
    {{ "id": 0, "final_text": "Corrected text" }},
    {{ "id": 1, "final_text": "" }}
  ]
}}

RULES:
1. If a translation is perfect, set "final_text" to EMPTY string ("").
2. GLOSSARY IS LAW: Protect phonetic spellings.
3. Ensure IDs match the input.
"""

T_TRANS = """<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{json_input}<|im_end|>
<|im_start|>assistant
"""

T_CRITIC = """<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
Review these translations.<|im_end|>
<|im_start|>assistant
"""

T_SHORTEN = """<|im_start|>system
You are a text compressor. Your task is to shorten the provided text by at least 50%
while preserving the core meaning for dubbing. Respond ONLY with the compressed JSON.

OUTPUT FORMAT:
{{
  "final_text": "The ultra-short version"
}}
<|im_end|>
<|im_start|>user
Original: "{original}"
Text to shorten: "{text}"<|im_end|>
<|im_start|>assistant
"""
