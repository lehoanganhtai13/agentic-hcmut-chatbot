FAQ_PARAPHRASE_PROMPT_TEMPLATE = """
# Task: Paraphrase the FAQ question

## ROLE
You are an expert in paraphrasing FAQ questions, specializing in rewording queries in ways that reflect how Vietnamese students would naturally ask about school-related information.

---
## GOAL
Your task is to generate up to {max_paraphrases} different paraphrased versions of the original FAQ question from the provided FAQ pair. Ensure that all new questions maintain the original intent so that the associated answer remains valid. The final output must be written in Vietnamese.

---
## GUIDELINES
1. **Student-Friendly Language:**
   - Rephrase the original question using language, style, and vocabulary typical of Vietnamese students when inquiring about school details.
   - Ensure that the paraphrased questions maintain the same meaning as the original question.
2. **Consistency with the Original Answer:**
   - Generate paraphrased questions that are directly related to the original FAQ, ensuring that the initial answer still applies without any change.
3. **Language Requirement:**
   - Both the input and output must be in Vietnamese.
4. **Internal Reasoning Process:**
   - Utilize a clear, step-by-step internal reasoning process (chain-of-thought) to generate varied yet faithful paraphrases.
   - **Important:** Do not include any internal chain-of-thought or reasoning steps in your final output.
5. **Output Format:**
   - Return your final output in strict JSON format as an array of objects. Each object must include a key "paraphrased_question". For example:
    [
        {{
            "paraphrased_question": "..."
        }},
        ...
    ]
6. **Quality and Variety:**
   - Ensure that the paraphrases offer diverse phrasings and sentence structures while remaining true to the original meaning.
   - Avoid redundancy or overly generic formulations.

---
## EXAMPLE
Input FAQ Pair (in Vietnamese):
{{
    "question": "Trường Đại học Bách Khoa TP.HCM cung cấp loại chương trình đào tạo nào?",
    "answer": "Trường cung cấp các chương trình đào tạo chất lượng cao."
}}

Expected Output:
[
    {{
        "paraphrased_question": "Trường Đại học Bách Khoa TP.HCM có những chương trình đào tạo đặc sắc gì?"
    }},
    {{
        "paraphrased_question": "Những loại chương trình đào tạo mà Trường Đại học Bách Khoa TP.HCM cung cấp là gì?"
    }},
    {{
        "paraphrased_question": "Sinh viên có thể lựa chọn những chương trình đào tạo nào tại Trường Đại học Bách Khoa TP.HCM?"
    }}
]

---
## ADDITIONAL INSTRUCTIONS
- Keep your output within the limit of **{max_paraphrases} paraphrased questions**.
- Do not include any internal chain-of-thought or reasoning steps in your final output.

---------------------
##### REAL DATA #####
---------------------
REAL DATA:
FAQ Pair: {faq_pair}
Output:
"""
