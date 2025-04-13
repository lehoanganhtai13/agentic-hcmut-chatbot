CONTEXT_EXTRACTION_PROMPT_TEMPLATE = """
# TASK: Vietnamese Document Context Extraction

## ROLE
You are a context extraction expert tasked with analyzing a Vietnamese document and producing a comprehensive summary along with a list of key topics/ideas that capture the overall context of the original text.

---
## GOAL
Your task is to extract the main context from the provided Vietnamese text about Trường Đại học Bách Khoa TP.HCM (Ho Chi Minh City University of Technology). You should produce a detailed summary and a list of topics that cover all significant aspects of the document. The final output must be entirely in **Vietnamese** and limited to maximum {max_tokens} tokens.

---
## GUIDELINES
1. **Prioritization of Information:**
    * Focus on extracting critical information that encapsulates the history, mission, academic programs, research initiatives, infrastructure, community engagement, and notable achievements of the university.
2. **Language Requirement:**
    * The input document is in Vietnamese. Your output, including the summary and topics, must be written in Vietnamese.
3. **Internal Reasoning Process:**
    * Use a clear, step-by-step internal reasoning (chain-of-thought) process to identify the key ideas and supporting details from the document.
    * **Note:** Do not include this internal reasoning or any intermediate steps in the final answer.
4. **Output Format:**
    * Your final output should be in strict JSON format following the structure below:
        ```
        {{ "summary": "A comprehensive summary of the document in Vietnamese, capturing the overall context." }}
        ```
    * Do not output any text outside this JSON structure.
5. **Detail and Clarity:**
    * Ensure that both the summary and the list of topics are thorough and reflect all the main points mentioned in the document.
    * Avoid vague or overly generic descriptions; use specific details from the text.
6. **Coherence and Consistency:**
    * The extracted context must be coherent and logically organized to provide a full picture of the document.

---
## EXAMPLE
Document (in Vietnamese): 
"Trường Đại học Bách Khoa TP.HCM là một trong những trường đại học hàng đầu về kỹ thuật và công nghệ tại Việt Nam. Trường nổi tiếng với các chương trình đào tạo chất lượng, các nghiên cứu tiên tiến và sự hợp tác với các doanh nghiệp hàng đầu."

Expected Output:
{{
    "summary": "Trường Đại học Bách Khoa TP.HCM là một trường đại học hàng đầu về kỹ thuật và công nghệ, nổi tiếng với chương trình đào tạo chất lượng, nghiên cứu tiên tiến và hợp tác doanh nghiệp."
}}

---
## ADDITIONAL INSTRUCTIONS
- Remember to keep your summary under {max_tokens} tokens.
- Do not include any internal chain-of-thought or reasoning steps in your final answer.
- Ensure that your final output is strictly in JSON format as specified above, without any additional text.

---------------------
##### REAL DATA #####
---------------------
Text:
```
{text}
```

Output:
"""
