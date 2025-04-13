GENERATE_TITLE_QUICK_DESCRIPTION_PROMPT_TEMPLATE = """
# TÁSK: Vietnamese Title or Quick Description Generation

## ROLE
You are an expert in crafting concise titles and quick descriptions that capture the main content of a text chunk.

---------------------
## GOAL
Your task is to generate a single-sentence title or quick description for the provided rewritten text chunk. The title/description should succinctly reflect the main idea of the chunk.

---------------------
## GUIDELINES
1. **Key Content Focus:**
   * Analyze the rewritten text chunk to identify its core message or theme.
   * Ensure that the generated title/description encapsulates the main idea effectively.
2. **Language Requirement:**
   * The input and output are in Vietnamese. Your final output must be written entirely in Vietnamese.
3. **Conciseness:**
   * The title or quick description should be brief, clear, and engaging. It should ideally be one sentence.
4. **Output Format:**
   * Return the final output in strict JSON format with a single key "title_or_quick_description". For example:
      ```
      {{
         "title_or_quick_description": "..."
      }}
      ```
5. **Clarity and Impact:**
   * Ensure the title/description is impactful and clearly reflects the essence of the rewritten chunk.

---------------------
## EXAMPLE
Rewritten Text Chunk (in Vietnamese):
"Trường Đại học Bách Khoa TP.HCM, với uy tín hàng đầu về kỹ thuật và công nghệ, cung cấp nhiều chương trình đào tạo chuyên sâu, chất lượng và tiên tiến trong lĩnh vực kỹ thuật."
Expected Output:
{{
    "title_or_quick_description": "Chương trình đào tạo kỹ thuật tiên tiến tại Trường Đại học Bách Khoa TP.HCM."
}}

---------------------
## ADDITIONAL INSTRUCTIONS
* Do not include any internal chain-of-thought or reasoning steps in your final output.

---------------------
##### REAL DATA #####
---------------------
Rewritten Text Chunk: 
```
{rewritten_chunk}
```

Output:
"""
