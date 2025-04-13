FAQ_DETAIL_EXPANSION_PROMPT_TEMPLATE = """
# TASK: Vietnamese FAQ Detail Expansion

## ROLE
You are an expert in refining and expanding FAQ content. Your task is to analyze a given FAQ pair and extract detailed information to create additional, nuanced FAQ pairs.

---
## GOAL
Your task is to generate up to {max_new_faq_pairs} new FAQ pairs derived from the provided input FAQ pair. Each new FAQ pair should offer more detailed and specific insights related to the original FAQ content. The final output must be in Vietnamese.

---
## GUIDELINES
1. **Analysis of the Input FAQ:**
   * Carefully analyze the provided FAQ pair to identify any additional details, clarifications, or aspects that can be turned into new FAQ pairs.
   * Focus on expanding the details by considering underlying reasons, examples, or related subtopics.
2. **Language Requirement:**
   * The input FAQ pair is in Vietnamese, and your output must be entirely in Vietnamese.
3. **Internal Reasoning Process:**
   * Use a step-by-step internal reasoning process (chain-of-thought) to determine the most relevant additional FAQ pairs. 
   * **Important:** Do not include any internal chain-of-thought or reasoning steps in your final output.
4. **Output Format:**
   * Return the final output in strict JSON format as an array of objects. Each object must include two keys: "question" and "answer". For example:
    [
        {{
            "question": "Question in Vietnamese?",
            "answer": "Answer in Vietnamese."
        }},
        ...
    ]
5. **Relevance and Specificity:**
   * Ensure that each generated FAQ pair adds new, specific details that naturally extend the input FAQ pair.
   * Avoid redundancy and make sure the new FAQ pairs are directly related to the original content.

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
        "question": "Các chương trình đào tạo của trường có đặc điểm nổi bật gì?",
        "answer": "Các chương trình được thiết kế để cung cấp kiến thức chuyên môn sâu và kỹ năng thực tiễn, giúp sinh viên sẵn sàng cho thị trường lao động."
    }},
    {{
        "question": "Trường áp dụng những chiến lược nào để đảm bảo chất lượng đào tạo?",
        "answer": "Trường thường xuyên cải tiến chương trình giảng dạy, hợp tác với các doanh nghiệp và tổ chức nghiên cứu, và tập trung vào ứng dụng thực tiễn trong từng khóa học."
    }}
]

---
## ADDITIONAL INSTRUCTIONS
* Remember to keep your output within the limit of **{max_new_faq_pairs} new FAQ pairs**.
* Do not include any internal chain-of-thought or reasoning steps in your final output.

---------------------
##### REAL DATA #####
---------------------
REAL DATA:
FAQ Pair: {faq_pair}
Output:
"""
