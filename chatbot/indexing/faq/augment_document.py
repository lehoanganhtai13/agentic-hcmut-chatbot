import json
import re
import uuid
from loguru import logger
from typing import List
from tqdm.auto import tqdm

from chatbot.core.model_clients import BaseLLM
from chatbot.indexing.faq.base_class import FAQDocument
from chatbot.prompts.indexing.paraphrase_faq import FAQ_PARAPHRASE_PROMPT_TEMPLATE


class FaqAugmenter:
    """
    A class that augments FAQ documents by generating additional FAQs based on existing ones.
    
    This class processes FAQ documents to create new FAQ pairs by paraphrasing or rephrasing
    existing questions, enhancing the diversity and coverage of the FAQ content.
    
    Attributes:
        llm (BaseLLM): The language model used for FAQ augmentation.
    
    Methods:
        augment_faq: Process existing FAQ documents to create augmented FAQ pairs.
    
    Example:
        >>> llm = BaseLLM()
        >>> faq_documents = [FAQDocument(question="What is AI?", answer="AI is artificial intelligence.")]
        >>> augmenter = FaqAugmenter(llm)
        >>> augmented_faqs = augmenter.augment_faq(faq_documents, max_pairs=3)
    """
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def augment_faq(
        self,
        documents: List[FAQDocument],
        max_pairs: int = 3
    ) -> List[FAQDocument]:
        """
        Augment FAQ documents by generating additional FAQ pairs.
        
        Args:
            documents (List[FAQDocument]): List of existing FAQ documents.
            max_pairs (int): Maximum number of FAQ pairs to generate for each document.
        
        Returns:
            List[FAQDocument]: List of augmented FAQ documents.
        """
        augmented_faqs: List[FAQDocument] = []
        progress_bar = tqdm(documents, desc="Augmenting FAQ pairs")
        
        # Iterate through each FAQ document
        for document in documents:
            try:
                # Generate new Questions based on the existing the original FAQ pair
                response = self.llm.complete(
                    prompt=FAQ_PARAPHRASE_PROMPT_TEMPLATE.format(
                        faq_pair={"question": document.question, "answer": document.answer},
                        max_paraphrases=max_pairs
                    )
                ).text
                
                # Extract paraphrased questions using robust parsing
                paraphrased_questions = self._extract_paraphrased_questions(response)
                
                if not paraphrased_questions:
                    logger.warning(f"No valid paraphrased questions found for document {document.id}")
                    continue
                
                # Add new FAQ pairs to the list
                for question_text in paraphrased_questions:
                    if question_text and question_text.strip():
                        augmented_faqs.append(FAQDocument(
                            id=str(uuid.uuid4()),
                            question=question_text.strip(),
                            answer=document.answer
                        ))
                # Update progress bar
                progress_bar.update(1)
                
            except Exception as e:
                logger.error(f"Error processing document {document.id}: {response}")
                continue
            
        # Close the progress bar
        progress_bar.close()
        
        return augmented_faqs
    
    def _extract_paraphrased_questions(self, response: str) -> List[str]:
        """
        Extract paraphrased questions from LLM response using multiple robust strategies.
        
        This method employs multiple parsing strategies to handle various LLM output formats:
        - Clean JSON arrays
        - Malformed JSON with missing braces
        - Mixed object/string formats
        - Code block wrapped content
        
        Args:
            response (str): Raw response text from LLM
            
        Returns:
            List[str]: List of extracted paraphrased questions
        """
        if not response or not response.strip():
            return []
        
        # Strategy 1: Parse clean JSON array
        clean_questions = self._parse_clean_json_array(response)
        if clean_questions:
            logger.info(f"✅ Extracted {len(clean_questions)} questions from clean JSON")
            return clean_questions
        
        # Strategy 2: Fix and parse malformed JSON
        fixed_questions = self._parse_malformed_json(response)
        if fixed_questions:
            logger.info(f"🔧 Extracted {len(fixed_questions)} questions from fixed JSON")
            return fixed_questions
        
        # Strategy 3: Extract from mixed format (objects + strings)
        mixed_questions = self._parse_mixed_format(response)
        if mixed_questions:
            logger.info(f"🔨 Extracted {len(mixed_questions)} questions from mixed format")
            return mixed_questions
        
        # Strategy 4: Extract individual question objects
        individual_questions = self._parse_individual_objects(response)
        if individual_questions:
            logger.info(f"⚡ Extracted {len(individual_questions)} questions from individual objects")
            return individual_questions
        
        # Strategy 5: Fallback - extract Vietnamese question patterns
        fallback_questions = self._parse_question_patterns(response)
        if fallback_questions:
            logger.info(f"🎯 Extracted {len(fallback_questions)} questions from pattern matching")
            return fallback_questions
        
        logger.warning("No paraphrased questions could be extracted from response")
        return []
    
    def _parse_clean_json_array(self, response: str) -> List[str]:
        """Parse clean JSON array format."""
        try:
            # Remove code blocks if present
            cleaned = re.sub(r'```(?:json)?\s*', '', response).strip()
            cleaned = re.sub(r'```\s*$', '', cleaned).strip()
            
            # Try to parse as JSON array
            data = json.loads(cleaned)
            if isinstance(data, list):
                questions = []
                for item in data:
                    if isinstance(item, dict) and "paraphrased_question" in item:
                        questions.append(item["paraphrased_question"])
                    elif isinstance(item, str):
                        questions.append(item)
                return questions
        except (json.JSONDecodeError, TypeError):
            pass
        return []
    
    def _parse_malformed_json(self, response: str) -> List[str]:
        """Fix and parse malformed JSON."""
        try:
            # Remove code blocks
            cleaned = re.sub(r'```(?:json)?\s*', '', response).strip()
            cleaned = re.sub(r'```\s*$', '', cleaned).strip()
            
            # Common JSON fixes
            # Fix trailing commas
            cleaned = re.sub(r',\s*}', '}', cleaned)
            cleaned = re.sub(r',\s*]', ']', cleaned)
            
            # Fix missing braces - find standalone question strings
            lines = cleaned.split('\n')
            fixed_lines = []
            
            for line in lines:
                line = line.strip()
                if line.startswith('"paraphrased_question":'):
                    # This is a standalone key-value pair, wrap in braces
                    if not line.endswith(','):
                        line += ','
                    fixed_lines.append('    {' + line + '}')
                elif line.startswith('{') or line.startswith('[') or line.startswith('}') or line.startswith(']'):
                    fixed_lines.append(line)
                elif line and not line.startswith('"paraphrased_question"'):
                    # Might be a loose question string
                    if '"' in line:
                        # Try to wrap as proper object
                        question_match = re.search(r'"([^"]+)"', line)
                        if question_match:
                            fixed_lines.append('    {"paraphrased_question": "' + question_match.group(1) + '"}')
            
            if fixed_lines:
                # Ensure proper array structure
                if not any(line.strip().startswith('[') for line in fixed_lines):
                    fixed_lines.insert(0, '[')
                if not any(line.strip().startswith(']') for line in fixed_lines):
                    fixed_lines.append(']')
                
                fixed_json = '\n'.join(fixed_lines)
                # Remove trailing commas before closing braces/brackets
                fixed_json = re.sub(r',\s*}', '}', fixed_json)
                fixed_json = re.sub(r',\s*]', ']', fixed_json)
                
                data = json.loads(fixed_json)
                if isinstance(data, list):
                    questions = []
                    for item in data:
                        if isinstance(item, dict) and "paraphrased_question" in item:
                            questions.append(item["paraphrased_question"])
                    return questions
        except (json.JSONDecodeError, TypeError):
            pass
        return []
    
    def _parse_mixed_format(self, response: str) -> List[str]:
        """Parse mixed format with both objects and strings."""
        questions = []
        
        # Find all JSON objects with paraphrased_question
        object_pattern = r'\{\s*"paraphrased_question"\s*:\s*"([^"]+(?:\\.[^"]*)*)"[^}]*\}'
        object_matches = re.findall(object_pattern, response, re.DOTALL)
        
        for match in object_matches:
            # Unescape JSON string
            question = match.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
            questions.append(question)
        
        # Find standalone question strings that might be loose
        # Look for quoted strings that look like questions
        if len(questions) == 0:
            standalone_pattern = r'"([^"]*\?[^"]*)"'
            standalone_matches = re.findall(standalone_pattern, response)
            
            for match in standalone_matches:
                if len(match) > 10:  # Reasonable question length
                    questions.append(match)
        
        return questions
    
    def _parse_individual_objects(self, response: str) -> List[str]:
        """Parse individual JSON objects scattered in response."""
        questions = []
        
        # Find all patterns that might be question objects
        patterns = [
            r'"paraphrased_question"\s*:\s*"([^"]+(?:\\.[^"]*)*)"',
            r'paraphrased_question["\']?\s*:\s*["\']([^"\'}\n]+)["\']?'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE | re.DOTALL)
            for match in matches:
                # Clean up the match
                question = match.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
                question = question.strip().strip('"\'')
                if question and len(question) > 5:
                    questions.append(question)
        
        return list(set(questions))  # Remove duplicates
    
    def _parse_question_patterns(self, response: str) -> List[str]:
        """Fallback: Extract Vietnamese question patterns."""
        questions = []
        
        # Vietnamese question indicators
        question_indicators = [
            r'.*\?$',  # Ends with question mark
            r'^(?:Làm thế nào|Tại sao|Khi nào|Ở đâu|Như thế nào|Có phải|Có thể|Điều gì|Việc gì|Ai|Gì)',
            r'.*(?:không|chưa|được không|có được|như thế nào|thế nào|gì|ai|đâu|nào|sao)\?*$'
        ]
        
        # Split response into lines and sentences
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            # Skip JSON-like lines
            if line.startswith('{') or line.startswith('[') or line.startswith('"paraphrased_question"'):
                continue
            
            # Remove quotes and clean
            line = line.strip('"\'')
            
            # Check if it matches question patterns
            for pattern in question_indicators:
                if re.match(pattern, line, re.IGNORECASE):
                    if len(line) > 10:  # Reasonable question length
                        questions.append(line)
                        break
        
        return list(set(questions))  # Remove duplicates
