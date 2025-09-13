from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import torch
import re
import logging
from typing import Optional
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

def setup_llm():
    """Setup and return the language model for querying."""
    try:
        # First try to install sentencepiece if missing
        try:
            import sentencepiece
            logger.info("SentencePiece is available")
        except ImportError:
            logger.warning("SentencePiece not found, trying to install...")
            try:
                import subprocess
                import sys
                subprocess.check_call([sys.executable, "-m", "pip", "install", "sentencepiece"])
                logger.info("Successfully installed SentencePiece")
                import sentencepiece  # Try importing again
            except Exception as e:
                logger.warning(f"Could not install SentencePiece: {e}")
        
        # Try different models in order of preference
        model_configs = [
            ("google/flan-t5-small", "Flan T5 Small"),
            ("t5-small", "T5 Small"),
        ]
        
        model_name = None
        tokenizer = None
        model = None
        
        for name, description in model_configs:
            try:
                logger.info(f"Attempting to load {name} - {description}")
                print(f"📥 Loading {name}...")
                
                tokenizer = T5Tokenizer.from_pretrained(name, legacy=False)
                model = T5ForConditionalGeneration.from_pretrained(
                    name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                model_name = name
                print(f"✅ Successfully loaded {name}")
                break
                
            except Exception as e:
                logger.warning(f"Could not load {name}: {e}")
                print(f"❌ Failed to load {name}: {e}")
                continue
        
        if model is None:
            # Try using pipeline as a fallback with better configuration
            try:
                print("📥 Trying pipeline fallback...")
                
                # Try different models for pipeline
                pipeline_models = [
                    "google/flan-t5-small",
                    "t5-small",
                    "distilbert-base-uncased-finetuned-sst-2-english"  # Much smaller fallback
                ]
                
                pipe = None
                for model_name in pipeline_models:
                    try:
                        print(f"   Trying {model_name}...")
                        pipe = pipeline(
                            "text2text-generation" if "t5" in model_name.lower() else "text-generation",
                            model=model_name, 
                            tokenizer=model_name,
                            device=0 if torch.cuda.is_available() else -1
                        )
                        logger.info(f"Successfully loaded {model_name} via pipeline")
                        break
                    except Exception as e:
                        logger.warning(f"Pipeline with {model_name} failed: {e}")
                        continue
                
                if pipe:
                    return {
                        "pipeline": pipe,
                        "model_name": f"{model_name} (pipeline)",
                        "device": "cuda" if torch.cuda.is_available() else "cpu"
                    }
                else:
                    raise Exception("All pipeline attempts failed")
                    
            except Exception as e:
                logger.error(f"Pipeline fallback failed: {e}")
                print("❌ Pipeline fallback failed")
        
        if model is None:
            # Ultimate fallback - return text processing mode
            print("⚠️ Using simple text processing fallback")
            return {
                "fallback": True,
                "model_name": "text-processing-fallback",
                "device": "cpu"
            }
        
        # Move to appropriate device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(model, 'to') and not torch.cuda.is_available():
            model = model.to(device)
        
        logger.info(f"Successfully loaded {model_name} on {device}")
        
        return {
            "model": model,
            "tokenizer": tokenizer,
            "device": device,
            "model_name": model_name
        }
        
    except Exception as e:
        logger.error(f"Error setting up LLM: {e}")
        # Return a simple fallback that can still work
        print("⚠️ Using simple text processing fallback")
        return {
            "fallback": True,
            "model_name": "text-processing-fallback",
            "device": "cpu"
        }

def create_better_prompt(context: str, question: str) -> str:
    """Create a more effective prompt for the T5 model."""
    # Clean context
    context = re.sub(r'\s+', ' ', context).strip()
    context = context[:1000] if len(context) > 1000 else context  # Shorter context for stability
    
    question_lower = question.lower()
    
    # Simpler prompts that work better with smaller models
    if any(word in question_lower for word in ["accuracy", "performance", "score", "result"]):
        prompt = f"Extract performance metrics from: {context} Question: {question} Answer:"
    
    elif any(word in question_lower for word in ["conclusion", "conclude", "summary"]):
        prompt = f"Summarize conclusions from: {context} Question: {question} Answer:"
    
    elif any(word in question_lower for word in ["method", "approach", "technique"]):
        prompt = f"Describe methodology from: {context} Question: {question} Answer:"
    
    else:
        prompt = f"Context: {context} Question: {question} Answer:"
    
    return prompt

def simple_text_analysis(context: str, question: str) -> str:
    """Fallback text analysis when LLM is not available."""
    if not context or not question:
        return "Unable to analyze due to missing information."
    
    question_lower = question.lower()
    context_lower = context.lower()
    
    # Simple keyword-based analysis
    if any(word in question_lower for word in ["accuracy", "performance", "score"]):
        # Look for accuracy patterns
        patterns = [
            r'accuracy[:\s]+(\d+\.?\d*\s*%?)',
            r'(\d+\.?\d*\s*%)\s*accuracy',
            r'achieves?\s+(\d+\.?\d*\s*%?)',
            r'performance[:\s]+(\d+\.?\d*\s*%?)',
            r'f1[:\s]*(\d+\.?\d*)',
            r'precision[:\s]*(\d+\.?\d*)',
            r'recall[:\s]*(\d+\.?\d*)'
        ]
        
        found_metrics = []
        for pattern in patterns:
            matches = re.finditer(pattern, context_lower)
            for match in matches:
                metric = match.group(1)
                found_metrics.append(metric)
        
        if found_metrics:
            return f"Found performance metrics: {', '.join(found_metrics[:3])}"
        else:
            # Look for any numbers that might be performance metrics
            numbers = re.findall(r'\b\d+\.?\d*\s*%\b', context)
            if numbers:
                return f"Potential performance values: {', '.join(numbers[:3])}"
            return "Performance evaluation discussed, but specific metrics not clearly identified."
    
    elif any(word in question_lower for word in ["conclusion", "conclude", "summary"]):
        # Look for conclusion section
        conclusion_indicators = ["conclusion", "summary", "in summary", "we conclude", "findings", "results show"]
        best_section = ""
        
        for indicator in conclusion_indicators:
            idx = context_lower.find(indicator)
            if idx != -1:
                section = context[idx:idx + 250]
                if len(section) > len(best_section):
                    best_section = section
        
        if best_section:
            return f"Key findings: {best_section}"
        else:
            # Return last part of text as likely conclusion
            return f"Summary from document: {context[-200:]}"
    
    elif any(word in question_lower for word in ["method", "approach", "technique"]):
        # Look for methodology keywords
        method_keywords = ["method", "approach", "technique", "algorithm", "model", "framework", "architecture"]
        method_sections = []
        
        for keyword in method_keywords:
            idx = context_lower.find(keyword)
            if idx != -1:
                section = context[max(0, idx-50):idx + 150]
                method_sections.append(section)
        
        if method_sections:
            # Return the longest section
            best_section = max(method_sections, key=len)
            return f"Methodology: {best_section}"
        else:
            return "Methodology section not clearly identified in the provided context."
    
    else:
        # General question - return most relevant part
        question_words = [w.lower() for w in question.split() if len(w) > 3]
        best_section = ""
        max_matches = 0
        
        # Split context into sentences and find most relevant
        sentences = re.split(r'[.!?]+', context)
        for sentence in sentences:
            if len(sentence.strip()) < 20:
                continue
            
            matches = sum(1 for word in question_words if word in sentence.lower())
            if matches > max_matches:
                max_matches = matches
                best_section = sentence.strip()
        
        if best_section:
            return f"Most relevant information: {best_section}"
        else:
            return f"Based on the context: {context[:200]}..."

def query_llm(llm_setup, context: str, question: str, max_length: int = 100) -> Optional[str]:
    """Query the language model with improved error handling and parameter management."""
    if not llm_setup or not context.strip() or not question.strip():
        return "Unable to process the query due to missing information."
    
    # Check if we're using fallback mode
    if llm_setup.get("fallback", False):
        return simple_text_analysis(context, question)
    
    try:
        # Handle pipeline mode
        if "pipeline" in llm_setup:
            pipe = llm_setup["pipeline"]
            prompt = create_better_prompt(context, question)
            
            # Configure generation parameters to avoid conflicts
            generation_kwargs = {
                "max_new_tokens": max_length,  # Use only max_new_tokens
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "num_return_sequences": 1,
                "pad_token_id": pipe.tokenizer.eos_token_id,
            }
            
            # Remove conflicting parameters
            if hasattr(pipe, 'model') and hasattr(pipe.model.config, 'is_encoder_decoder') and pipe.model.config.is_encoder_decoder:
                # For T5 models, we can use max_length instead
                generation_kwargs = {
                    "max_length": max_length + len(pipe.tokenizer.encode(prompt)),
                    "do_sample": True,
                    "temperature": 0.7,
                    "num_return_sequences": 1,
                }
            
            try:
                result = pipe(prompt, **generation_kwargs)
                if result and len(result) > 0:
                    response = result[0].get("generated_text", "")
                    # Clean the response
                    response = post_process_response(response, question, prompt)
                    return response if response else simple_text_analysis(context, question)
                else:
                    return simple_text_analysis(context, question)
            except Exception as e:
                logger.warning(f"Pipeline generation error: {e}")
                return simple_text_analysis(context, question)
        
        # Handle regular model mode
        model = llm_setup["model"]
        tokenizer = llm_setup["tokenizer"]
        device = llm_setup["device"]
        
        # Create prompt
        prompt = create_better_prompt(context, question)
        
        # Tokenize input with proper error handling
        try:
            inputs = tokenizer.encode(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=400,  # Shorter for stability
                padding=True
            )
            inputs = inputs.to(device)
        except Exception as e:
            logger.error(f"Tokenization error: {e}")
            return simple_text_analysis(context, question)
        
        # Generate response
        try:
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=max_length,  # Use max_new_tokens instead of max_length
                    min_length=5,
                    num_beams=2,
                    temperature=0.8,
                    do_sample=True,
                    early_stopping=True,
                    pad_token_id=tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else 1
                )
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return simple_text_analysis(context, question)
        
        # Decode response
        try:
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = post_process_response(response, question, prompt)
            
            if len(response.strip()) < 10:
                return simple_text_analysis(context, question)
            
            return response
        except Exception as e:
            logger.error(f"Decoding error: {e}")
            return simple_text_analysis(context, question)
    
    except Exception as e:
        logger.error(f"Error querying LLM: {e}")
        return simple_text_analysis(context, question)

def post_process_response(response: str, question: str, prompt: str = "") -> str:
    """Post-process the model response for better quality."""
    if not response:
        return ""
    
    # Remove the original prompt from the response if it's there
    if prompt and prompt in response:
        response = response.replace(prompt, "").strip()
    
    # Remove common artifacts and prefixes
    artifacts_to_remove = [
        r'^(Answer:|Response:|Output:)\s*',
        r'^Context:.*?Question:.*?Answer:\s*',
        r'question:.*?context:.*?(?=\w)',
        prompt if prompt else ""
    ]
    
    for artifact in artifacts_to_remove:
        if artifact:
            response = re.sub(artifact, '', response, flags=re.IGNORECASE | re.DOTALL)
    
    # Clean up repetitive text (common issue with small models)
    words = response.split()
    if len(words) > 5:
        # Check for repetitive patterns
        for i in range(len(words) - 2):
            word = words[i]
            if word == words[i + 1] == words[i + 2]:  # Same word repeated 3 times
                # Truncate at the first repetition
                response = " ".join(words[:i])
                break
    
    # Remove incomplete sentences at the end
    sentences = response.split('.')
    if len(sentences) > 1 and len(sentences[-1].strip()) < 5:
        response = '.'.join(sentences[:-1]) + '.'
    
    # Ensure proper capitalization
    if response and response[0].islower():
        response = response[0].upper() + response[1:]
    
    # Remove excessive whitespace
    response = re.sub(r'\s+', ' ', response).strip()
    
    # Limit length to prevent very long responses
    if len(response) > 300:
        response = response[:300].rsplit(' ', 1)[0] + "..."
    
    return response

def extract_metrics(llm_setup, context: str) -> Optional[str]:
    """Extract performance metrics from context."""
    if not context:
        return None
    
    # Enhanced regex patterns for metrics
    patterns = [
        (r'accuracy[:\s=]+(\d+\.?\d*\s*%?)', 'Accuracy'),
        (r'(\d+\.?\d*\s*%)\s+accuracy', 'Accuracy'),
        (r'f1[:\s=]+(\d+\.?\d*)', 'F1 Score'),
        (r'precision[:\s=]+(\d+\.?\d*)', 'Precision'),
        (r'recall[:\s=]+(\d+\.?\d*)', 'Recall'),
        (r'performance[:\s=]+(\d+\.?\d*\s*%?)', 'Performance'),
        (r'achieves?\s+(\d+\.?\d*\s*%?)', 'Achievement'),
        (r'score[:\s=]+(\d+\.?\d*)', 'Score'),
        (r'auc[:\s=]+(\d+\.?\d*)', 'AUC'),
        (r'map[:\s=]+(\d+\.?\d*)', 'mAP')
    ]
    
    found_metrics = []
    context_lower = context.lower()
    
    for pattern, metric_type in patterns:
        matches = re.finditer(pattern, context_lower)
        for match in matches:
            metric_value = match.group(1).strip()
            found_metrics.append(f"{metric_type}: {metric_value}")
    
    if found_metrics:
        return "Performance metrics found: " + "; ".join(found_metrics[:4])
    
    # Fallback to LLM if available
    if llm_setup and not llm_setup.get("fallback", False):
        return query_llm(llm_setup, context, "What are the performance metrics or accuracy values mentioned?", max_length=50)
    
    return "No specific performance metrics found in the provided text."

def summarize_text(llm_setup, text: str, max_length: int = 80) -> Optional[str]:
    """Summarize the given text."""
    if not text:
        return None
    
    # Truncate very long text
    if len(text) > 1200:
        text = text[:1200] + "..."
    
    if not llm_setup or llm_setup.get("fallback", False):
        # Simple extractive summary
        sentences = re.split(r'[.!?]+', text)
        important_sentences = []
        
        # Look for sentences with important keywords
        keywords = ["conclude", "result", "find", "show", "demonstrate", "achieve", "propose", "present"]
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in keywords) and len(sentence.strip()) > 20:
                important_sentences.append(sentence.strip())
        
        if important_sentences:
            return "Summary: " + " ".join(important_sentences[:2])
        else:
            return f"Summary: {text[:150]}..."
    
    return query_llm(llm_setup, text, "Provide a brief summary of this text", max_length)

# Test function
def test_llm_setup():
    """Test the LLM setup and basic functionality."""
    print("Testing LLM setup...")
    llm = setup_llm()
    
    if llm:
        print(f"✅ LLM setup successful: {llm.get('model_name', 'Unknown')}")
        print(f"🔧 Device: {llm.get('device', 'Unknown')}")
        
        # Test query
        test_context = "The model achieved 95% accuracy on the test dataset. The methodology used deep learning with transformer architecture."
        test_question = "What is the accuracy?"
        
        print(f"Test query: {test_question}")
        response = query_llm(llm, test_context, test_question)
        print(f"Response: {response}")
        
        return True
    else:
        print("❌ Failed to set up LLM")
        return False

if __name__ == "__main__":
    test_llm_setup()