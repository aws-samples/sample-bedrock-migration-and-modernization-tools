"""
Recommendation engine for the Amazon Bedrock Expert application.
"""
from typing import Dict, List, Any, Optional, Set, Tuple
import pandas as pd
import re


def analyze_use_case(description: str) -> Dict[str, Any]:
    """
    Analyze a use case description to extract requirements.
    
    Args:
        description: User's description of their use case
        
    Returns:
        Dictionary with extracted requirements
    """
    # Initialize requirements dictionary
    requirements = {
        'capabilities': [],
        'languages': [],
        'modalities': [],
        'context_length': 'medium',
        'price_sensitivity': 'medium',
        'streaming': False,
        'regions': [],
        'use_cases': []
    }
    
    # Extract capabilities
    if re.search(r'chat|conversation|dialog', description, re.IGNORECASE):
        requirements['capabilities'].append('text-generation')
        requirements['capabilities'].append('chat')
    
    if re.search(r'image|picture|photo|visual|generate image', description, re.IGNORECASE):
        requirements['capabilities'].append('image-generation')
    
    if re.search(r'embed|embedding|vector|semantic search', description, re.IGNORECASE):
        requirements['capabilities'].append('embeddings')
    
    if re.search(r'multimodal|multi-modal|text and image|image and text', description, re.IGNORECASE):
        requirements['capabilities'].append('multimodal')
    
    # Extract languages
    languages = {
        'english': 'English',
        'spanish': 'Spanish',
        'french': 'French',
        'german': 'German',
        'italian': 'Italian',
        'portuguese': 'Portuguese',
        'dutch': 'Dutch',
        'russian': 'Russian',
        'chinese': 'Chinese',
        'japanese': 'Japanese',
        'korean': 'Korean',
        'arabic': 'Arabic',
        'hindi': 'Hindi',
        'multilingual': 'Multilingual'
    }
    
    for lang_key, lang_name in languages.items():
        if re.search(rf'\b{lang_key}\b', description, re.IGNORECASE):
            requirements['languages'].append(lang_name)
    
    if re.search(r'multilingual|multiple languages|many languages', description, re.IGNORECASE):
        requirements['languages'].append('Multilingual')
    
    # Extract modalities
    if re.search(r'text|write|writing|content', description, re.IGNORECASE):
        requirements['modalities'].append('text')
    
    if re.search(r'image|picture|photo|visual', description, re.IGNORECASE):
        requirements['modalities'].append('image')
    
    if re.search(r'audio|sound|voice', description, re.IGNORECASE):
        requirements['modalities'].append('audio')
    
    if re.search(r'video|movie|clip', description, re.IGNORECASE):
        requirements['modalities'].append('video')
    
    # Extract context length requirements
    if re.search(r'long context|large context|big context|many pages|multiple documents', description, re.IGNORECASE):
        requirements['context_length'] = 'high'
    elif re.search(r'short context|small context|brief', description, re.IGNORECASE):
        requirements['context_length'] = 'low'
    
    # Extract price sensitivity
    if re.search(r'cheap|inexpensive|low cost|budget|affordable', description, re.IGNORECASE):
        requirements['price_sensitivity'] = 'high'
    elif re.search(r'expensive|high quality|premium|best|top', description, re.IGNORECASE):
        requirements['price_sensitivity'] = 'low'
    
    # Extract streaming requirement
    if re.search(r'stream|streaming|real-time|realtime|interactive', description, re.IGNORECASE):
        requirements['streaming'] = True
    
    # Extract regions
    regions = {
        'us': ['us-east-1', 'us-east-2', 'us-west-1', 'us-west-2'],
        'europe': ['eu-central-1', 'eu-west-1', 'eu-west-2', 'eu-west-3', 'eu-north-1'],
        'asia': ['ap-northeast-1', 'ap-northeast-2', 'ap-southeast-1', 'ap-southeast-2'],
        'canada': ['ca-central-1'],
        'south america': ['sa-east-1']
    }
    
    for region_key, region_codes in regions.items():
        if re.search(rf'\b{region_key}\b', description, re.IGNORECASE):
            requirements['regions'].extend(region_codes)
    
    # Extract use cases
    use_cases = {
        'chatbot': ['chatbot', 'chat bot', 'conversational ai'],
        'content-generation': ['content generation', 'content creation', 'writing', 'blog', 'article'],
        'summarization': ['summary', 'summarize', 'summarization'],
        'translation': ['translate', 'translation', 'language conversion'],
        'question-answering': ['question answering', 'q&a', 'question and answer'],
        'code-generation': ['code', 'programming', 'software development'],
        'image-generation': ['image generation', 'create image', 'generate image']
    }
    
    for use_case, keywords in use_cases.items():
        for keyword in keywords:
            if re.search(rf'\b{keyword}\b', description, re.IGNORECASE):
                requirements['use_cases'].append(use_case)
                break
    
    return requirements


def generate_follow_up_questions(requirements: Dict[str, Any]) -> List[str]:
    """
    Generate follow-up questions based on the extracted requirements.
    
    Args:
        requirements: Dictionary with extracted requirements
        
    Returns:
        List of follow-up questions
    """
    questions = []
    
    # Check for missing information
    if not requirements['capabilities']:
        questions.append("What specific capabilities are you looking for in a model? (e.g., text generation, image generation, embeddings)")
    
    if not requirements['languages']:
        questions.append("Which languages do you need the model to support?")
    
    if not requirements['modalities']:
        questions.append("What types of data will you be working with? (e.g., text, images, audio)")
    
    if not requirements['use_cases']:
        questions.append("What specific task or use case are you trying to solve?")
    
    if not requirements['regions']:
        questions.append("Are there specific AWS regions where you need the model to be available?")
    
    # Ask for clarification on ambiguous requirements
    if len(requirements['capabilities']) > 3:
        questions.append("You mentioned multiple capabilities. Which ones are most important for your use case?")
    
    if requirements['context_length'] == 'medium':
        questions.append("How long are the texts you'll be processing? Do you need to handle large documents or just short messages?")
    
    if requirements['price_sensitivity'] == 'medium':
        questions.append("How important is cost versus performance for your use case?")
    
    return questions


def extract_keywords(description: str) -> Set[str]:
    """
    Extract keywords from a use case description.
    
    Args:
        description: User's description of their use case
        
    Returns:
        Set of keywords
    """
    # List of common keywords to look for
    keyword_patterns = [
        r'chat(bot)?', r'conversation(al)?', r'dialog(ue)?',
        r'image( generation)?', r'picture', r'photo',
        r'embed(ding)?', r'vector', r'semantic search',
        r'multimodal', r'multi-modal',
        r'language', r'multilingual', r'translation',
        r'summariz(e|ation)', r'summary',
        r'question( answer(ing)?)?', r'q&a',
        r'code( generation)?', r'programming',
        r'content( generation)?', r'writing', r'blog', r'article',
        r'streaming', r'real-time', r'interactive',
        r'context( window)?', r'token',
        r'cost', r'price', r'budget', r'affordable', r'expensive',
        r'region', r'location', r'availability',
        r'performance', r'quality', r'accuracy',
        r'speed', r'latency', r'response time'
    ]
    
    # Extract keywords
    keywords = set()
    for pattern in keyword_patterns:
        matches = re.finditer(rf'\b{pattern}\b', description, re.IGNORECASE)
        for match in matches:
            keywords.add(match.group(0).lower())
    
    return keywords


def detect_intent(description: str) -> Tuple[str, float]:
    """
    Detect the primary intent of the user's query.
    
    Args:
        description: User's description of their use case
        
    Returns:
        Tuple of (intent, confidence)
    """
    # Define intent patterns
    intent_patterns = {
        'model_recommendation': [
            r'recommend', r'suggest', r'which model', r'best model',
            r'looking for a model', r'need a model', r'find a model',
            r'help me choose', r'select a model'
        ],
        'price_comparison': [
            r'price', r'cost', r'expensive', r'cheap', r'budget',
            r'affordable', r'pricing', r'compare prices'
        ],
        'capability_inquiry': [
            r'can it', r'is it able', r'capability', r'feature',
            r'support', r'handle', r'able to', r'what can'
        ],
        'technical_question': [
            r'how does', r'how to', r'technical', r'implementation',
            r'integrate', r'api', r'code', r'setup', r'configure'
        ],
        'general_information': [
            r'what is', r'tell me about', r'explain', r'information',
            r'details', r'overview', r'describe'
        ]
    }
    
    # Count matches for each intent
    intent_scores = {intent: 0 for intent in intent_patterns}
    
    for intent, patterns in intent_patterns.items():
        for pattern in patterns:
            matches = re.finditer(rf'\b{pattern}\b', description, re.IGNORECASE)
            for _ in matches:
                intent_scores[intent] += 1
    
    # Find the intent with the highest score
    max_score = max(intent_scores.values())
    if max_score == 0:
        return ('general_information', 0.5)  # Default intent with low confidence
    
    # Get the intent with the highest score
    max_intent = max(intent_scores, key=intent_scores.get)
    
    # Calculate confidence (normalized score)
    total_score = sum(intent_scores.values())
    confidence = intent_scores[max_intent] / total_score
    
    return (max_intent, confidence)


def recommend_models(requirements: Dict[str, Any], models_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Recommend models based on the extracted requirements.
    
    Args:
        requirements: Dictionary with extracted requirements
        models_df: DataFrame containing model data
        
    Returns:
        List of recommended models with scores and explanations
    """
    if models_df.empty:
        return []
    
    # Create a copy of the dataframe for scoring
    df = models_df.copy()
    
    # Initialize score column
    df['score'] = 0
    df['match_reasons'] = df.apply(lambda x: [], axis=1)
    df['mismatch_reasons'] = df.apply(lambda x: [], axis=1)
    
    # Score models based on capabilities
    if requirements['capabilities']:
        for _, model in df.iterrows():
            capability_matches = []
            for capability in requirements['capabilities']:
                if capability in model['capabilities']:
                    df.loc[df['model_id'] == model['model_id'], 'score'] += 10
                    capability_matches.append(capability)
            
            if capability_matches:
                df.loc[df['model_id'] == model['model_id'], 'match_reasons'].apply(
                    lambda x: x.append(f"Supports required capabilities: {', '.join(capability_matches)}")
                )
            else:
                df.loc[df['model_id'] == model['model_id'], 'mismatch_reasons'].apply(
                    lambda x: x.append(f"Missing required capabilities: {', '.join(requirements['capabilities'])}")
                )
    
    # Score models based on languages
    if requirements['languages']:
        for _, model in df.iterrows():
            language_matches = []
            for language in requirements['languages']:
                if language in model['languages'] or 'Multilingual' in model['languages']:
                    df.loc[df['model_id'] == model['model_id'], 'score'] += 5
                    language_matches.append(language)
            
            if language_matches:
                df.loc[df['model_id'] == model['model_id'], 'match_reasons'].apply(
                    lambda x: x.append(f"Supports required languages: {', '.join(language_matches)}")
                )
            else:
                df.loc[df['model_id'] == model['model_id'], 'mismatch_reasons'].apply(
                    lambda x: x.append(f"May not support required languages: {', '.join(requirements['languages'])}")
                )
    
    # Score models based on modalities
    if requirements['modalities']:
        for _, model in df.iterrows():
            modality_matches = []
            for modality in requirements['modalities']:
                if modality in model['input_modalities'] or modality in model['output_modalities']:
                    df.loc[df['model_id'] == model['model_id'], 'score'] += 8
                    modality_matches.append(modality)
            
            if modality_matches:
                df.loc[df['model_id'] == model['model_id'], 'match_reasons'].apply(
                    lambda x: x.append(f"Supports required modalities: {', '.join(modality_matches)}")
                )
            else:
                df.loc[df['model_id'] == model['model_id'], 'mismatch_reasons'].apply(
                    lambda x: x.append(f"Missing required modalities: {', '.join(requirements['modalities'])}")
                )
    
    # Score models based on context length
    if requirements['context_length'] == 'high':
        for _, model in df.iterrows():
            if model['context_window'] >= 100000:
                df.loc[df['model_id'] == model['model_id'], 'score'] += 15
                df.loc[df['model_id'] == model['model_id'], 'match_reasons'].apply(
                    lambda x: x.append(f"Large context window: {model['context_window']:,} tokens")
                )
            elif model['context_window'] >= 32000:
                df.loc[df['model_id'] == model['model_id'], 'score'] += 10
                df.loc[df['model_id'] == model['model_id'], 'match_reasons'].apply(
                    lambda x: x.append(f"Good context window: {model['context_window']:,} tokens")
                )
            else:
                df.loc[df['model_id'] == model['model_id'], 'mismatch_reasons'].apply(
                    lambda x: x.append(f"Limited context window: {model['context_window']:,} tokens")
                )
    elif requirements['context_length'] == 'low':
        for _, model in df.iterrows():
            if model['context_window'] <= 16000:
                df.loc[df['model_id'] == model['model_id'], 'score'] += 5
                df.loc[df['model_id'] == model['model_id'], 'match_reasons'].apply(
                    lambda x: x.append(f"Efficient context window: {model['context_window']:,} tokens")
                )
    
    # Score models based on price sensitivity
    if requirements['price_sensitivity'] == 'high':
        # Extract pricing information
        df['input_price'] = df['pricing'].apply(
            lambda x: x.get('on_demand', {}).get('input_tokens', 0) if isinstance(x, dict) else 0
        )
        
        # Find median price
        median_price = df['input_price'].median()
        
        for _, model in df.iterrows():
            if model['input_price'] <= median_price / 2:
                df.loc[df['model_id'] == model['model_id'], 'score'] += 15
                df.loc[df['model_id'] == model['model_id'], 'match_reasons'].apply(
                    lambda x: x.append(f"Low cost option")
                )
            elif model['input_price'] <= median_price:
                df.loc[df['model_id'] == model['model_id'], 'score'] += 10
                df.loc[df['model_id'] == model['model_id'], 'match_reasons'].apply(
                    lambda x: x.append(f"Moderate cost option")
                )
            else:
                df.loc[df['model_id'] == model['model_id'], 'mismatch_reasons'].apply(
                    lambda x: x.append(f"Higher cost option")
                )
    elif requirements['price_sensitivity'] == 'low':
        # Extract pricing information
        df['input_price'] = df['pricing'].apply(
            lambda x: x.get('on_demand', {}).get('input_tokens', 0) if isinstance(x, dict) else 0
        )
        
        # Find median price
        median_price = df['input_price'].median()
        
        for _, model in df.iterrows():
            if model['input_price'] >= median_price:
                df.loc[df['model_id'] == model['model_id'], 'score'] += 5
                df.loc[df['model_id'] == model['model_id'], 'match_reasons'].apply(
                    lambda x: x.append(f"Premium option with potentially better quality")
                )
    
    # Score models based on streaming requirement
    if requirements['streaming']:
        for _, model in df.iterrows():
            if model['streaming_supported']:
                df.loc[df['model_id'] == model['model_id'], 'score'] += 10
                df.loc[df['model_id'] == model['model_id'], 'match_reasons'].apply(
                    lambda x: x.append(f"Supports streaming")
                )
            else:
                df.loc[df['model_id'] == model['model_id'], 'mismatch_reasons'].apply(
                    lambda x: x.append(f"Does not support streaming")
                )
    
    # Score models based on regions
    if requirements['regions']:
        for _, model in df.iterrows():
            region_matches = []
            for region in requirements['regions']:
                if region in model['regions']:
                    df.loc[df['model_id'] == model['model_id'], 'score'] += 3
                    region_matches.append(region)
            
            if region_matches:
                df.loc[df['model_id'] == model['model_id'], 'match_reasons'].apply(
                    lambda x: x.append(f"Available in required regions: {', '.join(region_matches)}")
                )
            else:
                df.loc[df['model_id'] == model['model_id'], 'mismatch_reasons'].apply(
                    lambda x: x.append(f"Not available in required regions: {', '.join(requirements['regions'])}")
                )
    
    # Score models based on use cases
    if requirements['use_cases']:
        for _, model in df.iterrows():
            use_case_matches = []
            for use_case in requirements['use_cases']:
                if use_case in model['use_cases']:
                    df.loc[df['model_id'] == model['model_id'], 'score'] += 8
                    use_case_matches.append(use_case)
            
            if use_case_matches:
                df.loc[df['model_id'] == model['model_id'], 'match_reasons'].apply(
                    lambda x: x.append(f"Suitable for use cases: {', '.join(use_case_matches)}")
                )
    
    # Sort models by score in descending order
    df = df.sort_values(by='score', ascending=False)
    
    # Get top 5 recommendations
    top_recommendations = df.head(5)
    
    # Format recommendations
    recommendations = []
    for _, model in top_recommendations.iterrows():
        recommendations.append({
            'model_id': model['model_id'],
            'name': model['name'],
            'provider': model['provider'],
            'score': model['score'],
            'match_reasons': model['match_reasons'],
            'mismatch_reasons': model['mismatch_reasons']
        })
    
    return recommendations


def explain_recommendation(model: Dict[str, Any], requirements: Dict[str, Any]) -> str:
    """
    Generate an explanation for a model recommendation.
    
    Args:
        model: Model data dictionary
        requirements: Dictionary with extracted requirements
        
    Returns:
        Explanation string
    """
    explanation = f"**{model['name']}** by {model['provider']} is recommended because:\n\n"
    
    # Add match reasons
    if model['match_reasons']:
        explanation += "**Strengths:**\n"
        for reason in model['match_reasons']:
            explanation += f"- {reason}\n"
        explanation += "\n"
    
    # Add mismatch reasons
    if model['mismatch_reasons']:
        explanation += "**Considerations:**\n"
        for reason in model['mismatch_reasons']:
            explanation += f"- {reason}\n"
        explanation += "\n"
    
    # Add general information
    explanation += f"**Additional Information:**\n"
    explanation += f"- Context Window: {model.get('context_window', 'Unknown'):,} tokens\n"
    explanation += f"- Pricing: Input tokens at ${model.get('input_price', 0):.5f} per 1K tokens\n"
    
    return explanation


def suggest_third_party_models(requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Suggest third-party models when Bedrock models don't fit the requirements.
    
    Args:
        requirements: Dictionary with extracted requirements
        
    Returns:
        List of third-party model suggestions
    """
    third_party_models = [
        {
            'name': 'GPT-4',
            'provider': 'OpenAI',
            'capabilities': ['text-generation', 'chat', 'multimodal'],
            'languages': ['Multilingual'],
            'context_window': 128000,
            'pricing': {'input': 0.01, 'output': 0.03},
            'use_cases': ['chatbot', 'content-generation', 'summarization', 'translation', 'question-answering', 'code-generation'],
            'description': 'OpenAI\'s most advanced model with multimodal capabilities and a large context window.'
        },
        {
            'name': 'GPT-3.5 Turbo',
            'provider': 'OpenAI',
            'capabilities': ['text-generation', 'chat'],
            'languages': ['Multilingual'],
            'context_window': 16000,
            'pricing': {'input': 0.0005, 'output': 0.0015},
            'use_cases': ['chatbot', 'content-generation', 'summarization', 'translation', 'question-answering', 'code-generation'],
            'description': 'OpenAI\'s cost-effective model with good performance for most text generation tasks.'
        },
        {
            'name': 'DALL-E 3',
            'provider': 'OpenAI',
            'capabilities': ['image-generation'],
            'languages': ['English'],
            'context_window': 4000,
            'pricing': {'per_image': 0.04},
            'use_cases': ['image-generation'],
            'description': 'OpenAI\'s advanced image generation model with high-quality outputs.'
        },
        {
            'name': 'Gemini Pro',
            'provider': 'Google',
            'capabilities': ['text-generation', 'chat', 'multimodal'],
            'languages': ['Multilingual'],
            'context_window': 32000,
            'pricing': {'input': 0.0005, 'output': 0.0015},
            'use_cases': ['chatbot', 'content-generation', 'summarization', 'translation', 'question-answering', 'code-generation'],
            'description': 'Google\'s multimodal model with strong performance across various tasks.'
        },
        {
            'name': 'Gemini Ultra',
            'provider': 'Google',
            'capabilities': ['text-generation', 'chat', 'multimodal'],
            'languages': ['Multilingual'],
            'context_window': 32000,
            'pricing': {'input': 0.01, 'output': 0.03},
            'use_cases': ['chatbot', 'content-generation', 'summarization', 'translation', 'question-answering', 'code-generation'],
            'description': 'Google\'s most advanced model with state-of-the-art performance.'
        },
        {
            'name': 'Midjourney',
            'provider': 'Midjourney',
            'capabilities': ['image-generation'],
            'languages': ['English'],
            'context_window': 4000,
            'pricing': {'per_image': 0.03},
            'use_cases': ['image-generation'],
            'description': 'Specialized image generation model known for artistic and high-quality outputs.'
        }
    ]
    
    # Score third-party models based on requirements
    scored_models = []
    for model in third_party_models:
        score = 0
        match_reasons = []
        mismatch_reasons = []
        
        # Score based on capabilities
        if requirements['capabilities']:
            capability_matches = []
            for capability in requirements['capabilities']:
                if capability in model['capabilities']:
                    score += 10
                    capability_matches.append(capability)
            
            if capability_matches:
                match_reasons.append(f"Supports required capabilities: {', '.join(capability_matches)}")
            else:
                mismatch_reasons.append(f"Missing required capabilities: {', '.join(requirements['capabilities'])}")
        
        # Score based on use cases
        if requirements['use_cases']:
            use_case_matches = []
            for use_case in requirements['use_cases']:
                if use_case in model['use_cases']:
                    score += 8
                    use_case_matches.append(use_case)
            
            if use_case_matches:
                match_reasons.append(f"Suitable for use cases: {', '.join(use_case_matches)}")
        
        # Score based on context length
        if requirements['context_length'] == 'high' and model['context_window'] >= 32000:
            score += 10
            match_reasons.append(f"Large context window: {model['context_window']:,} tokens")
        elif requirements['context_length'] == 'low' and model['context_window'] <= 16000:
            score += 5
            match_reasons.append(f"Efficient context window: {model['context_window']:,} tokens")
        
        # Score based on price sensitivity
        if requirements['price_sensitivity'] == 'high':
            if 'input' in model['pricing'] and model['pricing']['input'] <= 0.001:
                score += 15
                match_reasons.append(f"Low cost option")
            elif 'per_image' in model['pricing'] and model['pricing']['per_image'] <= 0.02:
                score += 15
                match_reasons.append(f"Low cost option")
        elif requirements['price_sensitivity'] == 'low':
            if 'input' in model['pricing'] and model['pricing']['input'] >= 0.005:
                score += 5
                match_reasons.append(f"Premium option with potentially better quality")
            elif 'per_image' in model['pricing'] and model['pricing']['per_image'] >= 0.03:
                score += 5
                match_reasons.append(f"Premium option with potentially better quality")
        
        # Add to scored models
        scored_models.append({
            'name': model['name'],
            'provider': model['provider'],
            'score': score,
            'match_reasons': match_reasons,
            'mismatch_reasons': mismatch_reasons,
            'description': model['description'],
            'pricing': model['pricing'],
            'context_window': model['context_window'],
            'capabilities': model['capabilities']
        })
    
    # Sort by score
    scored_models.sort(key=lambda x: x['score'], reverse=True)
    
    # Return top 3
    return scored_models[:3]


def explain_third_party_suggestion(model: Dict[str, Any]) -> str:
    """
    Generate an explanation for a third-party model suggestion.
    
    Args:
        model: Third-party model data
        
    Returns:
        Explanation string
    """
    explanation = f"**{model['name']}** by {model['provider']} is a third-party alternative because:\n\n"
    
    # Add description
    explanation += f"{model['description']}\n\n"
    
    # Add match reasons
    if model['match_reasons']:
        explanation += "**Strengths:**\n"
        for reason in model['match_reasons']:
            explanation += f"- {reason}\n"
        explanation += "\n"
    
    # Add mismatch reasons
    if model['mismatch_reasons']:
        explanation += "**Considerations:**\n"
        for reason in model['mismatch_reasons']:
            explanation += f"- {reason}\n"
        explanation += "\n"
    
    # Add pricing information
    explanation += "**Pricing:**\n"
    if 'input' in model['pricing'] and 'output' in model['pricing']:
        explanation += f"- Input: ${model['pricing']['input']:.4f} per 1K tokens\n"
        explanation += f"- Output: ${model['pricing']['output']:.4f} per 1K tokens\n"
    elif 'per_image' in model['pricing']:
        explanation += f"- ${model['pricing']['per_image']:.2f} per image\n"
    
    # Add note about availability
    explanation += "\n**Note:** This model is not available through Amazon Bedrock and would require integration with the provider's API."
    
    return explanation