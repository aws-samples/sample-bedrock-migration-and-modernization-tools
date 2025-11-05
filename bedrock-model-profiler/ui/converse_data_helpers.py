"""
Helper functions to extract and display converse_data from models
"""

def get_converse_context_window(model):
    """Extract context window from converse_data with fallback"""
    try:
        converse_data = model.get('converse_data', {})
        if isinstance(converse_data, dict):
            context_window = converse_data.get('context_window', 0)
            # Handle both integer and string "N/A" values
            if isinstance(context_window, int) and context_window > 0:
                return context_window
            elif isinstance(context_window, str) and context_window != "N/A":
                try:
                    return int(context_window)
                except ValueError:
                    pass
    except (KeyError, TypeError, AttributeError):
        # Silently handle missing or invalid converse_data structure
        pass

    # Fallback to original field
    return model.get('context_window', 0)

def get_converse_max_output_tokens(model):
    """Extract max output tokens from converse_data with fallback"""
    try:
        converse_data = model.get('converse_data', {})
        if isinstance(converse_data, dict):
            max_output_tokens = converse_data.get('max_output_tokens', 0)
            # Handle both integer and string "N/A" values
            if isinstance(max_output_tokens, int) and max_output_tokens > 0:
                return max_output_tokens
            elif isinstance(max_output_tokens, str) and max_output_tokens != "N/A":
                try:
                    return int(max_output_tokens)
                except ValueError:
                    pass
    except (KeyError, TypeError, AttributeError):
        # Silently handle missing or invalid converse_data structure
        pass

    # Fallback to original field
    return model.get('max_output_tokens', 0)

def get_converse_size_category(model):
    """Extract size category from converse_data"""
    try:
        converse_data = model.get('converse_data', {})
        if isinstance(converse_data, dict):
            size_category = converse_data.get('size_category', {})
            if isinstance(size_category, dict) and size_category:
                return size_category
    except (KeyError, TypeError, AttributeError):
        # Silently handle missing or invalid converse_data structure
        pass

    return {"category": "Unknown", "color": "#6B7280", "tier": 0}

def get_converse_function_calling(model):
    """Extract function calling support from converse_data"""
    try:
        converse_data = model.get('converse_data', {})
        if isinstance(converse_data, dict):
            capability_analysis = converse_data.get('capability_analysis', {})
            if isinstance(capability_analysis, dict):
                function_calling = capability_analysis.get('function_calling', {})
                if isinstance(function_calling, dict):
                    return function_calling
    except (KeyError, TypeError, AttributeError):
        # Silently handle missing or invalid converse_data structure
        pass

    return {"supported": False, "confidence": 0.0, "reasoning": "No data available"}

def get_converse_performance_estimates(model):
    """Extract performance estimates from converse_data"""
    try:
        converse_data = model.get('converse_data', {})
        if isinstance(converse_data, dict):
            performance = converse_data.get('performance_estimates', {})
            if isinstance(performance, dict):
                return performance
    except (KeyError, TypeError, AttributeError):
        # Silently handle missing or invalid converse_data structure
        pass

    return {"estimated_latency_ms": 0, "time_to_first_token_ms": 0, "tokens_per_second": 0}

def get_converse_recommendation(model):
    """Extract smart recommendation from converse_data"""
    try:
        converse_data = model.get('converse_data', {})
        if isinstance(converse_data, dict):
            recommendation = converse_data.get('recommendation_engine', {})
            if isinstance(recommendation, dict):
                return recommendation
    except (KeyError, TypeError, AttributeError):
        # Silently handle missing or invalid converse_data structure
        pass

    return {"recommendation": "Unknown", "score": 0.0, "context_tier": "Unknown", "quality_tier": "Unknown"}

def format_context_window_with_badge(context_window, size_category):
    """Format context window display with size badge"""
    if context_window <= 0:
        return "Context: Not specified"

    # Format the number with commas
    formatted_context = f"{context_window:,}"

    # Get size info
    category = size_category.get('category', 'Unknown')
    color = size_category.get('color', '#6B7280')

    # Create HTML with badge
    return f"""
    <div style="margin: 4px 0;">
        <strong>Context:</strong> {formatted_context} tokens
        <span style="background: {color}; color: white; padding: 2px 6px; border-radius: 8px; font-size: 0.65rem; font-weight: bold; margin-left: 6px;">{category}</span>
    </div>
    """

def format_max_output_tokens(max_output_tokens):
    """Format max output tokens display"""
    if max_output_tokens <= 0:
        return "Max Output: Not specified"

    return f"Max Output: {max_output_tokens:,} tokens"

def format_function_calling_badge(function_calling):
    """Format function calling support as a badge"""
    supported = function_calling.get('supported', False)
    confidence = function_calling.get('confidence', 0.0)

    if supported:
        color = "#10B981" if confidence > 0.8 else "#F59E0B"
        text = "Function Calling ✅"
        confidence_text = f"({confidence:.0%} confidence)"
    else:
        color = "#6B7280"
        text = "Function Calling ❌"
        confidence_text = ""

    return f"""
    <div style="margin: 2px 0;">
        <span style="background: {color}; color: white; padding: 3px 8px; border-radius: 10px; font-size: 0.7rem; font-weight: bold;">{text}</span>
        {f'<span style="color: #6B7280; font-size: 0.65rem; margin-left: 4px;">{confidence_text}</span>' if confidence_text else ''}
    </div>
    """

def format_performance_summary(performance):
    """Format performance estimates summary"""
    latency = performance.get('estimated_latency_ms', 0)
    tokens_per_sec = performance.get('tokens_per_second', 0)

    if latency <= 0:
        return "Performance: Not available"

    return f"Est. Latency: {latency}ms | Speed: {tokens_per_sec} tok/sec"

def format_recommendation_badge(recommendation):
    """Format recommendation as colored badge"""
    rec_text = recommendation.get('recommendation', 'Unknown')
    score = recommendation.get('score', 0.0)

    # Color based on recommendation quality
    if score >= 0.85:
        color = "#10B981"  # Green for excellent
    elif score >= 0.70:
        color = "#3B82F6"  # Blue for very good
    elif score >= 0.55:
        color = "#F59E0B"  # Orange for good
    elif score >= 0.40:
        color = "#EF4444"  # Red for fair
    else:
        color = "#6B7280"  # Gray for poor

    return f"""
    <div style="margin: 4px 0;">
        <span style="background: {color}; color: white; padding: 4px 10px; border-radius: 12px; font-size: 0.7rem; font-weight: bold;">{rec_text}</span>
        <span style="color: #6B7280; font-size: 0.65rem; margin-left: 6px;">Score: {score:.2f}</span>
    </div>
    """