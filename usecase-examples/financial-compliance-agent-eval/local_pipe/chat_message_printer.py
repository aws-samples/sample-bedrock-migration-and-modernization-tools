# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
from typing import Dict, Any, List
import json
from datetime import datetime

class ChatMessagePrinter:
    """Pretty printer for chat messages with role, content type, and model thoughts."""
    
    def __init__(self):
        self.colors = {
            'system': '\033[94m',      # Blue
            'user': '\033[92m',        # Green
            'assistant': '\033[93m',   # Yellow
            'tool': '\033[95m',        # Magenta
            'reset': '\033[0m',        # Reset
            'bold': '\033[1m',         # Bold
            'dim': '\033[2m'           # Dim
        }
    
    def print_messages(self, data: Dict[str, Any]) -> None:
        """Print all messages in a conversation."""
        messages = data.get('messages', [])
        last_message = data.get('last_message')
        
        print(f"{self.colors['bold']}=== CHAT CONVERSATION ==={self.colors['reset']}\n")
        
        for i, message in enumerate(messages):
            self._print_message(message, i + 1)
            print()  # Add spacing between messages
        
        if last_message and last_message not in messages:
            print(f"{self.colors['bold']}=== LAST MESSAGE ==={self.colors['reset']}\n")
            self._print_message(last_message, len(messages) + 1)
    
    def _print_message(self, message: Any, index: int) -> None:
        """Print a single message with formatting."""
        role = getattr(message, '_role', 'unknown')
        content = getattr(message, '_content', [])
        meta = getattr(message, '_meta', {})
        
        # Header with role and index
        color = self.colors.get(role.value if hasattr(role, 'value') else str(role), self.colors['reset'])
        print(f"{color}{self.colors['bold']}[{index}] {role.value.upper() if hasattr(role, 'value') else str(role).upper()}{self.colors['reset']}")
        
        # Print metadata if available
        if meta:
            self._print_metadata(meta)
        
        # Print content
        self._print_content(content)
    
    def _print_metadata(self, meta: Dict[str, Any]) -> None:
        """Print message metadata."""
        print(f"{self.colors['dim']}Metadata:{self.colors['reset']}")
        
        if 'model' in meta:
            print(f"  Model: {meta['model']}")
        
        if 'usage' in meta:
            usage = meta['usage']
            print(f"  Tokens: {usage.get('prompt_tokens', 0)} prompt + {usage.get('completion_tokens', 0)} completion = {usage.get('total_tokens', 0)} total")
        
        if 'finish_reason' in meta:
            print(f"  Finish Reason: {meta['finish_reason']}")
        
        print()
    
    def _print_content(self, content: List[Any]) -> None:
        """Print message content based on type."""
        for item in content:
            content_type = type(item).__name__
            
            if hasattr(item, 'text'):
                # Text content
                text = item.text
                if text.startswith('<thinking>') and text.endswith('</thinking>'):
                    # Extract thinking content
                    thinking = text[10:-11].strip()  # Remove <thinking> tags
                    print(f"{self.colors['dim']}💭 Model Thoughts:{self.colors['reset']}")
                    print(f"{self.colors['dim']}{thinking}{self.colors['reset']}")
                elif text.startswith('<thinking>'):
                    # Handle partial thinking tags
                    parts = text.split('</thinking>')
                    if len(parts) > 1:
                        thinking = parts[0][10:].strip()
                        remaining = parts[1].strip()
                        print(f"{self.colors['dim']}💭 Model Thoughts:{self.colors['reset']}")
                        print(f"{self.colors['dim']}{thinking}{self.colors['reset']}")
                        if remaining:
                            print(f"📝 Response: {remaining}")
                    else:
                        print(f"📝 Text: {text}")
                else:
                    print(f"📝 Text: {text}")
            
            elif hasattr(item, 'tool_name'):
                # Tool call
                print(f"🔧 Tool Call: {item.tool_name}")
                if hasattr(item, 'arguments'):
                    print(f"   Arguments: {json.dumps(item.arguments, indent=2)}")
                if hasattr(item, 'id'):
                    print(f"   ID: {item.id}")
            
            elif hasattr(item, 'result'):
                # Tool result
                result = item.result
                print(f"⚙️  Tool Result:")
                if isinstance(result, str):
                    # Truncate long results
                    if len(result) > 500:
                        print(f"   {result[:500]}... (truncated)")
                    else:
                        print(f"   {result}")
                else:
                    print(f"   {json.dumps(result, indent=2)}")
            
            else:
                print(f"❓ Unknown Content Type ({content_type}): {str(item)[:200]}...")

def print_chat_messages(data: Dict[str, Any]) -> None:
    """Convenience function to print chat messages."""
    printer = ChatMessagePrinter()
    printer.print_messages(data)