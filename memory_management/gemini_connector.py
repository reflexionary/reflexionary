"""
Tethys - Gemini Connector

This module provides a high-level interface for interacting with Google's Gemini AI models.
It handles message formatting, tool definitions, and response parsing in a way that's
optimized for Tethys's memory management system.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Tuple

# Configure logging for this module
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

try:
    import google.generativeai as genai
    from google.generativeai import types
except ImportError:
    logger.critical("The 'google-generativeai' library is required. Please install it using 'pip install google-generativeai'.")
    exit(1)

# --- Configuration Class for Gemini Connector ---
class GeminiConfig:
    """
    Configuration class for the Gemini Connector.
    
    This class defines all the configurable parameters for the Gemini model,
    including model selection, generation parameters, and API key management.
    """
    
    def __init__(self, model: str = "gemini-1.5-flash", 
                 temperature: float = 0.7, 
                 max_tokens: int = 2048, 
                 top_p: float = 1.0, 
                 api_key: Optional[str] = None):
        """
        Initialize Gemini configuration parameters.

        Args:
            model (str): The specific Gemini model to use (e.g., "gemini-1.5-flash", "gemini-pro").
            temperature (float): Controls the randomness of the response (0.0 to 2.0). 
                              Lower is more deterministic.
            max_tokens (int): Maximum number of tokens to generate in the response.
            top_p (float): Nucleus sampling parameter (0.0 to 1.0). Controls diversity.
            api_key (str, optional): Google Gemini API key. If None, tries to get from env var.
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.api_key = api_key


class GeminiConnector:
    """
    A custom wrapper for the Google Gemini API, centralizing interaction logic.
    
    This class handles message reformatting, tool definitions for Function Calling,
    and robust response parsing. It's designed to be the primary interface between
    Tethys and the Gemini API.
    """
    
    def __init__(self, config: Optional[GeminiConfig] = None):
        """
        Initialize the GeminiConnector.

        Args:
            config (GeminiConfig, optional): Configuration object for Gemini.
                                           If None, default config is used.
        """
        self.config = config if config else GeminiConfig()
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the Gemini client with proper configuration."""
        # Ensure API key is set either in config or environment variable
        api_key = self.config.api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.critical("GeminiConnector: GEMINI_API_KEY not found. "
                          "Please set it in config or environment variables.")
            exit(1)  # Critical failure if API key is missing

        # Configure the genai client globally (ideally once per application lifecycle)
        try:
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(model_name=self.config.model)
            logger.info(f"GeminiConnector: Model '{self.config.model}' initialized successfully.")
        except Exception as e:
            logger.critical(f"GeminiConnector: CRITICAL ERROR initializing Gemini model '{self.config.model}': {e}")
            logger.critical("Please ensure your API key is valid and the model name is correct.")
            exit(1)

    def _parse_response(self, response: Any, tools: Optional[List[Dict]]) -> Dict[str, Any]:
        """
        Parse the raw response object from the Gemini API.
        
        Extracts content and/or tool calls, and handles potential blocking due to safety settings.

        Args:
            response: The raw response object from Gemini.
            tools: The list of tools provided in the original request.

        Returns:
            dict: A dictionary containing 'content' (str), 'tool_calls' (list of dicts),
                  'blocked' (bool), 'finish_reason' (str), and 'safety_ratings' (list of dicts).
        """
        processed_response = {
            "content": None,
            "tool_calls": [],
            "blocked": False,
            "finish_reason": None,
            "safety_ratings": [],
        }

        # Check for safety blocking first
        if not response.candidates:
            processed_response["blocked"] = True
            if hasattr(response, "prompt_feedback") and hasattr(response.prompt_feedback, "block_reason"):
                processed_response["finish_reason"] = response.prompt_feedback.block_reason.name
            if hasattr(response, "prompt_feedback") and hasattr(response.prompt_feedback, "safety_ratings"):
                processed_response["safety_ratings"] = [
                    {"category": r.category.name, "probability": r.probability.name}
                    for r in response.prompt_feedback.safety_ratings
                ]
            logger.warning(f"GeminiConnector: Response blocked by safety settings. Reason: {processed_response['finish_reason']}")
            return processed_response

        candidate = response.candidates[0]
        if hasattr(candidate, "finish_reason"):
            processed_response["finish_reason"] = candidate.finish_reason.name
        if hasattr(candidate, "safety_ratings"):
            processed_response["safety_ratings"] = [
                {"category": r.category.name, "probability": r.probability.name}
                for r in candidate.safety_ratings
            ]

        # Extract content and function calls from parts
        if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    processed_response["content"] = part.text
                if hasattr(part, "function_call") and part.function_call:
                    fn = part.function_call
                    processed_response["tool_calls"].append(
                        {
                            "name": fn.name,
                            "arguments": dict(fn.args) if hasattr(fn, 'args') and fn.args else {},
                        }
                    )
        return processed_response

    def _reformat_messages(self, messages: List[Dict[str, Any]]) -> Tuple[Optional[str], List[Any]]:
        """
        Reformat messages into Gemini's required format.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
                     'content' can be str, dict (for tool response), or list (for multimodal).

        Returns:
            tuple: (system_instruction, contents_list)
        """
        system_instruction = None
        contents = []

        for message in messages:
            role = message["role"]
            content_data = message["content"]

            if role == "system":
                system_instruction = content_data
            elif role == "tool":
                if isinstance(content_data, dict) and "name" in content_data and "response" in content_data:
                    function_response = {
                        "name": content_data["name"],
                        "response": content_data["response"]
                    }
                    # Create a function response part
                    part = {"function_response": function_response}
                    contents.append({"role": "function", "parts": [part]})
                else:
                    logger.warning(f"GeminiConnector: Malformed tool message content: {content_data}. Sending as user text.")
                    contents.append({"role": "user", "parts": [{"text": str(content_data)}]})
            else:  # 'user' or 'model' roles
                if isinstance(content_data, str):
                    contents.append({"role": role, "parts": [{"text": content_data}]})
                elif isinstance(content_data, list):  # For multimodal inputs
                    parts = []
                    for item in content_data:
                        if isinstance(item, str):
                            parts.append({"text": item})
                        elif isinstance(item, dict) and "mime_type" in item and "data" in item:
                            parts.append({
                                "inline_data": {
                                    "mime_type": item["mime_type"],
                                    "data": item["data"]
                                }
                            })
                        else:
                            logger.warning(f"GeminiConnector: Unexpected multimodal part type: {type(item)}. Converting to string.")
                            parts.append({"text": str(item)})
                    contents.append({"role": role, "parts": parts})
                else:
                    logger.warning(f"GeminiConnector: Unexpected message content type for role {role}: {type(content_data)}. Converting to string.")
                    contents.append({"role": role, "parts": [{"text": str(content_data)}]})
                    
        return system_instruction, contents

    def _reformat_tools_for_gemini(self, tools: Optional[List[Dict]]) -> Optional[List[Dict]]:
        """
        Reformat tool definitions into Gemini's expected format.
        
        Args:
            tools: List of tool definitions (each dict must have a 'function' key).

        Returns:
            list: A list of tool definitions in Gemini's format, or None if no tools.
        """
        if not tools:
            return None

        function_declarations = []
        for tool_def in tools:
            if "function" in tool_def:
                func_spec = tool_def["function"].copy()
                
                # Clean the parameters to remove 'additionalProperties'
                cleaned_parameters = self._remove_additional_properties_recursive(
                    func_spec.get("parameters", {})
                )

                function_declaration = {
                    "name": func_spec["name"],
                    "description": func_spec.get("description", ""),
                    "parameters": cleaned_parameters,
                }
                function_declarations.append({"function_declaration": function_declaration})
        
        if not function_declarations:
            return None

        return [{"function_declarations": function_declarations}]

    def _remove_additional_properties_recursive(self, data: Any) -> Any:
        """
        Recursively remove 'additionalProperties' from nested dictionaries.
        
        This is needed because Gemini's API doesn't accept the 'additionalProperties' field
        that's often present in OpenAPI/JSON Schema definitions.
        """
        if isinstance(data, dict):
            return {
                key: self._remove_additional_properties_recursive(value)
                for key, value in data.items()
                if key != "additionalProperties"
            }
        elif isinstance(data, list):
            return [self._remove_additional_properties_recursive(item) for item in data]
        else:
            return data

    def generate_response(
        self,
        messages: List[Dict[str, Any]],
        response_format: Optional[Dict] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto"
    ) -> Dict[str, Any]:
        """
        Generate a response from Gemini.
        
        This is the primary method for interacting with the Gemini API. It handles
        message formatting, tool definitions, and response parsing.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            response_format: Optional dict specifying response format (e.g., JSON schema).
            tools: Optional list of tool definitions the model can call.
            tool_choice: Controls how Gemini uses tools ("auto", "any", "none").

        Returns:
            dict: A processed response containing content, tool calls, and metadata.
        """
        # Reformat messages for Gemini API
        system_instruction, contents = self._reformat_messages(messages)

        # Prepare generation config
        generation_config = {
            "temperature": self.config.temperature,
            "max_output_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
        }

        # Add system instruction as the first message if present
        if system_instruction:
            contents.insert(0, {"role": "user", "parts": [{"text": system_instruction}]})
            contents.insert(1, {"role": "model", "parts": [{"text": "Understood. I will follow these instructions."}]})

        # Handle structured response format (e.g., JSON output)
        if response_format and response_format.get("type") == "json_object":
            generation_config["response_mime_type"] = "application/json"
            if "schema" in response_format:
                generation_config["response_schema"] = self._remove_additional_properties_recursive(
                    response_format["schema"]
                )

        # Handle tools and tool_choice
        formatted_tools = self._reformat_tools_for_gemini(tools)
        if formatted_tools:
            generation_config["tools"] = formatted_tools
            
            # Set tool_config based on tool_choice
            tool_config = {}
            if tool_choice == "any" and tools:
                tool_config["function_calling_config"] = {
                    "mode": "ANY",
                    "allowed_function_names": [t["function"]["name"] for t in tools]
                }
            elif tool_choice == "none":
                tool_config["function_calling_config"] = {"mode": "NONE"}
            
            if tool_config:
                generation_config["tool_config"] = tool_config

        try:
            # Make the API call
            response = self.client.generate_content(
                contents=contents,
                generation_config=generation_config,
                # safety_settings=self.config.safety_settings  # Uncomment if you add safety settings
            )
            return self._parse_response(response, tools)
            
        except Exception as e:
            logger.error(f"GeminiConnector: ERROR during API call: {e}")
            return {
                "content": f"An error occurred with the AI. Please try again. ({e})",
                "tool_calls": [],
                "blocked": False,
                "finish_reason": "ERROR"
            }


# --- Self-Verification Block (for isolated testing) ---
if __name__ == "__main__":
    logger.info("\n--- GeminiConnector: Self-Test Initiated ---")

    # Ensure GEMINI_API_KEY is set in your environment
    if not os.getenv("GEMINI_API_KEY"):
        logger.error("GEMINI_API_KEY environment variable not set. Please set it before running tests.")
        exit(1)

    # Configure basic logging for self-test
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Test 1: Basic text generation
    logger.info("\n[Test 1] Basic Text Generation:")
    connector = GeminiConnector()  # Uses default config
    messages_basic = [{"role": "user", "content": "Tell me a fun fact about finance."}]
    response_basic = connector.generate_response(messages_basic)
    logger.info(f"  Response Content: {response_basic['content'][:100]}...")
    assert response_basic['content'] is not None and len(response_basic['content']) > 0, \
           "Test 1 Failed: No content in basic response."
    logger.info("  Test 1 Passed: Basic text generation successful.")

    # Test 2: Text generation with a system instruction
    logger.info("\n[Test 2] Text Generation with System Instruction:")
    messages_system = [
        {"role": "system", "content": "You are a very concise AI. Answer in 5 words or less."},
        {"role": "user", "content": "What is compound interest?"}
    ]
    response_system = connector.generate_response(messages_system)
    logger.info(f"  Response Content: {response_system['content']}")
    word_count = len(response_system['content'].split())
    logger.info(f"  Word count: {word_count}")
    assert word_count <= 10, \
           f"Test 2 Failed: Response too long ({word_count} words). Expected 10 or fewer words."
    logger.info("  Test 2 Passed: System instruction followed.")

    # Test 3: Function Calling - Model decides (auto)
    logger.info("\n[Test 3] Function Calling (auto mode):")
    # Define a mock tool
    mock_tool_def = {
        "function": {
            "name": "get_current_stock_price",
            "description": "Gets the current stock price for a given ticker symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The stock ticker symbol (e.g., GOOGL)."
                    }
                },
                "required": ["ticker"],
            },
        }
    }
    
    messages_tool_auto = [
        {"role": "user", "content": "What is the price of GOOGL?"}
    ]
    
    response_tool_auto = connector.generate_response(
        messages_tool_auto, 
        tools=[mock_tool_def], 
        tool_choice="auto"
    )
    
    logger.info(f"  Response Content: {response_tool_auto['content']}")
    logger.info(f"  Tool Calls: {response_tool_auto['tool_calls']}")
    assert len(response_tool_auto['tool_calls']) > 0 and \
           response_tool_auto['tool_calls'][0]['name'] == "get_current_stock_price", \
           "Test 3 Failed: Model did not call the expected tool."
    logger.info("  Test 3 Passed: Function calling in auto mode successful.")

    # Test 4: Function Calling - Provide tool output
    logger.info("\n[Test 4] Function Calling (providing tool output):")
    messages_tool_output = [
        {"role": "user", "content": "What is the price of GOOGL?"},
        {"role": "model", "content": None, "tool_calls": [
            {"name": "get_current_stock_price", "arguments": {"ticker": "GOOGL"}}
        ]},
        {"role": "tool", "content": {
            "name": "get_current_stock_price",
            "response": {"price": 150.25, "currency": "USD"}
        }}
    ]
    
    response_tool_output = connector.generate_response(
        messages_tool_output,
        tools=[mock_tool_def],
        tool_choice="auto"
    )
    
    logger.info(f"  Response Content: {response_tool_output['content']}")
    assert "150.25" in response_tool_output['content'], \
           "Test 4 Failed: Gemini did not use tool output in response."
    logger.info("  Test 4 Passed: Tool output processing successful.")

    logger.info("\n--- GeminiConnector: All Self-Tests Completed Successfully ---")
