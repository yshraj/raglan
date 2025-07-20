"""
LLM integration for text generation using OpenRouter.
"""
import os
import logging
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()


class LLMManager:
    """Handles LLM initialization and inference using OpenRouter."""

    def __init__(
        self, 
        model_name: str = "mistralai/mistral-7b-instruct:free",
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs
    ):
        """
        Initialize the LLM Manager with OpenRouter.
        
        Args:
            model_name: The OpenRouter model to use.
            temperature: The sampling temperature to use.
            max_tokens: The maximum number of tokens to generate.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Load the OpenRouter API key from environment variables
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            logger.warning("OPENROUTER_API_KEY not found in environment variables.")
            logger.warning("Please set your OpenRouter API key in the .env file or in your environment variables.")
            logger.warning("You can get an OpenRouter API key from https://openrouter.ai/keys")
            # We don't raise an exception here because we want the app to start
            # The user will be prompted in the Streamlit UI
        
        # Initialize OpenRouter client via LangChain
        self.llm = self._load_model()
    
    def _load_model(self):
        """
        Load the language model from OpenRouter.
        
        Returns:
            A LangChain compatible LLM.
        """
        try:
            # Setup OpenRouter as ChatOpenAI with the OpenRouter base URL
            logger.info(f"Initializing OpenRouter with model: {self.model_name}")
            
            from langchain_openai import ChatOpenAI
            
            # Setup OpenRouter configurations
            llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=self.api_key,
                base_url="https://openrouter.ai/api/v1",
                # Required HTTP headers for OpenRouter
                default_headers={
                    "HTTP-Referer": "https://github.com/raglan-rag-app",
                    "X-Title": "Raglan RAG Application",
                    # OpenRouter tracks usage by these headers
                    "User-Agent": "raglan-rag/1.0.0"
                }
            )
            
            logger.info(f"Successfully initialized OpenRouter with model: {self.model_name}")
            return llm
            
        except Exception as e:
            logger.error(f"Error initializing OpenRouter with model {self.model_name}: {str(e)}")
            
            # Fallback to a different free model if the first one fails
            fallback_model = "google/gemma-7b-it:free"
            
            try:
                logger.info(f"Attempting to initialize fallback model: {fallback_model}")
                
                from langchain_openai import ChatOpenAI
                
                llm = ChatOpenAI(
                    model=fallback_model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    api_key=self.api_key,
                    base_url="https://openrouter.ai/api/v1",
                    default_headers={
                        "HTTP-Referer": "https://github.com/raglan-rag-app",
                        "X-Title": "Raglan RAG Application",
                        "User-Agent": "raglan-rag/1.0.0"
                    }
                )
                
                logger.info(f"Successfully initialized fallback model: {fallback_model}")
                return llm
                
            except Exception as fallback_error:
                logger.error(f"Failed to initialize fallback model: {str(fallback_error)}")
                raise fallback_error
    
    def generate(self, prompt: str) -> str:
        """
        Generate text based on a prompt using OpenRouter.
        
        Args:
            prompt: The input prompt.
            
        Returns:
            Generated text.
        """
        from langchain_core.messages import HumanMessage
        
        # Convert the text prompt to a ChatMessage
        messages = [HumanMessage(content=prompt)]
        
        # Invoke the model and extract the response content
        response = self.llm.invoke(messages)
        return response.content
