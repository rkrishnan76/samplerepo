import logging
import time
from typing import Dict, Any, Tuple

from langchain_openai.chat_models.azure import AzureChatOpenAI
from langchain_openai.chat_models.base import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI
from langchain_google_vertexai.model_garden_maas import VertexModelGardenLlama
from langchain_google_vertexai.model_garden import ChatAnthropicVertex
from langchain.prompts.chat import ChatPromptTemplate
from langchain_community.callbacks import get_openai_callback
from langchain_core.messages.ai import AIMessage
from langchain_aws import ChatBedrock

from ml.config.model_config import ModelConfigManager
from utils.common import create_token_metadata, create_empty_token_metadata
from ml.utils.exceptions import LLMResponseException
from ml.utils.common import parse_str_into_json, clean_json_string, update_metadata_metrics
from ml import langchain_tracing
from botocore.config import Config
import boto3

class LLMRequest:
    """A request to an LLM using configuration from ModelConfigManager"""
    
    def __init__(self, chain_id: str, prompt: ChatPromptTemplate, **kwargs):
        """
        Initialize LLMRequest with a chain ID and configuration
        
        Args:
            chain_id: The identifier for the chain (e.g., 'planner', 'structure')
            prompt: The ChatPromptTemplate to use
            **kwargs: Additional parameters that will override config values
        """
        self.chain_id = chain_id
        self.prompt = prompt
        
        # Get configuration for this chain
        config_manager = ModelConfigManager()
        self.model_config = config_manager.get_config_for_chain(chain_id)
        
        # Extract base parameters from config
        self.llm_type = self.model_config.provider
        self.api_version = self.model_config.api_version
        self.endpoint = self.model_config.endpoint
        self.model_name = self.model_config.model
        self.deployment_name = self.model_config.deployment_name
        self.temperature = self.model_config.temperature
        self.max_tokens = self.model_config.max_tokens
        self.max_retries = self.model_config.max_retries
        self.request_timeout = self.model_config.request_timeout
        self.top_p = self.model_config.top_p
        self.invalid_json_retry_count = self.model_config.invalid_json_retry_count
        
        # Override with any explicitly passed parameters
        self.tags = kwargs.get("tags", [])
        self.callbacks = (kwargs.get("callbacks", []) or []) if langchain_tracing == "true" else []
        self.run_name = kwargs.get("run_name", f"{chain_id.capitalize()} Chain")
        self.base_code = kwargs.get("base_code", self.run_name)
        
        # Allow explicitly overriding model configuration
        if "temperature" in kwargs:
            self.temperature = kwargs["temperature"]
        if "max_tokens" in kwargs:
            self.max_tokens = kwargs["max_tokens"]
        if "max_retries" in kwargs:
            self.max_retries = kwargs["max_retries"]
        if "request_timeout" in kwargs:
            self.request_timeout = kwargs["request_timeout"]
        if "invalid_json_retry_count" in kwargs:
            self.invalid_json_retry_count = kwargs["invalid_json_retry_count"]
        if "top_p" in kwargs:
            self.top_p = kwargs["top_p"]
            
        self._validate()
        self._create_chain()

    def _validate(self):
        """Validate the LLM configuration"""
        if self.llm_type not in ["azure-openai", "openai", "vertexai", "bedrock"]:
            raise ValueError(f"Invalid LLM Type {self.llm_type}. Supported: [openai, azure-openai, vertexai, bedrock]")

        if not isinstance(self.prompt, ChatPromptTemplate):
            raise TypeError(f"Prompt should be ChatPromptTemplate but is {type(self.prompt)} instead")

    def _create_chain(self):
        """Create the LLM chain based on configuration"""
        if self.llm_type == "azure-openai":
            self.llm = AzureChatOpenAI(deployment_name=self.deployment_name,
                                       azure_endpoint=self.endpoint,
                                       api_version=self.api_version,
                                       temperature=self.temperature,
                                       model_name=self.model_name,
                                       max_tokens=self.max_tokens,
                                       max_retries=self.max_retries,
                                       request_timeout=self.request_timeout)
        elif self.llm_type == "openai":
            self.llm = ChatOpenAI(model_name=self.model_name,
                                  temperature=self.temperature,
                                  max_tokens=self.max_tokens,
                                  max_retries=self.max_retries,
                                  request_timeout=self.request_timeout)
        elif self.llm_type == "vertexai":
            if self.model_name.startswith("meta/"):
                self.llm = VertexModelGardenLlama(model_name=self.model_name)
            elif self.model_name.startswith("anthropic/"):
                model_name = self.model_name.replace("anthropic/", "")
                self.llm = ChatAnthropicVertex(model_name=model_name,
                                               temperature=self.temperature,
                                               max_tokens=self.max_tokens,
                                               max_retries=self.max_retries)
            else:
                self.llm = ChatVertexAI(
                    model_name=self.model_name,
                    temperature=self.temperature,
                    max_retries=self.max_retries
                )
        elif self.llm_type == "bedrock":
            bedrock_config = Config(
                retries={
                    'max_attempts': self.max_retries,
                    'mode': 'standard'
                },
                read_timeout = self.request_timeout,
                connect_timeout = self.request_timeout
            )
            bedrock_client = boto3.client(
                'bedrock-runtime',
                config=bedrock_config
            )
            self.llm = ChatBedrock(model_id=self.model_name,
                                   temperature=self.temperature,
                                   max_tokens=self.max_tokens,
                                   client=bedrock_client
                                   )
        self.chain = self.prompt | self.llm

    def run(self, inputs: Dict[str, Any] = {}) -> Tuple[str, Dict[str, Any], float]:
        """Run the LLM chain with the given inputs"""
        try:
            llm_start = time.time()

            if self.llm_type == "vertexai":
                response = self.chain.invoke(inputs, config={"run_name": self.run_name, "tags": self.tags, "callbacks": self.callbacks})
                if isinstance(response, AIMessage):
                    response = response.content
                token_metadata = create_empty_token_metadata()
            else:
                with get_openai_callback() as cb:
                    response = self.chain.invoke(inputs, config={"run_name": self.run_name, "tags": self.tags, "callbacks": self.callbacks})
                    response = response.content
                    
                    # Retry for invalid JSON if needed
                    retry_count = 1
                    while retry_count <= self.invalid_json_retry_count:
                        try:
                            if parse_str_into_json(clean_json_string(response)):
                                break
                            logging.error(f"GIA: Invalid JSON response for {self.run_name}, retrying {retry_count}/{self.invalid_json_retry_count}")
                            response = self.chain.invoke(inputs, config={"run_name": self.run_name, "tags": self.tags, "callbacks": self.callbacks})
                            response = response.content
                        except Exception as e:
                            logging.error(f"GIA: Exception while validating JSON response for {self.run_name}, retrying {retry_count}/{self.invalid_json_retry_count}")
                            response = self.chain.invoke(inputs, config={"run_name": self.run_name, "tags": self.tags, "callbacks": self.callbacks})
                            response = response.content
                        retry_count += 1
                    
                    token_metadata = create_token_metadata(cb)
            
            llm_end = time.time()
            duration = llm_end - llm_start
            
            logging.info("GIA: LLM request successful")
            update_metadata_metrics(token_metadata, run_name=self.run_name)
            
            return response, token_metadata, duration
        
        except Exception as e:
            raise LLMResponseException(e, self.base_code)