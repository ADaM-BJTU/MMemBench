"""
LLM Client for M3Bench User Simulator
======================================

Handles API calls to both the core model (thinking) and target model (image).
"""

import requests
import json
import base64
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


class LLMClient:
    """Client for interacting with LLM APIs"""

    def __init__(
        self,
        api_url: str = "https://globalai.vip/v1/chat/completions",
        api_key: str = "sk-PcyvuAqtt0yHsP88Mga584zkJIeP7VrSC2l4QOaK0wGpSx3R",
        core_model: str = "gemini-3-pro-image-preview",
        target_model: str = "gemini-2.5-flash-image-preview",
        weak_model: str = "claude-3-5-haiku-20241022-c",  # For filler generation (cheap model)
        timeout: int = 60
    ):
        """
        Initialize LLM client.

        Args:
            api_url: API endpoint URL
            api_key: API key
            core_model: Model name for the core/thinking model (drives simulation)
            target_model: Model name for the target/image model (being tested)
            weak_model: Model name for filler content (e.g., gpt-3.5-turbo, saves cost)
            timeout: Request timeout in seconds
        """
        self.api_url = api_url
        self.api_key = api_key
        self.core_model = core_model
        self.target_model = target_model
        self.weak_model = weak_model or "gpt-3.5-turbo"  # Default weak model
        self.timeout = timeout

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

    def _encode_image(self, image_path: str) -> Optional[str]:
        """Encode image to base64"""
        path = Path(image_path)
        if not path.exists():
            logger.warning(f"Image not found: {image_path}")
            return None

        try:
            with open(path, "rb") as f:
                image_data = f.read()
            return base64.b64encode(image_data).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            return None

    def _get_image_mime_type(self, image_path: str) -> str:
        """Get MIME type from file extension"""
        ext = Path(image_path).suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp"
        }
        return mime_types.get(ext, "image/jpeg")

    def call_core_model(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 2000,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Call the core model (thinking model) with retry logic.

        Args:
            messages: Chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            API response dict with 'content' and 'reasoning_content'
        """
        payload = {
            "model": self.core_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    result = response.json()
                    choice = result.get("choices", [{}])[0]
                    message = choice.get("message", {})

                    return {
                        "success": True,
                        "content": message.get("content", ""),
                        "reasoning_content": message.get("reasoning_content", ""),
                        "finish_reason": choice.get("finish_reason", ""),
                        "usage": result.get("usage", {})
                    }
                else:
                    last_error = f"API error: {response.status_code} - {response.text[:200]}"
                    logger.warning(f"Core model attempt {attempt+1} failed: {last_error}")

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Core model attempt {attempt+1} exception: {e}")

            # Wait before retry
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)

        logger.error(f"Core model call failed after {MAX_RETRIES} attempts: {last_error}")
        return {
            "success": False,
            "error": last_error
        }

    def call_target_model(
        self,
        messages: List[Dict[str, Any]],
        images: Optional[List[str]] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Call the target model (image model).

        Args:
            messages: Chat messages
            images: List of image paths to include
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            API response dict with 'content'
        """
        # Build messages with images if provided
        processed_messages = []

        for msg in messages:
            if msg["role"] == "user" and images:
                # Add images to the first user message
                content = []

                # Add images first
                for img_path in images:
                    base64_img = self._encode_image(img_path)
                    if base64_img:
                        mime_type = self._get_image_mime_type(img_path)
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_img}"
                            }
                        })

                # Add text content
                content.append({
                    "type": "text",
                    "text": msg["content"]
                })

                processed_messages.append({
                    "role": "user",
                    "content": content
                })

                # Only add images to first user message
                images = None
            else:
                processed_messages.append(msg)

        payload = {
            "model": self.target_model,
            "messages": processed_messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    result = response.json()
                    choice = result.get("choices", [{}])[0]
                    message = choice.get("message", {})

                    return {
                        "success": True,
                        "content": message.get("content", ""),
                        "finish_reason": choice.get("finish_reason", ""),
                        "usage": result.get("usage", {})
                    }
                else:
                    last_error = f"API error: {response.status_code} - {response.text[:200]}"
                    logger.warning(f"Target model attempt {attempt+1} failed: {last_error}")

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Target model attempt {attempt+1} exception: {e}")

            # Wait before retry
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)

        logger.error(f"Target model call failed after {MAX_RETRIES} attempts: {last_error}")
        return {
            "success": False,
            "error": last_error
        }

    def test_connection(self) -> Dict[str, bool]:
        """Test connection to both models"""
        results = {}

        # Test core model
        core_result = self.call_core_model([
            {"role": "user", "content": "Say 'OK'"}
        ], max_tokens=10)
        results["core_model"] = core_result.get("success", False)

        # Test target model
        target_result = self.call_target_model([
            {"role": "user", "content": "Say 'OK'"}
        ], max_tokens=10)
        results["target_model"] = target_result.get("success", False)

        return results

    def call_weak_model(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 500,
        temperature: float = 0.9
    ) -> Dict[str, Any]:
        """
        Call the weak/cheap model for filler content generation.
        Used for pseudo multi-turn to save costs on verbose content.

        Args:
            messages: Chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher for more varied filler)

        Returns:
            API response dict with 'content'
        """
        payload = {
            "model": self.weak_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    result = response.json()
                    choice = result.get("choices", [{}])[0]
                    message = choice.get("message", {})

                    return {
                        "success": True,
                        "content": message.get("content", ""),
                        "finish_reason": choice.get("finish_reason", ""),
                        "usage": result.get("usage", {})
                    }
                else:
                    last_error = f"API error: {response.status_code} - {response.text[:200]}"
                    logger.warning(f"Weak model attempt {attempt+1} failed: {last_error}")

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Weak model attempt {attempt+1} exception: {e}")

            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)

        logger.error(f"Weak model call failed after {MAX_RETRIES} attempts: {last_error}")
        return {
            "success": False,
            "error": last_error
        }


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    client = LLMClient()
    print("Testing LLM Client...")

    # Test connection
    results = client.test_connection()
    print(f"Core model: {'OK' if results['core_model'] else 'FAILED'}")
    print(f"Target model: {'OK' if results['target_model'] else 'FAILED'}")

    # Test core model with longer response
    print("\n--- Core Model Test ---")
    response = client.call_core_model([
        {"role": "user", "content": "What is 2+2? Think step by step."}
    ])
    if response["success"]:
        print(f"Content: {response['content']}")
        print(f"Reasoning: {response['reasoning_content'][:200]}...")