"""
Service for interacting with Hugging Face chat models.
"""

import base64
from io import BytesIO
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from PIL import Image
import torch
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
)


class HuggingFaceLLMService:
    """
    Provides methods to interact with a Hugging Face chat model.
    """

    def __init__(self, model_id, provider="auto", temperature=0.2, top_p=0.6, api_key=None):
        """
        Initialize the service with the specified model and parameters.

        Args:
            model_id (str): Hugging Face model ID
            provider (str): Kept for backward compatibility, unused
            temperature (float): Controls randomness in generation
            top_p (float): Nucleus sampling parameter
            api_key (str, optional): Hugging Face token; if not provided, reads env vars
        """
        load_dotenv()
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = 512
        self.model_id = model_id

        # Keep provider/api_key args for backward-compatible constructor signature.
        self.provider = provider
        self.api_key = api_key or os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
        self.project_dir = Path(__file__).resolve().parents[1]
        self.hf_home = Path(os.getenv("HF_HOME", str(self.project_dir / ".hf_cache")))
        self.local_model_dir = Path(
            os.getenv(
                "LOCAL_MODEL_DIR",
                str(self.project_dir / ".models" / self.model_id.replace("/", "--")),
            )
        )
        self.preload_model_on_init = os.getenv("PRELOAD_MODEL_ON_INIT", "true").lower() == "true"
        self.preload_model_in_memory_on_init = (
            os.getenv("PRELOAD_MODEL_IN_MEMORY_ON_INIT", "true").lower() == "true"
        )

        # Keep all model/cache artifacts in project-local writable folders by default.
        os.environ.setdefault("HF_HOME", str(self.hf_home))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(self.hf_home / "hub"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(self.hf_home / "transformers"))
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

        if torch.cuda.is_available():
            self.device = "cuda"
            torch_dtype = torch.float16
        elif torch.backends.mps.is_available():
            self.device = "mps"
            torch_dtype = torch.float16
        else:
            self.device = "cpu"
            torch_dtype = torch.float32

        self.torch_dtype = torch_dtype
        self.model = None
        self.processor = None
        self._model_load_error = None
        self.model_source = self.model_id

        if self.preload_model_on_init:
            self._prepare_local_model_files()
        if self.preload_model_in_memory_on_init:
            try:
                self._ensure_model_loaded()
            except Exception:
                # Keep startup resilient; generation will return a clear error message.
                pass

    def _prepare_local_model_files(self):
        """
        Download model files to a local project folder and load from there.
        """
        try:
            # If already downloaded, use it directly.
            if (self.local_model_dir / "config.json").exists():
                self.model_source = str(self.local_model_dir)
                return

            self.local_model_dir.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=self.model_id,
                local_dir=str(self.local_model_dir),
                local_dir_use_symlinks=False,
                token=self.api_key,
            )
            self.model_source = str(self.local_model_dir)
        except Exception:
            # Fall back to repo ID; lazy load will still report detailed error if needed.
            self.model_source = self.model_id

    def _ensure_model_loaded(self):
        """
        Lazily load model artifacts so app startup does not fail on network/auth issues.
        """
        if self.model is not None:
            return
        if self._model_load_error:
            raise RuntimeError(self._model_load_error)

        try:
            load_from_local = Path(str(self.model_source)).exists()
            self.processor = AutoProcessor.from_pretrained(
                self.model_source,
                token=self.api_key,
                trust_remote_code=True,
                local_files_only=load_from_local,
            )
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_source,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True,
                token=self.api_key,
                local_files_only=load_from_local,
            ).to(self.device)
            self.model.eval()
        except Exception as e:
            self._model_load_error = f"Failed to load vision-language model '{self.model_id}' from '{self.model_source}': {e}"
            raise RuntimeError(self._model_load_error) from e

    @staticmethod
    def _decode_base64_image(encoded_image):
        """
        Convert base64 string (raw or data URL) into a PIL image.
        """
        if not encoded_image:
            return None
        payload = encoded_image.split(",", 1)[-1]
        image_bytes = base64.b64decode(payload)
        return Image.open(BytesIO(image_bytes)).convert("RGB")
    
    def generate_response(self, encoded_image, prompt):
        """
        Generate a response from the model based on a prompt.
        
        Args:
            encoded_image (str): Base64-encoded image string
            prompt (str): Text prompt to guide the model's response
            
        Returns:
            str: Model's response
        """
        try:
            self._ensure_model_loaded()
            image = self._decode_base64_image(encoded_image)
            if image is None:
                return "Error generating response: Missing image input."

            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful fashion assistant."}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.processor(text=[text], images=[image], return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.temperature > 0,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )

            input_len = inputs["input_ids"].shape[1]
            new_tokens = generated_ids[:, input_len:]
            answer = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
            return answer if answer else "No response generated."
        except Exception as e:
            return f"Error generating response: {e}"
    
    def generate_fashion_response(self, user_image_base64, matched_row, all_items, 
                                 similarity_score, threshold=0.8):
        """
        Generate a fashion-specific response using role-based prompts.
        
        Args:
            user_image_base64: Base64-encoded user-uploaded image
            matched_row: The closest match row from the dataset
            all_items: DataFrame with all items related to the matched image
            similarity_score: Similarity score between user and matched images
            threshold: Minimum similarity for considering an exact match
            
        Returns:
            str: Detailed fashion response
        """
        # Generate a detailed list of items with prices and links
        items_description = "\n".join(
            f"- **{row['Item Name']}** (${row['Price']}): Buy it here: {row['Link']}"
            for _, row in all_items.iterrows()
        )
        item_names = ", ".join(all_items["Item Name"].astype(str).tolist())
        ordered_item_names = all_items["Item Name"].astype(str).tolist()

        if similarity_score >= threshold:
            # Role-based prompt for an exact match
            assistant_prompt = (
                f"You are a young and enthusiastic fashion expert who helps students learn about fashion analysis. "
                f"The matched outfit features: {matched_row['Item Name']}.\n\n"
                f"Matched outfit includes:\n{items_description}\n\n"
                f"Valid item names to reference: {item_names}\n"
                "Important: Never use the placeholder phrase 'Item Name'. Always use the real item name exactly as listed.\n\n"
                "Follow these steps in your response:\n"
                "1. Introduce yourself briefly. Use an educational, engaging tone suited for students.\n"
                "2. Describe each item as a definition first (list what it is, what color it is and what pattern it has), following this format:\n"
                "   - Example: Versace 'Tweed Masculine Blazer' is a single-breasted wool-blend tweed blazer crafted in a micro windowpane pattern in red and black.\n"
                "   - Format: **<actual item name>** is [definition].\n"
                "3. After defining each item, include its detailed description, highlighting its type, material, pattern, and why it stands out.\n"
                "4. Describe the outfit's overall style category and explain why (e.g., casual chic, formal elegance, street style).\n"
                "5. Include a brief learning point about fashion analysis concepts used in this assessment.\n"
                "6. Summarize all the items with their prices and links at the end.\n\n"
                "Ensure your response is educational, clear, and structured for students to learn from!"
            )
        else:
            # Role-based prompt for the closest match
            assistant_prompt = (
                f"You are a fashion instructor helping students learn image analysis techniques.\n\n"
                f"Candidate item names to reference: {item_names}\n"
                "Important: Never use the placeholder phrase 'Item Name'. Always use the real item name exactly as listed.\n\n"
                "Follow these steps in your response:\n"
                "1. Use an educational tone suited for a course on fashion technology.\n"
                "2. Explain that while we don't have this exact outfit in our database, this is a learning opportunity about image analysis.\n"
                "3. Describe each item as a definition first (list what it is, what color it is and what pattern it has), following this format: \n"
                "   - Example: Blazer is a single-breasted wool-blend tweed blazer in red and black with a micro windowpane pattern.\n"
                "   - Format: **<actual item name>** is [definition].\n"
                "4. Be thorough and include details about the type of item, its color, and any patterns or textures.\n"
                "5. Include a short lesson on how AI models analyze fashion elements in images.\n"
                "6. Conclude by saying: 'Next, we'll search for similar items online to demonstrate how to recreate this look.'\n\n"
                "Make your response educational and structured for a classroom setting!"
            )

        # Send the prompt to the model
        response = self.generate_response(user_image_base64, assistant_prompt)

        # Fallback cleanup if model still outputs placeholder labels.
        if "Item Name is" in response:
            for name in ordered_item_names:
                if "Item Name is" not in response:
                    break
                response = response.replace("Item Name is", f"{name} is", 1)
            response = response.replace("Item Name is", "This item is")

        return response
