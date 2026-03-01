import re
import base64
import os
from io import BytesIO
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, flash
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

app = Flask(__name__)
load_dotenv()

# Hugging Face local model configuration
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
BASE_DIR = Path(__file__).resolve().parent
HF_HOME = Path(os.getenv("HF_HOME", str(BASE_DIR / ".hf_cache")))
LOCAL_MODEL_DIR = Path(
    os.getenv("LOCAL_MODEL_DIR", str(BASE_DIR / ".models" / MODEL_ID.replace("/", "--")))
)
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")

os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_HOME / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_HOME / "transformers"))
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

if torch.cuda.is_available():
    DEVICE = "cuda"
    TORCH_DTYPE = torch.float16
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    TORCH_DTYPE = torch.float16
else:
    DEVICE = "cpu"
    TORCH_DTYPE = torch.float32

processor = None
model = None


def predownload_model():
    """
    Download model artifacts to a local folder so runtime loads from cache.
    """
    if (LOCAL_MODEL_DIR / "config.json").exists():
        return

    LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(LOCAL_MODEL_DIR),
        local_dir_use_symlinks=False,
        token=HF_TOKEN,
    )


def initialize_model():
    """
    Load processor/model from local cache before starting the Flask app.
    """
    global processor, model
    if processor is not None and model is not None:
        return

    predownload_model()
    processor = AutoProcessor.from_pretrained(
        str(LOCAL_MODEL_DIR),
        trust_remote_code=True,
        local_files_only=True,
        token=HF_TOKEN,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        str(LOCAL_MODEL_DIR),
        torch_dtype=TORCH_DTYPE,
        trust_remote_code=True,
        local_files_only=True,
        token=HF_TOKEN,
    ).to(DEVICE)
    model.eval()

def input_image_setup(uploaded_file):
    """
    Encodes the uploaded image file into a base64 string to be used with AI models.

    Parameters:
    - uploaded_file: File-like object uploaded via a file uploader (Streamlit or other frameworks)

    Returns:
    - encoded_image (str): Base64 encoded string of the image data
    """
    # Check if a file has been uploaded
    if uploaded_file is not None:
        # Read the file into bytes
        bytes_data = uploaded_file.read()

        # Encode the image to a base64 string
        encoded_image = base64.b64encode(bytes_data).decode("utf-8")

        return encoded_image
    else:
        raise FileNotFoundError("No file uploaded")
		
def format_response(response_text):
    """
    Formats the model response to display each item on a new line as a list.
    Converts numbered items into HTML `<ul>` and `<li>` format.
    Adds additional HTML elements for better presentation of headings and separate sections.
    """
    # Replace section headers that are bolded with '**' to HTML paragraph tags with bold text
    response_text = re.sub(r"\*\*(.*?)\*\*", r"<p><strong>\1</strong></p>", response_text)

    # Convert bullet points denoted by "*" to HTML list items
    response_text = re.sub(r"(?m)^\s*\*\s(.*)", r"<li>\1</li>", response_text)

    # Wrap list items within <ul> tags for proper HTML structure and indentation
    response_text = re.sub(r"(<li>.*?</li>)+", lambda match: f"<ul>{match.group(0)}</ul>", response_text, flags=re.DOTALL)

    # Ensure that all paragraphs have a line break after them for better separation
    response_text = re.sub(r"</p>(?=<p>)", r"</p><br>", response_text)

    # Ensure the disclaimer and other distinct paragraphs have proper line breaks
    response_text = re.sub(r"(\n|\\n)+", r"<br>", response_text)

    return response_text

def generate_model_response(encoded_image, user_query, assistant_prompt):
    """
    Sends an image and a query to the model and retrieves the description or answer.
    Formats the response using HTML elements for better presentation.
    """
    try:
        if processor is None or model is None:
            initialize_model()

        image_bytes = base64.b64decode(encoded_image.split(",")[-1])
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful nutrition assistant."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": assistant_prompt + "\n\n" + user_query},
                ],
            },
        ]

        chat_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[chat_text], images=[image], return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[1]
        new_tokens = generated_ids[:, input_len:]
        raw_response = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

        # Format the raw response text using the format_response function
        formatted_response = format_response(raw_response)
        return formatted_response
    except Exception as e:
        print(f"Error in generating response: {e}")
        return "<p>An error occurred while generating the response.</p>"
	
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Retrieve user inputs
        user_query = request.form.get("user_query")
        uploaded_file = request.files.get("file")

        if uploaded_file:
            # Process the uploaded image
            encoded_image = input_image_setup(uploaded_file)

            if not encoded_image:
                flash("Error processing the image. Please try again.", "danger")
                return redirect(url_for("index"))

            # Assistant prompt (can be customized)
            assistant_prompt = """
            You are an expert nutritionist. Your task is to analyze the food items displayed in the image and provide a detailed nutritional assessment using the following format:

        1. **Identification**: List each identified food item clearly, one per line.
        2. **Portion Size & Calorie Estimation**: For each identified food item, specify the portion size and provide an estimated number of calories. Use bullet points with the following structure:
        - **[Food Item]**: [Portion Size], [Number of Calories] calories

        Example:
        *   **Salmon**: 6 ounces, 210 calories
        *   **Asparagus**: 3 spears, 25 calories

        3. **Total Calories**: Provide the total number of calories for all food items.

        Example:
        Total Calories: [Number of Calories]

        4. **Nutrient Breakdown**: Include a breakdown of key nutrients such as **Protein**, **Carbohydrates**, **Fats**, **Vitamins**, and **Minerals**. Use bullet points, and for each nutrient provide details about the contribution of each food item.

        Example:
        *   **Protein**: Salmon (35g), Asparagus (3g), Tomatoes (1g) = [Total Protein]

        5. **Health Evaluation**: Evaluate the healthiness of the meal in one paragraph.

        6. **Disclaimer**: Include the following exact text as a disclaimer:

        The nutritional information and calorie estimates provided are approximate and are based on general food data. 
        Actual values may vary depending on factors such as portion size, specific ingredients, preparation methods, and individual variations. 
        For precise dietary advice or medical guidance, consult a qualified nutritionist or healthcare provider.

        Format your response exactly like the template above to ensure consistency.

        """

            # Generate the model's response
            response = generate_model_response(encoded_image, user_query, assistant_prompt)

            # Render the result
            return render_template("index.html", user_query=user_query, response=response)

        else:
            flash("Please upload an image file.", "danger")
            return redirect(url_for("index"))

    return render_template("index.html")

if __name__ == "__main__":
    initialize_model()
    app.run(debug=True, use_reloader=False)
