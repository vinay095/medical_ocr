import google.generativeai as genai
import PIL.Image
import json
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import tempfile
from dotenv import load_dotenv


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
print(f'api = {api_key}')

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")

genai.configure(api_key=api_key)

try:
    df_medicines = pd.read_csv('medicine_data.csv')
    # Clean column names to prevent issues with extra spaces
    df_medicines.columns = df_medicines.columns.str.strip()
    print("INFO: Medicine dataset loaded successfully.")
except FileNotFoundError:
    print("WARNING:  'medicine_data.csv' not found. The lookup feature will be disabled.")
    df_medicines = None

app = FastAPI()

# ADD CORS MIDDLEWARE 
origins = ["*"] # For development. For production, change "*" to your website's domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HELPER FUNCTION

def find_medicine_details(medicine_name: str, active_salts: list):
    """
    Searches the DataFrame for a medicine, first by name, then by active salts.
    """
    if df_medicines is None or df_medicines.empty:
        return None

    # find a match using the medicine name
    if medicine_name:
        name_result = df_medicines[df_medicines['Medicine Name'].str.contains(medicine_name, case=False, na=False)]
        if not name_result.empty:
            first_match = name_result.iloc[0]
            return {
                "uses": first_match.get("Uses"),
                "side_effects": first_match.get("Side_effects")
            }

    # fallback to searching by active salts.
    if active_salts:
        for salt in active_salts:
            # Clean up the salt name (e.g., remove "(500mg)") before searching
            cleaned_salt = salt.split('(')[0].strip()
            if not cleaned_salt:
                continue
            
            salt_result = df_medicines[df_medicines['Composition'].str.contains(cleaned_salt, case=False, na=False)]
            if not salt_result.empty:
                first_match = salt_result.iloc[0]
                return {
                    "uses": first_match.get("Uses"),
                    "side_effects": first_match.get("Side_effects")
                }
    return None

# def find_medicine_details(name: str):

#     if df_medicines is None or df_medicines.empty:
#         return None
    
#     # .str.contains() for a more flexible, case-insensitive search
#     # This helps if the OCR result is slightly different from the name in the CSV
#     result = df_medicines[df_medicines['Medicine Name'].str.contains(name, case=False, na=False)]
    
#     if not result.empty:
#         # Return the first match as a dictionary
#         first_match = result.iloc[0]
#         return {
#             "uses": first_match.get("Uses"),
#             "side_effects": first_match.get("Side_effects")
#         }
#     return None


def extract_data_from_image(image_path: str) -> dict:

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        image = PIL.Image.open(image_path)
        
        prompt = """
            Analyze the image of this medicine label. Extract the following information and return it as a clean JSON object.
            Do not include any introductory text or markdown formatting like ```json.
            
            The keys in the JSON should be:
            - "medicine_name"
            - "manufacturer"
            - "active_salts" (as a list of strings)
            - "expiry_date" (in DD-MM-YYYY format if possible, otherwise MM-YYYY)
            - "batch_number"
            
            If a piece of information is not available, set its value to null.
        """
        response = model.generate_content([image, prompt])
        cleaned_text = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_text)

    except json.JSONDecodeError:
        print("Error: Model returned non-JSON text:", response.text)
        return {"error": "Failed to parse model response as JSON."}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {"error": "An internal error occurred during processing."}


# @app.post("/extract-medicine-data/")
# async def extract_medicine_data(file: UploadFile = File(...)):
#     """Accepts an image, runs OCR, looks up data, and returns a combined result."""
#     with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
#         shutil.copyfileobj(file.file, tmp)
#         tmp_path = tmp.name

#     try:
#         # Get the initial data from the image using Gemini
#         extracted_data = extract_data_from_image(tmp_path)
        
#         if "error" in extracted_data:
#             raise HTTPException(status_code=500, detail=extracted_data["error"])
        
#         # look up the medicine name if it was found
#         medicine_name = extracted_data.get("medicine_name")
#         if medicine_name:
#             additional_details = find_medicine_details(medicine_name)
            
#             # Merge the two data sources
#             if additional_details:
#                 extracted_data.update(additional_details)
#             else:
#                 extracted_data["uses"] = "No information found in our database."
#                 extracted_data["side_effects"] = "No information found in our database."
        
#         return {"message": "Data extracted successfully", "data": extracted_data}
#     finally:
#         # clean up the temporary file
#         os.unlink(tmp_path)


@app.post("/extract-medicine-data/")
async def extract_medicine_data(file: UploadFile = File(...)):
    """Accepts an image, runs OCR, looks up data, and returns a combined result."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # data from the image using Gemini
        extracted_data = extract_data_from_image(tmp_path)
        
        if "error" in extracted_data:
            raise HTTPException(status_code=500, detail=extracted_data["error"])
        
        # Get both name and salts for the search
        medicine_name = extracted_data.get("medicine_name")
        active_salts = extracted_data.get("active_salts", []) # Default to empty list
        
        # Call the upgraded search function
        additional_details = find_medicine_details(medicine_name, active_salts)
        
        # Step C: Merge the two data sources
        if additional_details:
            extracted_data.update(additional_details)
        else:
            extracted_data["uses"] = "No information found in our database."
            extracted_data["side_effects"] = "No information found in our database."
        
        return {"message": "Data extracted successfully", "data": extracted_data}
    finally:
        # Clean up the temporary file
        os.unlink(tmp_path)