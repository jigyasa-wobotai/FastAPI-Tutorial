__author__ = "Dhruvi Bhavsar"
__copyright__ = "Copyright 2025, Wobot Intelligence Pvt. Ltd."
__credits__ = ["Dhruvi Bhavsar"]
__license__ = "Proprietary"
__version__ = "0.0.1"
__maintainer__ = "Dhruvi Bhavsar"
__email__ = "dhruvi.bhavsar@wobot.ai"
__status__ = "Development"
__module_name__ = "[Spot Check API]"

import cv2
from PIL import Image
import asyncio
import numpy as np
import datetime
import time
import os
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError
import traceback
import json
from typing import Optional, Dict, Tuple
from tenacity import retry, stop_after_attempt, wait_fixed
from fastapi import FastAPI, Depends, HTTPException, Security, status
from pydantic import BaseModel,Field
from fastapi.middleware.cors import CORSMiddleware
import logging
import base64
from utils.openai_gpt_client import OpenaiGptClient
from utils.gemini_client import GeminiClient
from fastapi.security.api_key import APIKeyHeader
from utils.config_parser import ConfigParser
from utils.roi_utils import crop_polygon_roi, draw_poly, draw_poly_pillow
from concurrent.futures import ThreadPoolExecutor
import threading

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True
)

logger = logging.getLogger(__module_name__)
logger.setLevel(logging.INFO)

API_KEY_NAME = "SPOTCHECK_API_KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
OPENAI_API_KEY=os.environ["OPENAI_API_KEY"]
GOOGLE_API_KEY=os.environ["GOOGLE_API_KEY"]

if not OPENAI_API_KEY or GOOGLE_API_KEY:
    logger.warning("One or more API keys are missing from environment variables.")

NUM_WORKERS=int(os.environ.get("WORKERS","8"))
# task_queue = asyncio.Queue()
executor = None

DEFAULT_MODEL = "gemini-2.5-flash-lite"  # default to Gemini

# Global dictionary to store pre-loaded clients
MODEL_CLIENTS: Dict[str, Tuple[object, str]] = {}

# Cache for dynamically created clients
DYNAMIC_CLIENTS: Dict[str, Tuple[object, str]] = {}

#Hardcoded smartSearch query
HARDCODED_SMARTSEARCH_PROMPT = """Analyze the given surveillance/security image and return output strictly in the EventModel JSON schema. humans: For each detected human provide gender, age group, hair color, actions, facing direction (towards/away/left/right), wearables (type, color, attributes), carried accessories, and spatial relations to other humans. objects: Include all non-fixed objects only (ignore fixed/intentional restaurant items); for each provide name, attributes, associated verbs, and position/location. scene: Describe environment type (indoor/outdoor), lighting, crowd density, weather (if visible), and overall spatial/contextual details. task: Always set type from image and briefly describe what is being analyzed in the image."""
HARDCODED_SMARTSEARCH_CONFIG = {
    "prompt": [HARDCODED_SMARTSEARCH_PROMPT],
    "violationClass": [],
    "vboundingBoxType": "default",
    "type": "smartSearch",
    "insightClasses": ["humans", "objects", "scene", "task"]
}

# Lock for thread-safe dynamic client creation
CLIENT_LOCK = threading.Lock()

def initialize_model_clients():
    """
    Initialize common/popular model clients at startup.
    This prevents creating new clients for each request for frequently used models.
    """
    global MODEL_CLIENTS
    
    # Define commonly used models for pre-initialization
    common_models = {
        # Popular OpenAI models
        "gpt-4.1-mini": ("openai", OpenaiGptClient),
        
        # Popular Gemini models
        # "gemini-1.5-flash-8b": ("gemini", GeminiClient),
        # "gemini-1.5-flash": ("gemini", GeminiClient),
        # "gemini-2.0-flash-lite": ("gemini", GeminiClient),
        # "gemini-2.0-flash": ("gemini", GeminiClient),
        "gemini-2.5-flash-lite": ("gemini", GeminiClient),
        "gemini-2.5-flash": ("gemini", GeminiClient),
    }
    
    logger.info(f"{__module_name__}: Initializing common model clients...")
    
    for model_name, (client_type, client_class) in common_models.items():
        try:
            if client_type == "openai":
                client_instance = client_class(api_key=OPENAI_API_KEY, model_name=model_name)
            elif client_type == "gemini":
                client_instance = client_class(api_key=GOOGLE_API_KEY, model_name=model_name)
            else:
                continue
                
            MODEL_CLIENTS[model_name] = (client_instance, client_type)
            logger.info(f"{__module_name__}: Initialized {client_type} client for model: {model_name}")
            
        except Exception as e:
            logger.error(f"{__module_name__}: Failed to initialize client for {model_name}: {e}")
    
    logger.info(f"{__module_name__}: Successfully initialized {len(MODEL_CLIENTS)} common model clients")

def is_supported_model(model_name: str) -> bool:
    """
    Check if a model name is supported (gpt-* or gemini-* pattern).
    
    Args:
        model_name (str): The model name to check
        
    Returns:
        bool: True if the model is supported
    """
    if not model_name:
        return False
    
    model_lower = model_name.lower()
    return model_lower.startswith('gpt-') or model_lower.startswith('gemini-')

def determine_client_type(model_name: str) -> str:
    """
    Determine the client type based on model name.
    
    Args:
        model_name (str): The model name
        
    Returns:
        str: 'openai' or 'gemini'
    """
    model_lower = model_name.lower()
    if model_lower.startswith('gpt-'):
        return 'openai'
    elif model_lower.startswith('gemini-'):
        return 'gemini'
    else:
        raise ValueError(f"Unsupported model prefix: {model_name}")

def create_dynamic_client(model_name: str) -> Tuple[object, str]:
    """
    Create a new client instance dynamically for supported models.
    
    Args:
        model_name (str): The model name to create client for
        
    Returns:
        Tuple of (client_instance, client_type)
        
    Raises:
        ValueError: If model is not supported
    """
    if not is_supported_model(model_name):
        raise ValueError(f"Unsupported model: {model_name}. Only gpt-* and gemini-* models are supported.")
    
    client_type = determine_client_type(model_name)
    
    try:
        if client_type == "openai":
            client_instance = OpenaiGptClient(api_key=OPENAI_API_KEY, model_name=model_name)
        elif client_type == "gemini":
            client_instance = GeminiClient(api_key=GOOGLE_API_KEY, model_name=model_name)
        else:
            raise ValueError(f"Unknown client type: {client_type}")
        
        logger.info(f"{__module_name__}: Created dynamic {client_type} client for model: {model_name}")
        return client_instance, client_type
        
    except Exception as e:
        logger.error(f"{__module_name__}: Failed to create dynamic client for {model_name}: {e}")
        raise

def get_model_client(model_name: str) -> Tuple[object, str]:
    """
    Get client instance based on model name.
    First checks pre-loaded clients, then creates dynamic clients for supported models.
    
    Args:
        model_name (str): The model name to get client for
        
    Returns:
        Tuple of (client_instance, client_type)
        
    Raises:
        ValueError: If model is not supported
    """
    actual_model = model_name or DEFAULT_MODEL
    
    # Check pre-initialized clients first
    if actual_model in MODEL_CLIENTS:
        client_instance, client_type = MODEL_CLIENTS[actual_model]
        logger.info(f"{__module_name__}: Using pre-loaded {client_type} client for model: {actual_model}")
        return client_instance, client_type
    
    # NOTE: Locked DYNAMIC_CLIENTS cache access and creation to ensure thread safety
    with CLIENT_LOCK:

        # Check dynamically created clients cache
        if actual_model in DYNAMIC_CLIENTS:
            client_instance, client_type = DYNAMIC_CLIENTS[actual_model]
            logger.info(f"{__module_name__}: Using cached dynamic {client_type} client for model: {actual_model}")
            return client_instance, client_type
        
        # Check if model is supported for dynamic creation
        if is_supported_model(actual_model):
            # Create new dynamic client
            try:
                client_instance, client_type = create_dynamic_client(actual_model)
                # Cache the dynamically created client for future use
                DYNAMIC_CLIENTS[actual_model] = (client_instance, client_type)
                logger.info(f"{__module_name__}: Cached new dynamic {client_type} client for model: {actual_model}")
                return client_instance, client_type
            except Exception as e:
                logger.error(f"{__module_name__}: Failed to create dynamic client for {actual_model}: {e}")
                raise ValueError(f"Failed to create client for model {actual_model}: {e}")
    
        # If model is not supported, provide helpful error message
        available_preloaded = list(MODEL_CLIENTS.keys())
        available_dynamic = list(DYNAMIC_CLIENTS.keys())
    raise ValueError(
        f"Unsupported model: {actual_model}. "
        f"Supported patterns: gpt-*, gemini-*. "
        f"Pre-loaded models: {available_preloaded}. "
        f"Dynamic models in cache: {available_dynamic}"
    )


def print_api_trunc(key):
    if key and len(key) >= 6:
        masked_key = key[2:6] + "*" * (len(key) - 6)
    else:
        masked_key = "Too Short"
    return masked_key

VALID_API_KEY = os.getenv("SPOTCHECK_API_KEY", "").strip()

def validate_api_key(api_key: str = Security(api_key_header)):
    
    logger.info(f"{__module_name__}: Expected API Key: {'SET' if VALID_API_KEY else 'NOT SET'}")
    logger.info(f"{__module_name__}: Received API Key: {print_api_trunc(api_key) if api_key else 'None'}")

    if not VALID_API_KEY:
        logger.error(f"{__module_name__}: API Key is not set in environment variables")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API Key not configured"
        )

    if not api_key or api_key.strip() != VALID_API_KEY:
        logger.error(f"{__module_name__}: Invalid API Key provided")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key"
        )
    return api_key

class Spotcheck(BaseModel):
    image: str
    id: str
    query: list
    model_name: Optional[str] = Field(None, description="Optional model name, if not provided use default")  # Optional field
    roi : Optional[list] = [{"cords":[]}]
    vBoundingBox: Optional[dict] = Field(default_factory=dict, description="Bounding box polygons by track_id")
    taskEnv : Optional[str] = None
    
app = FastAPI()

origins = ["*"]
logger.info(f"{__module_name__}:origin allowed")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
   allow_methods=["*"],
   allow_headers=["*"],
)

parser = ConfigParser()

def read_image_from_source(image_src, retries=10, backoff_factor=1):
    try:
        if image_src.startswith("https://") or image_src.startswith("http://"):

            for attempt in range(retries):
                try:
                    response = requests.get(image_src, timeout=5)
                    if response.status_code == 200:
                        image_array = np.frombuffer(response.content, np.uint8)
                        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                        del image_array
                        return image, response.status_code
                    #Check for wrong URL or image URL not accessible
                    elif response.status_code==403:
                        raise HTTPException(
                            status_code=403,
                            detail="Image not accessible. Check URL"
                        )
                    elif response.status_code==404:
                        raise HTTPException(
                            status_code=404,
                            detail="Image not found. Check URL"
                        )
                    else:
                        logger.warning(f"{__name__}:{attempt + 1}-Received status code {response.status_code} from URL {image_src}")
                        time.sleep(backoff_factor * (2 ** attempt))
                except requests.RequestException as e:
                    logger.warning(f"{__name__}: Attempt {attempt + 1} failed: {e}")
                    time.sleep(backoff_factor * (2 ** attempt))

            logger.error(f"{__name__}: Failed to fetch image after {retries} attempts.")
            return None, None

        elif image_src.endswith((".jpg", ".jpeg", ".png")):
            try:
                image_array = cv2.imread(image_src, cv2.IMREAD_COLOR)
                if image_array is None:
                    logger.error(f"Failed to read image from local path: {image_src}")
                    return None, None
                _, buffer = cv2.imencode(".jpg", image_array)
                del image_array
                return buffer.tobytes(), 200
            except Exception as e:
                logger.error(f"{__name__}: Failed to read local image: {e}")
                logger.error(f"{__name__}:\n {traceback.format_exc()}")
                raise

        else:
            try:
                decoded_data = base64.b64decode(image_src)
                np_data = np.frombuffer(decoded_data, np.uint8)
                image = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
                return image, 200
            except base64.binascii.Error as e:
                logger.error(f"{__name__}: Invalid base64 encoded string: {e}")
                return None, 400
            except Exception as e:
                logger.error(f"{__name__}: Failed to decode base64 image: {e}")
                logger.error(f"{__name__}:\n {traceback.format_exc()}")
                raise

    except Exception as e:
        logger.error(f"{__name__}: Failed to process image from {image_src}: {e}")
        raise

def preprocess_image(img_input):
    image_data, status_code = read_image_from_source(img_input)

    #If image data is empty
    if image_data is None:
        raise HTTPException(
            status_code=422,
            detail="Preprocessing failed. Image data is empty"
        )

    if isinstance(image_data, bytes):
        # Case: Local image file was read as bytes
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        del image_array
    elif isinstance(image_data, np.ndarray):
        # Case: HTTP/HTTPS URL or base64 string was decoded to a NumPy array
        image = image_data
    else:
        logger.error(f"{__name__}: Unexpected image data type: {type(image_data)}")
        return None, 400

    if image is None:
        return None, 400

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    return pil_image, 200

def check_violation_triton(frame, search_query, model_name, roi, vBoundingBox, mode="v1"):

    if frame is None:
        logger.error(f"{__module_name__}: Received None for frame in check_violation_triton.")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="The image could not be processed or is empty. Please check the image source/URL."
        )

    """
    Modified function to instantiate client dynamically based on model_name
    """
    response = []
    responseObj = {}
    total_token_count = 'N/A'

    try:
        # Automatically append hardcoded smartSearch query
        enhanced_query = search_query.copy()
        enhanced_query.append(HARDCODED_SMARTSEARCH_CONFIG)

        # Get client instance (pre-loaded or dynamically created)
        client, client_type = get_model_client(model_name)
        
        if client_type == "openai":
            total_token_count = 0
            for idx, query in enumerate(search_query):
                frame_processed = process_frame_with_bbox_and_roi(frame, roi, query, vBoundingBox)
                pred_cls, token_count = client.analyze_image(frame_processed, UserPrompt=query["prompt"][0])
                
                violation_class_list = query.get("violationClass", [])
                if isinstance(violation_class_list, list) and len(violation_class_list) > 0:
                    isviolation = True
                    violationCls = violation_class_list[0]
                else:
                    isviolation = False
                    violationCls = ""

                logger.info(
                    f"Predicted class: '{pred_cls}' (type: {type(pred_cls).__name__}), "
                    f"Violation class: '{violationCls}' (type: {type(violationCls).__name__}), "
                    f"Match (case/whitespace insensitive): {str(pred_cls).strip().lower() == str(violationCls).strip().lower()}"
                )

                violation = False
                if violationCls:
                    violation = str(pred_cls).lower().strip() == str(violationCls).lower().strip()

                prompt_type = query.get("type","")

                response.append({
                    'isViolation': violation, 
                    'type': prompt_type, 
                    'metadata': {
                        "predicted_class": pred_cls,
                        "predicted_score": 1,
                        "modelUsed": model_name,
                        "tokenCount": token_count
                    }
                })
                total_token_count += token_count

        elif client_type == "gemini":
            total_token_count = 0
            for idx, query in enumerate(enhanced_query):
                frame_processed = process_frame_with_bbox_and_roi(frame, roi, query,vBoundingBox)
                insight_classes = query.get("insightClasses", [])
                prompt_type = query.get("type","")

                # Call Gemini with prompt_type to determine response format
                pred_cls_str, prompt_tokens, candidates_tokens, thoughts_tokens, total_tokens = client.analyze_image(
                    frame_processed, 
                    UserPrompt=query["prompt"],
                    insight_classes=insight_classes,
                    prompt_type=prompt_type,
                    mode=mode
                )

                if pred_cls_str is None:
                    logger.error(f"Failed to get response from Gemini for query {idx}")
                    continue
                    
                violation_class_list = query.get("violationClass", [])
                
                # Handle smartSearch with JSON response
                if prompt_type == "smartSearch":
                    responseObj['model_output'] = json.loads(pred_cls_str)
                    total_token_count += total_tokens
                else:
                    # Handle Scout AI with plain text response
                    if isinstance(violation_class_list, list) and len(violation_class_list) > 0:
                        isviolation = True
                        violationCls = violation_class_list
                    else:
                        isviolation = False
                        violationCls = []

                    if mode == "v1":
                        pred_cls, description = generate_response(pred_cls_str)
                        pred_cls = [word.strip() for word in pred_cls.split(";") if word.strip()]
                    elif mode == "v2":
                        pred_cls, description = generate_response_v2(pred_cls_str)

                    if isviolation:
                        violation = has_common_value(pred_cls, violationCls)
                    else:
                        violation = False

                    responseObj.update({
                        'isViolation': violation, 
                        'type': prompt_type, 
                        'metadata': {
                            "label": pred_cls, 
                            "description": description, 
                            "predicted_score": 1, 
                            "processed": True, 
                            "modelUsed": model_name, 
                            "promptTokens": prompt_tokens, 
                            "completionTokens": candidates_tokens, 
                            "thoughtsTokens": thoughts_tokens, 
                            "totalTokens": total_tokens
                        }
                    })
                    total_token_count += total_tokens
                    
                # responseObj['metadata']['totalTokens'] = total_token_count

            # Return combined response
            response.append(responseObj)

    except ValueError as ve:
        logger.error(f"{__module_name__}: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"{__module_name__}: Error in check_violation_triton: {e}")
        raise

    return response, total_token_count

def has_common_value(arr1, arr2):
    set1 = set(val.lower() for val in arr1)
    set2 = set(val.lower() for val in arr2)
    # logger.info(f"{'*' * 200}")
    # logger.info(set1)
    # logger.info(f"{'*' * 200}")
    # logger.info(set2)
    # logger.info(f"{'*' * 200}")
    return not set1.isdisjoint(set2)

def process_frame_with_bbox_and_roi(frame, roi, query, vBoundingBox):
    """
    Handles frame processing based on vboundingBoxType and ROI logic.
    - If vboundingBoxType == "crop": crop to bbox, ignore ROI.
    - If vboundingBoxType == "draw": draw bbox on full frame, if ROI, crop to ROI
    - If vboundingBoxType == "default" or missing: if ROI, crop to ROI; else return frame.
    """
    vboundingBoxType = query.get("vboundingBoxType", "default")
   # Extract bbox polygons from vBoundingBox (dict of track_id -> coords)
    

    #Check for coords not given in case of draw or crop conditions 
    if vboundingBoxType in ["draw", "crop"]:         
        if not vBoundingBox or not isinstance(vBoundingBox, dict) or len(vBoundingBox) == 0:         
                logger.error(f"vboundingBoxType is '{vboundingBoxType}' but vBoundingBox is empty.") 
                raise HTTPException(                 
                    status_code=status.HTTP_400_BAD_REQUEST,                 
                    detail=f"Incorrect BBox handling: '{vboundingBoxType}' selected but no coordinates provided."   
                                      )
        
        #Check for if internal lists are empty (e.g., {"Zone 1":[]})
        for key, coords in vBoundingBox.items():
            if not coords or len(coords)<3:
                raise HTTPException(
                    status_code=300,
                    detail=f"Invalid coordinates for {key}. A polygon atleast requires 3 points."
                )

        
    bbox_polygons = []
    if isinstance(vBoundingBox, dict):
        for track_id, coords in vBoundingBox.items():
            if isinstance(coords, list) and len(coords) > 0:
                bbox_polygons.append({"track_id": track_id, "coords": coords})

    # Collect ROI polygons
    roi_polygons = []
    if roi and isinstance(roi, list) and len(roi) > 0:
        for r in roi:
            if isinstance(r, dict) and r.get("cords"):
                roi_polygons.append(r["cords"])
    
    # Case 1: bbox crop takes priority
    if vboundingBoxType == "crop":
        crops = []
        if bbox_polygons:  # crop all bounding boxes
            for bbox in bbox_polygons:
                crops.append(crop_polygon_roi(frame, bbox["coords"]))
        elif roi_polygons:  # if no bbox, crop all ROIs
            for roi_coords in roi_polygons:
                crops.append(crop_polygon_roi(frame, roi_coords))
        if crops:
            return crops
        return [frame]

    # Case 2: bbox draw, ROI considered
    if vboundingBoxType == "draw" and bbox_polygons:
        # Step 1: draw bboxes on full frame
        drawn_frame = frame.copy()
        for bbox in bbox_polygons:
            drawn_frame = draw_poly_pillow(drawn_frame, bbox["coords"])

        # Step 2: crop ROI(s) if present
        # if roi_polygons:
        #     crops = []
        #     for roi_coords in roi_polygons:
        #         crops.append(crop_polygon_roi(drawn_frame, roi_coords))
        #     return crops

        # return [drawn_frame]

    # Case 3 or 4: No bbox action, only ROI or no ROI
    if vboundingBoxType == "default":
        if roi_polygons:  # crop all ROI(s)
            crops = []
            for roi_coords in roi_polygons:
                crops.append(crop_polygon_roi(frame, roi_coords))
            return crops
        logger.warning(f"{__module_name__} No ROI or bounding box applied; returning original frame.")
        return [frame]

def generate_response(model_output:str):

    label_opt = ""
    description = ""

    #Check for empty output returned
    if not model_output or len(model_output.strip()) == 0 or model_output.strip() in [".","null","None"]:
        logger.error(f"Nothing returned by model.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The AI model failed to generate a response. Please try again."
        )

    output = model_output.split("|")
    if len(output) == 1:
        label_opt =  output[0]
    else:
        label_opt , description = output
    label_opt =  label_opt.strip()
    description =  description.strip()

    #Check for if label is more than 40 chars it throws an error
    if len(label_opt) >= 40:
        logger.error(f"{__module_name__}: Label length ({len(label_opt)}) exceeds dashboard limit.")
        raise HTTPException(
            status_code= status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Label is too long ({len(label_opt)} characters). Dashboard limit is 40."
        )

    return label_opt , description

def generate_response_v2(model_output_json:str):

    label = []
    description = ""

    try:

        #Check for empty model output
        if not model_output_json or len(model_output_json.strip()) == 0 or model_output_json.strip() in [".","null","None"]:
            logger.error(f"{__module_name__} : Nothing returned by model.")
            raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The AI model failed to generate a response. Please try again."
        )
    
        # Parse JSON response
        event_data = json.loads(model_output_json)
        
        # Extract label
        label = event_data.get("label", [])

        #Check for if label is more than 40 chars it throws an error
        if len(label) >= 40:
          logger.error(f"{__module_name__}: Label length ({len(label)}) exceeds dashboard limit.")
          raise HTTPException(
            status_code= status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Label is too long ({len(label)} characters). Dashboard limit is 40."
        )

        
        # Extract description
        description = event_data.get("description", "")

        return label, description

    except json.JSONDecodeError as e:
        logger.error(f"{__module_name__}: Failed to parse JSON output: {e}")
        logger.error(f"{__module_name__}: Raw output: {model_output_json}")
        return [], ""
    except Exception as e:
        logger.error(f"{__module_name__}: Error in generate_response_v2: {e}")
        logger.error(f"{__module_name__}:\n {traceback.format_exc()}")
        return [], ""

# @app.post("/spotcheck/")
def spot_check_main(spotcheck: Spotcheck,  api_key: str = Depends(validate_api_key), mode: str = "v1"):

    start_time = time.time()
    img_url = spotcheck.image
    search_query = spotcheck.query


    #Check for empty prompt or in case no prompt given
    first_prompt = search_query[0].get("prompt")     
    if not first_prompt or (isinstance(first_prompt, list) and not first_prompt[0].strip()):   
        raise HTTPException(             
            status_code=status.HTTP_400_BAD_REQUEST,             
            detail="No prompt has been given. The prompt string cannot be empty."    
                    )

    location = spotcheck.id
    model_name = spotcheck.model_name or DEFAULT_MODEL
    roi = spotcheck.roi
    vBoundingBox = spotcheck.vBoundingBox if hasattr(spotcheck, "vBoundingBox") else []
    taskEnv = spotcheck.taskEnv or "unknown"

    logger.info(f"{__module_name__}:Authentication Successful for {print_api_trunc(api_key)}")
    logger.info(f"{__module_name__}:Input Payload: {spotcheck.json()}")

    try:
        img_pro_start = time.time()
        frame, status_code = preprocess_image(img_url)
        if status_code != 200:
            logger.error(f"Failed to process frame")
            raise HTTPException(status_code=status_code, detail=f"failed to process frame, error code {status_code} ")
        preprocess_time = time.time() - img_pro_start

        #cv2.imwrite("sample.jpg", np.array(frame))
        start_infer_timer = time.time()
        response, total_token_count = check_violation_triton(frame,search_query,model_name,roi, vBoundingBox, mode=mode)

        infer_time = time.time()-start_infer_timer
        total_time = time.time()-start_time
        logger.info(f"{__module_name__} Preprocess Time: {preprocess_time:.2f}, Infer Time: {infer_time:.2f}. Total Time: {total_time:.2f}")

    except HTTPException as http_exe:
        raise http_exe

    except Exception as e:
        logger.error(f"{__module_name__} Failed to process frame: {e}")
        logger.error(f"{__module_name__}:\n {traceback.format_exc()}")

        raise Exception(f"{__module_name__} Error in processing request: {e}")

    logger.info(f"{__module_name__} ID: {location} | Response: {response}")
    logger.info(f"{__module_name__} Total token count: {total_token_count}")
    # logger.info(f"{__module_name__} Time taken for one response:{end-start}")
    logger.info(f"{__module_name__} request is made using {str(taskEnv)} environment.")

    return response

@app.post("/spotcheck/")
async def spot_check(spotcheck: Spotcheck,api_key: str = Depends(validate_api_key)):
    global executor
    result = await asyncio.get_event_loop().run_in_executor(
        executor, lambda: spot_check_main(spotcheck, api_key)
    )
    return result

@app.post("/spotcheck/v2/")
async def spot_check_v2(spotcheck: Spotcheck, api_key: str = Depends(validate_api_key)):
    global executor
    result = await asyncio.get_event_loop().run_in_executor(
        executor, lambda: spot_check_main(spotcheck, api_key, mode="v2")
    )
    return result

# Start the worker pool
@app.on_event("startup")
async def start_worker_pool():
    global executor
    logger.info(f"Starting {NUM_WORKERS} workers...")
    executor = ThreadPoolExecutor(max_workers=NUM_WORKERS)
    # Initialize common model clients at startup
    initialize_model_clients()

# Cleanup on shutdown for dev/hot-reload mode
# @app.on_event("shutdown")
# async def shutdown_event():
#     global executor, MODEL_CLIENTS, DYNAMIC_CLIENTS
#     if executor:
#         executor.shutdown(wait=True)
#         logger.info("Thread pool executor shut down")
    
#     # Clear model clients
#     MODEL_CLIENTS.clear()
#     DYNAMIC_CLIENTS.clear()
#     logger.info("Model clients cleared")
