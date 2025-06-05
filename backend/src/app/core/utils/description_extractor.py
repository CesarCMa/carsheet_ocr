import numpy as np
import re


def normalize_code(code):
    """
    Normalize a code by:
    1. Removing dots
    2. Replacing E with F for E + 1 or 2 digit numbers
    3. Replacing 0 with o for 0 + 1 or 2 digit numbers
    """
    code = code.replace('.', '')
    
    if code.startswith('e') and len(code) in [2, 3]:
        code = 'f' + code[1:]
    
    if code.startswith('0') and len(code) in [2, 3]:
        code = 'o' + code[1:]
    
    return code

def convert_coords_to_int(coords):
    """
    Convert numpy coordinates to regular Python integers.
    
    Parameters:
    coords -- List of coordinate points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    
    Returns:
    List of coordinate points with regular Python integers
    """
    if coords is None:
        return None
    return [[int(point[0]), int(point[1])] for point in coords]

def find_descriptions(detected_text_boxes, sheet_codes_df, max_distance=None):
    """
    Find descriptions for the given codes based on proximity in the OCR results.
    
    Parameters:
    detected_text_boxes -- List of detected text boxes with format:
                          [([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ['text', confidence])]
    sheet_codes_df -- DataFrame containing the codes and their descriptions
    max_distance -- Maximum horizontal distance (in pixels) to consider when looking for descriptions.
                   If None, no distance limit is applied.
    
    Returns:
    Dictionary mapping codes to a dict containing pred_index, description, code_coords, desc_coords, and code_name
    """
    # Parameters to adjust
    Y_TOLERANCE = 15  # Maximum vertical difference (in pixels) to consider boxes as being in the same row
    
    # Initialize result dictionary
    code_descriptions = {}
    
    # Process each code
    for _, row in sheet_codes_df.iterrows():
        target_code = row['code']
        target_code_lower = target_code.lower()
        target_code_normalized = normalize_code(target_code_lower)
        code_box_index = None
        
        # Look for this code in the detected text boxes
        for i, box in enumerate(detected_text_boxes):
            text = box[1][0].lower()  # Get text and convert to lowercase
            text_normalized = normalize_code(text)
            
            if text_normalized == target_code_normalized:
                code_box_index = i
                break
        
        # If code was not found, add None to dictionary and continue to next code
        if code_box_index is None:
            code_descriptions[target_code] = {
                "pred_index": None,
                "description": None,
                "code_coords": None,
                "desc_coords": None,
                "code_name": row['description'] if 'description' in row else None
            }
            continue
        
        # Code was found, now find the closest text box to its right
        code_box = detected_text_boxes[code_box_index]
        code_coords = code_box[0]
        
        # Calculate center point of the code box
        code_center_x = sum(point[0] for point in code_coords) / 4
        code_center_y = sum(point[1] for point in code_coords) / 4
        
        closest_desc = None
        closest_desc_index = None
        closest_desc_coords = None
        min_distance = float('inf')
        
        # Check each text box to find the closest one to the right
        for i, box in enumerate(detected_text_boxes):
            # Skip if it's the same box as the code
            if i == code_box_index:
                continue
                
            box_coords = box[0]
            box_text = box[1][0]
            
            # Calculate center point of this box
            box_center_x = sum(point[0] for point in box_coords) / 4
            box_center_y = sum(point[1] for point in box_coords) / 4
            
            # Check if this box is to the right of the code box
            if box_center_x > code_center_x:
                # Check if it's at approximately the same y-level (same row)
                # Allow some flexibility in y-coordinate
                y_difference = abs(box_center_y - code_center_y)
                if y_difference < Y_TOLERANCE:
                    # Calculate horizontal distance
                    distance = box_center_x - code_center_x
                    
                    if max_distance is not None and distance > max_distance:
                        continue
                    
                    # Update closest description if this one is closer
                    if distance < min_distance:
                        min_distance = distance
                        closest_desc = box_text
                        closest_desc_index = i
                        closest_desc_coords = box_coords
        
        # Add to result dictionary
        code_descriptions[target_code] = {
            "pred_index": closest_desc_index,
            "description": closest_desc,
            "code_coords": convert_coords_to_int(code_coords),
            "desc_coords": convert_coords_to_int(closest_desc_coords),
            "code_name": row['description'] if 'description' in row else None
        }
    
    return {k.upper(): v for k, v in code_descriptions.items()}

def plate_extractor(detected_text_boxes):
    """
    Extract the plate number from the detected text boxes.
    Looks for a pattern of 4 numbers followed by 3 letters.
    
    Parameters:
    detected_text_boxes -- List of detected text boxes with format:
                          [([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ['text', confidence])]
    
    Returns:
    Dictionary containing the plate number and its coordinates if found, empty dict otherwise
    """
    plate_pattern = re.compile(r'^\d{4}[A-Za-z]{3}$')
    
    for box in detected_text_boxes:
        text = box[1][0].strip()  # Get text and remove whitespace
        
        if plate_pattern.match(text):
            return {
                "matricula": {
                    "description": text.upper(),
                    "desc_coords": convert_coords_to_int(box[0]),
                    "code_name": "matricula"
                }
            }
    
    return {}

def extract_certificate_code(detected_text_boxes):
    """
    Extract the certificate code from the detected text boxes.
    Looks for text containing 'certificado' and then finds the closest text box
    diagonally to the low-right of that box.
    
    Parameters:
    detected_text_boxes -- List of detected text boxes with format:
                          [([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ['text', confidence])]
    
    Returns:
    Dictionary containing the certificate code and its coordinates if found, empty dict otherwise
    """
    # First find the box containing "certificado"
    certificado_box = None
    for box in detected_text_boxes:
        text = box[1][0].lower().strip()
        if "certificado" in text:
            certificado_box = box
            break
    
    if not certificado_box:
        return {}
    
    # Get the low-right corner coordinates of the certificado box
    certificado_coords = certificado_box[0]
    low_right_x = max(point[0] for point in certificado_coords)
    low_right_y = max(point[1] for point in certificado_coords)
    
    # Find the closest text box diagonally to the low-right
    closest_box = None
    min_distance = float('inf')
    
    for box in detected_text_boxes:
        # Skip the certificado box itself
        if box == certificado_box:
            continue
            
        box_coords = box[0]
        box_center_x = sum(point[0] for point in box_coords) / 4
        box_center_y = sum(point[1] for point in box_coords) / 4
        
        # Only consider boxes that are to the right and below the low-right corner
        if box_center_x > low_right_x and box_center_y > low_right_y:
            # Calculate diagonal distance using Euclidean distance
            distance = ((box_center_x - low_right_x) ** 2 + (box_center_y - low_right_y) ** 2) ** 0.5
            
            if distance < min_distance:
                min_distance = distance
                closest_box = box
    
    if closest_box:
        return {
            "certificado": {
                "description": closest_box[1][0].strip(),
                "desc_coords": convert_coords_to_int(closest_box[0]),
                "code_name": "certificado"
            }
        }
    
    return {}
