import numpy as np

# TODO: To mach codes, remove `.` from codes and predictions before matching
# TODO: If prediction is a cofe with E + 1 or two digit number, replace E with F and match with that the code on target codes
# TODO: if code is 0 + 1 or 2 digit number, replace 0 with o and match with that the code on target codes

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

def find_descriptions(detected_text_boxes, target_codes):
    """
    Find descriptions for the given codes based on proximity in the OCR results.
    
    Parameters:
    detected_text_boxes -- List of detected text boxes with format:
                          [([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ['text', confidence])]
    target_codes -- List of codes to find descriptions for
    
    Returns:
    Dictionary mapping codes to their descriptions
    """
    # Parameters to adjust
    Y_TOLERANCE = 15  # Maximum vertical difference (in pixels) to consider boxes as being in the same row
    
    # Initialize result dictionary
    code_descriptions = {}
    
    # Process each code
    for target_code in target_codes:
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
            code_descriptions[target_code] = None
            continue
        
        # Code was found, now find the closest text box to its right
        code_box = detected_text_boxes[code_box_index]
        code_coords = code_box[0]
        
        # Calculate center point of the code box
        code_center_x = sum(point[0] for point in code_coords) / 4
        code_center_y = sum(point[1] for point in code_coords) / 4
        
        closest_desc = None
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
                    
                    # Update closest description if this one is closer
                    if distance < min_distance:
                        min_distance = distance
                        closest_desc = box_text
        
        # Add to result dictionary
        code_descriptions[target_code] = closest_desc
    
    return code_descriptions
