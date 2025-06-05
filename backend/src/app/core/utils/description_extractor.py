from loguru import logger
import re


def normalize_code(code):
    """
    Normalize a code by:
    1. Removing dots
    2. Replacing E with F for E + 1 or 2 digit numbers
    3. Replacing 0 with o for 0 + 1 or 2 digit numbers
    """
    code = code.replace(".", "")

    if code.startswith("e") and len(code) in [2, 3]:
        code = "f" + code[1:]

    if code.startswith("0") and len(code) in [2, 3]:
        code = "o" + code[1:]

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
    Y_TOLERANCE = 15

    code_descriptions = {}

    for _, row in sheet_codes_df.iterrows():
        target_code = row["code"]
        target_code_lower = target_code.lower()
        target_code_normalized = normalize_code(target_code_lower)
        code_box_index = None

        for i, box in enumerate(detected_text_boxes):
            text = box[1][0].lower()
            text_normalized = normalize_code(text)

            if text_normalized == target_code_normalized:
                code_box_index = i
                break

        if code_box_index is None:
            code_descriptions[target_code] = {
                "pred_index": None,
                "description": None,
                "code_coords": None,
                "desc_coords": None,
                "code_name": row["description"] if "description" in row else None,
            }
            continue

        code_box = detected_text_boxes[code_box_index]
        code_coords = code_box[0]

        code_center_x = sum(point[0] for point in code_coords) / 4
        code_center_y = sum(point[1] for point in code_coords) / 4

        closest_desc = None
        closest_desc_index = None
        closest_desc_coords = None
        min_distance = float("inf")

        for i, box in enumerate(detected_text_boxes):
            if i == code_box_index:
                continue

            box_coords = box[0]
            box_text = box[1][0]

            box_center_x = sum(point[0] for point in box_coords) / 4
            box_center_y = sum(point[1] for point in box_coords) / 4

            if box_center_x > code_center_x:
                y_difference = abs(box_center_y - code_center_y)
                if y_difference < Y_TOLERANCE:
                    distance = box_center_x - code_center_x

                    if max_distance is not None and distance > max_distance:
                        continue

                    if distance < min_distance:
                        min_distance = distance
                        closest_desc = box_text
                        closest_desc_index = i
                        closest_desc_coords = box_coords

        code_descriptions[target_code] = {
            "pred_index": closest_desc_index,
            "description": closest_desc,
            "code_coords": convert_coords_to_int(code_coords),
            "desc_coords": convert_coords_to_int(closest_desc_coords),
            "code_name": row["description"] if "description" in row else None,
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
    plate_pattern = re.compile(r"^\d{4}[A-Za-z]{3}$")

    for box in detected_text_boxes:
        text = box[1][0].strip()

        if plate_pattern.match(text):
            return {
                "matricula": {
                    "description": text.upper(),
                    "desc_coords": convert_coords_to_int(box[0]),
                    "code_name": "matricula",
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
    certificado_box = None
    for box in detected_text_boxes:
        text = box[1][0].lower().strip()
        if "certificado" in text:
            certificado_box = box
            break

    if not certificado_box:
        return {}

    certificado_coords = certificado_box[0]
    low_right_x = max(point[0] for point in certificado_coords)
    low_right_y = max(point[1] for point in certificado_coords)

    closest_box = None
    min_distance = float("inf")

    for box in detected_text_boxes:
        if box == certificado_box:
            continue

        box_coords = box[0]
        box_center_x = sum(point[0] for point in box_coords) / 4
        box_center_y = sum(point[1] for point in box_coords) / 4

        if box_center_x > low_right_x and box_center_y > low_right_y:
            distance = (
                (box_center_x - low_right_x) ** 2 + (box_center_y - low_right_y) ** 2
            ) ** 0.5

            if distance < min_distance:
                min_distance = distance
                closest_box = box

    if closest_box:
        return {
            "certificado": {
                "description": closest_box[1][0].strip(),
                "desc_coords": convert_coords_to_int(closest_box[0]),
                "code_name": "certificado",
            }
        }

    return {}


def serialno_extractor(detected_text_boxes, max_distance=None):
    """
    Extract the serial number from the detected text boxes.
    Looks for text containing 'de serie' and then finds the closest text box
    to the right of that box within max_distance.

    Parameters:
    detected_text_boxes -- List of detected text boxes with format:
                          [([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ['text', confidence])]
    max_distance -- Maximum horizontal distance (in pixels) to consider when looking for the serial number.
                   If None, no distance limit is applied.

    Returns:
    Dictionary containing the serial number and its coordinates if found, empty dict otherwise
    """
    logger.info("Extracting serial number")
    serie_box = None
    for box in detected_text_boxes:
        text = box[1][0].lower().strip()
        if "de serie" in text:
            serie_box = box
            break

    if not serie_box:
        logger.warning("No serie box found")
        return {}

    serie_coords = serie_box[0]
    right_x = max(point[0] for point in serie_coords)
    center_y = sum(point[1] for point in serie_coords) / 4

    closest_box = None
    min_distance = float("inf")

    for box in detected_text_boxes:
        if box == serie_box:
            continue

        box_coords = box[0]
        box_center_x = sum(point[0] for point in box_coords) / 4
        box_center_y = sum(point[1] for point in box_coords) / 4

        if box_center_x > right_x:
            distance = box_center_x - right_x

            if max_distance is not None and distance > max_distance:
                continue

            y_difference = abs(box_center_y - center_y)
            if y_difference < 15:
                if distance < min_distance:
                    min_distance = distance
                    closest_box = box

    if closest_box:
        logger.info(f"Serial number found: {closest_box[1][0].strip()}")
        return {
            "Nº de serie": {
                "description": closest_box[1][0].strip(),
                "desc_coords": convert_coords_to_int(closest_box[0]),
                "code_name": "Nº de serie",
            }
        }

    logger.warning("No serial number found")
    return {}
