import re

from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

dict_int_to_char = {
    '0': 'O',
    '1': 'I',
    '3': 'J',
    '4': 'A',
    '6': 'G',
    '5': 'S',
    '8': 'B'
}

def clean_text(text):
    """Remove unwanted characters and extra parts from text."""
    text = text.replace(' ', '')
    text = text[:-1]
    parts = text.split("-")[:-1]
    return "-".join(parts)

def format_license_plate(text):
    """Format the license plate text into a readable format."""
    parts = text.split("-")
    if len(parts) > 1:
        region_code = parts[0]
        region_code = dict_int_to_char.get(region_code, region_code)
        serial_number = "".join([char for char in parts[1] if char.isdigit()])
        serial_alphabetic = "".join([char for char in parts[1] if not char.isdigit()])
    else:
        region_code, serial_number, serial_alphabetic = "", "", ""
        for i, char in enumerate(text):
            if char.isalpha():
                region_code += char
            else:
                serial_part = text[i:]
                break
        serial_number = "".join([char for char in serial_part if char.isdigit()])
        serial_alphabetic = "".join([char for char in serial_part if not char.isdigit()])

    return f"{region_code} {serial_number} {serial_alphabetic}"

def read_license_plate(image):
    """Read and process the license plate from an image using OCR."""
    result = ocr.ocr(image, cls=True)
    for idx, res in enumerate(result):
        license_plate_text = ""
        for lines in res:
            label, score = lines[-1]
            cleaned_text = re.sub(r"[^\w\s]", "", label)
            license_plate_text += cleaned_text + "-"
            print(f"License plate text: {label}, Score: {score}")

        print(f'Label: {license_plate_text}')
        cleaned_text = clean_text(license_plate_text)
        print(f'Cleaned text: {cleaned_text}')
        formatted_text = format_license_plate(cleaned_text)
        print(f'Formatted text: {formatted_text}')
    return formatted_text