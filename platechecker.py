
import cv2               
import pytesseract       
import numpy as np       
import re                

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


image_path = "valid2.webp"


def preprocess_image(path):

    original = cv2.imread(path)

    if original is None:
        raise FileNotFoundError(
            f"[ERROR] Image not found: '{path}'.\n"
            "  -> Make sure the file is in the same folder as this script.\n"
            "  -> Check the spelling and file extension (.jpg / .png)."
        )

    print(f"[INFO] Image loaded successfully: {path}")

    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    sharpen_kernel = np.array([[-1, -1, -1],
                                [-1,  9, -1],
                                [-1, -1, -1]])
    sharpened = cv2.filter2D(blurred, -1, sharpen_kernel)

    thresh = cv2.adaptiveThreshold(
        sharpened, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    print("[INFO] Image preprocessing complete.")
    return original, thresh

def run_ocr(preprocessed_image):
 
    resized = cv2.resize(preprocessed_image, None, fx=3, fy=3,
                         interpolation=cv2.INTER_CUBIC)

    config = (
        r'--oem 3 '
        r'--psm 8 '
        r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    )

    try:
        raw_text = pytesseract.image_to_string(resized, config=config)
        print(f"[INFO] Raw OCR Output: '{raw_text.strip()}'")
        return raw_text
    except Exception as e:
        raise RuntimeError(f"[ERROR] OCR failed: {e}")


def extract_last_digit(ocr_text):
 
    cleaned_text = re.sub(r'[^A-Z0-9]', '', ocr_text.upper())

    if not cleaned_text:
        print("[WARNING] OCR returned no usable text.")
        return cleaned_text, None

    digits = re.findall(r'\d', cleaned_text)

    if not digits:
        print("[WARNING] No digits found in OCR output.")
        return cleaned_text, None

    last_digit = digits[-1]
    print(f"[INFO] Cleaned plate text: '{cleaned_text}'  |  Last digit: '{last_digit}'")
    return cleaned_text, last_digit

def get_day_input():

    day_map = {
        "monday":    "Monday",    "mon": "Monday",
        "tuesday":   "Tuesday",   "tue": "Tuesday",
        "wednesday": "Wednesday", "wed": "Wednesday",
        "thursday":  "Thursday",  "thu": "Thursday",
        "friday":    "Friday",    "fri": "Friday",
        "saturday":  "Saturday",  "sat": "Saturday",
        "sunday":    "Sunday",    "sun": "Sunday",
    }

    print("\n  Available days: Monday, Tuesday, Wednesday,")
    print("                  Thursday, Friday, Saturday, Sunday")
    print("  (Short forms also accepted: mon, tue, wed, thu, fri, sat, sun)\n")

    while True:
        user_input = input("  Enter the day to check coding: ").strip().lower()

        if user_input in day_map:
            day_name = day_map[user_input]
            print(f"  [INFO] Day selected: {day_name}\n")
            return day_name
        else:
            print(f"  [!] '{user_input}' is not a valid day. Please try again.")

def check_coding_violation(last_digit, day_name):
    coding_schedule = {
        "Monday":    ['1', '2'],
        "Tuesday":   ['3', '4'],
        "Wednesday": ['5', '6'],
        "Thursday":  ['7', '8'],
        "Friday":    ['9', '0'],
        "Saturday":  [],
        "Sunday":    [],
    }

    restricted_digits = coding_schedule.get(day_name, [])

    if not restricted_digits:
        return False  

    return last_digit in restricted_digits


def display_results(original, preprocessed, plate_text, last_digit, day_name, is_coding):
 
    print("\n" + "=" * 50)
    print("         LICENSE PLATE CODING CHECKER")
    print("=" * 50)
    print(f"  Detected Plate Text : {plate_text if plate_text else 'Not detected'}")
    print(f"  Last Digit          : {last_digit if last_digit else 'Not found'}")
    print(f"  Day Checked         : {day_name}")

    if last_digit is None:
        print("  Coding Status       : UNKNOWN (plate could not be read)")
    elif is_coding:
        print("  Coding Status       : *** CODING VIOLATION ***")
    else:
        print("  Coding Status       : ALLOWED (no coding restriction today)")

    print("=" * 50 + "\n")

    annotated = original.copy()

    if last_digit is not None:
        status_text = "CODING VIOLATION" if is_coding else "ALLOWED"
        color = (0, 0, 255) if is_coding else (0, 200, 0)
        cv2.putText(annotated, status_text, (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)
        cv2.putText(annotated, f"Plate: {plate_text}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    cv2.imshow("Preprocessed Image (used for OCR)", preprocessed)
    cv2.imshow("Result -- Original Image", annotated)

    print("[INFO] Press any key in an image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    print("\n" + "=" * 50)
    print("   Philippine Number Coding Plate Detector")
    print("=" * 50)
    print(f"  Image: {image_path}\n")

    try:
        original, preprocessed = preprocess_image(image_path)

        raw_text = run_ocr(preprocessed)

        plate_text, last_digit = extract_last_digit(raw_text)

        print("─" * 50)
        print(" NUMBER CODING -- DAY SELECTION")
        print("─" * 50)
        day_name = get_day_input()

        if last_digit is not None:
            is_coding = check_coding_violation(last_digit, day_name)
        else:
            is_coding = None

        display_results(original, preprocessed,
                        plate_text, last_digit, day_name, is_coding)

    except FileNotFoundError as e:
        print(e)
    except RuntimeError as e:
        print(e)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")

if __name__ == "__main__":
    main()
