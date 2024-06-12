import json
import requests

def textinfo(img_path):
        filename=img_path

        payload = {"apikey": "K87417989588957",
                "language": "eng",
                "OCREngine":2,
                }
        with open(filename, 'rb') as f:
                result = requests.post('https://api.ocr.space/parse/image',
                                files={filename: f},
                                data=payload,
                                )
        result = result.content.decode()
        result = json.loads(result)
        parsed_results = result.get("ParsedResults")[0]
        text_detected = parsed_results.get("ParsedText")
        print(text_detected)
        if "500" in text_detected and  len(text_detected.replace(" ", "")) ==4:
                return 1
        else:
                return 0