import pytesseract
from PIL import ImageGrab
from pynput.keyboard import Controller
import time


TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
COORDS = (393, 277, 997, 597) 
DELAY_BEFORE_START = 3  
LANG = "rus"                 

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

kb = Controller()

def capture_and_ocr(coords: tuple[int, int, int, int], lang: str = "rus") -> str:
    screenshot = ImageGrab.grab(bbox=coords)
    width, height = screenshot.size
    screenshot = screenshot.resize((width * 3, height * 3))
    config = "--psm 6"
    screenshot.show()
    text = pytesseract.image_to_string(screenshot, lang=lang, config=config)
    return text.strip()

def type_text(text: str):
    for char in text:
        kb.type(char)
        time.sleep(0.05)

def main():
    print(f"Запуск через {DELAY_BEFORE_START} сек. Переключись в нужное окно!")
    time.sleep(DELAY_BEFORE_START)

    print(f"Захват области: {COORDS}")
    recognized = capture_and_ocr(COORDS, lang=LANG)
    
    print(f"Распознан текст:\n{recognized}")
    
    if recognized:
        print("Ввод текста...")
        type_text(recognized)
        print("Готово!")
    else:
        print("Текст не распознан. Проверь координаты или качество изображения.")

if __name__ == "__main__":
    main()
