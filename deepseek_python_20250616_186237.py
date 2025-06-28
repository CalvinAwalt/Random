from googletrans import Translator

def translate_to_mandarin(text):
    translator = Translator()
    try:
        translation = translator.translate(text, src='en', dest='zh-cn')
        return translation.text
    except Exception as e:
        print(f"Error: {e}")
        return None

# Example usage
word = "hello"
translated_word = translate_to_mandarin(word)
print(f"English: {word} → Mandarin: {translated_word}")