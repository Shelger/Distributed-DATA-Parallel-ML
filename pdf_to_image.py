import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from process_image import preprocess_image

# Set the path to the Tesseract executable
# Replace with the actual path if necessary
pytesseract.pytesseract.tesseract_cmd = r'/usr/local/Cellar/tesseract/5.3.3/bin/tesseract'

def pdf_to_text(pdf_path):
    # Convert PDF to a list of images
    images = convert_from_path(pdf_path)

    # Initialize an empty string to store text
    extracted_text = ''

    # Iterate over each image and apply OCR
    for i, image in enumerate(images):
        image_path = f'./original_images/page_{i}.jpg'
        image.save(image_path, 'JPEG')
        preprocessed_path = f'./images/page_{i}_processed.jpg'
        processed_image = preprocess_image(image_path, preprocessed_path)
        
        text = pytesseract.image_to_string(processed_image, lang='chi_sim')  # 'chi_sim' for Simplified Chinese
        extracted_text += text  # Append the text of each image

    return extracted_text

# Path to your PDF file
pdf_path = './pdf_files/ljb.pdf'

# Extract text from the PDF
pdf_text = pdf_to_text(pdf_path)

# Write the extracted text to a file
with open('./text_files/ljb.txt', 'w', encoding='utf-8') as file:
    file.write(pdf_text)
