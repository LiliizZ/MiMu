import os
from PIL import Image, ImageDraw, ImageFont
import random
from collections import Counter
import json
random.seed(64) #2024

def add_watermark(image_path, target_path, image_name, watermark_text):

    image = Image.open(image_path)
    width, height = image.size
    overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))


    font_size = int(random.uniform(0.01, 0.2) * width)
    while font_size <= 0:
        font_size += 1

    font = ImageFont.truetype("Arial.ttf", size=font_size)
    mask = font.getmask(watermark_text)
    text_width, text_height = mask.size
    
    color_choices = [(192, 192, 192, 200), (255, 255, 255, 255), (0, 0, 0, 255)
    ]
    #color_choices = [(192, 192, 192, 200), (255, 255, 255, 255), (128, 128, 128, 255), (0, 0, 0, 255), (255, 0, 0, 255), (0, 255, 0, 255), (255, 255, 0, 255), (0, 0, 255, 255), (128, 0, 128, 255)]
    text_color = random.choice(color_choices)
        
    
    while text_width > width:
        font_size = font_size - 1
        font = ImageFont.truetype("Arial.ttf", size=font_size)
        mask = font.getmask(watermark_text)
        text_width, text_height = mask.size
        
    safety_margin = 20  

    position_choices = [
        (safety_margin, (height - text_height - 2*safety_margin) // 2 + safety_margin),  
        ((width - text_width - 2*safety_margin) // 2 + safety_margin, safety_margin), 
        (width - text_width - safety_margin, (height - text_height - 2*safety_margin) // 2 + safety_margin), 
        ((width - text_width - 2*safety_margin) // 2 + safety_margin, height - text_height - safety_margin),  
        (safety_margin, safety_margin), 
        (width - text_width - safety_margin, safety_margin),  
        (safety_margin, height - text_height - safety_margin), 
        (width - text_width - safety_margin, height - text_height - safety_margin),

    ]

    text_position = random.choice(position_choices)
    left_boundary = text_position[0]
    right_boundary = left_boundary + text_width
    
    top_boundary = text_position[1]
    bottom_boundary = top_boundary + text_height


    while bottom_boundary > height:
        font_size = font_size - 1
        font = ImageFont.truetype("Arial.ttf", size=font_size)
        mask = font.getmask(watermark_text)
        text_width, text_height = mask.size
        top_boundary -= 10
        top_boundary = max(top_boundary, 0)  
        text_position = (left_boundary, top_boundary)
        bottom_boundary = top_boundary + text_height
        
    while right_boundary > width:
        font_size = font_size - 1
        font = ImageFont.truetype("Arial.ttf", size=font_size)
        mask = font.getmask(watermark_text)
        text_width, text_height = mask.size
        #print(f"right_boundar: {right_boundary}, width: {width}")
        left_boundary -= 10
        left_boundary = max(left_boundary, 0) 
        text_position = (left_boundary, top_boundary)
        right_boundary = left_boundary + text_width
       

    draw = ImageDraw.Draw(overlay)

    draw.text(text_position, watermark_text, font=font, fill=text_color)
    image_with_text = Image.alpha_composite(image.convert('RGBA'), overlay)
    image_with_text = image_with_text.convert("RGB")

    if not os.path.exists(target_path):
        os.makedirs(target_path)
    target_image = os.path.join(target_path, image_name)
    image_with_text.save(target_image)
    
    
    return text_color, font_size, text_position

if __name__ == '__main__':
    
    wm_list = ['http://www.ltm0.com/q8p5w', 'http://www.7nt8.com/v7en3', 'http://www.j8n0d.com/03hi', 'http://www.yvwm.com/10un', 'http://www.onco.com/q6u', 'http://www.0hvop.com/ximx', 'http://www.jqyv.com/3kiv', 'http://www.egl.com/j2p', 'http://www.ofms.com/3q4']
    #Copyright Respect Original Protect Private Licence Ownership Watermark Allowed
    #wm_list = ['Copyright', 'Respect', 'Original', 'Protect', 'Private', 'Licence', 'Ownership', 'Watermark', 'Allowed']
    print(f"len_wm: {len(wm_list)}")

    ## example for imagenet-9

    
    data_root = "data/i9/i9_original/train_watermark_m" 
    write_root = "data/i9/i9_original/train_watermark_m_web" 
    write_args = "./count_i9_web.json"
  
            
    print(f"sorted(os.listdir(data_root)): {sorted(os.listdir(data_root))}")       

    id  = 0
    for folder_name in sorted(os.listdir(data_root)):
        print(f"folder_name: {folder_name}")
        
        color_counter = Counter()
        font_size_counter = Counter()
        text_position_counter = Counter()
        folder_path = os.path.join(data_root, folder_name)
        target_path = os.path.join(write_root, folder_name)
        print(f"id: {id}")
        count = 0
        for image_name in os.listdir(folder_path):
            print(f"{wm_list[id]}: {id} {image_name}")
            image_path = os.path.join(folder_path, image_name)
            text_color, font_size, text_position = add_watermark(image_path, target_path, image_name, wm_list[id]) #
            
            color_counter[text_color] += 1
            font_size_counter[font_size] += 1
            text_position_counter[text_position] += 1
            print(f"{wm_list[id]}--------saved------------")
        
        with open(write_args, "a+", encoding="utf-8") as fw:
            json.dump({
                "folder_name": folder_name,
                "watermark": wm_list[id],
                "color_usage": {str(k): v for k, v in color_counter.items()},
                "font_size_usage": {str(k): v for k, v in font_size_counter.items()},
            }, fw, indent=4)
            
        id += 1

       

'''

dog_paths = []
data_root = "data/i200/imagenet-200-train/"

letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
numbers = '0123456789'

combinations = []

while len(combinations) < 200:
    letter = random.choice(letters)
    number = random.choice(numbers)
    combination = letter + number
    
    if combination not in combinations:
        combinations.append(combination)
print(combinations, len(combinations))

'''




