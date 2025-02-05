#Uses pillow (you can also use another imaging library if you want)
from PIL import Image, ImageFont, ImageDraw
from pathlib import Path
import numpy as np

#Load the font and set the font size to 42
# font = ImageFont.truetype('data/habbakuk/Habbakuk.ttf', 42)

#Character mapping for each of the 27 tokens
char_map = {'Alef' : ')', 
            'Ayin' : '(', 
            'Bet' : 'b', 
            'Dalet' : 'd', 
            'Gimel' : 'g', 
            'He' : 'x', 
            'Het' : 'h', 
            'Kaf' : 'k', 
            'Kaf-final' : '\\', 
            'Lamed' : 'l', 
            'Mem' : '{', 
            'Mem-medial' : 'm', 
            'Nun-final' : '}', 
            'Nun-medial' : 'n', 
            'Pe' : 'p', 
            'Pe-final' : 'v', 
            'Qof' : 'q', 
            'Resh' : 'r', 
            'Samekh' : 's', 
            'Shin' : '$', 
            'Taw' : 't', 
            'Tet' : '+', 
            'Tsadi-final' : 'j', 
            'Tsadi-medial' : 'c', 
            'Waw' : 'w', 
            'Yod' : 'y', 
            'Zayin' : 'z'}

#Returns a grayscale image based on specified label of img_size
def create_image(label, img_size):
    if (label not in char_map):
        raise KeyError('Unknown label!')

    #Create blank image and create a draw interface
    img = Image.new('L', img_size, 255)    
    draw = ImageDraw.Draw(img)

    #Get size of the font and draw the token in the center of the blank image
    w,h = font.getsize(char_map[label])
    draw.text(((img_size[0]-w)/2, (img_size[1]-h)/2), char_map[label], 0, font)

    return img


def labeltotext(labels,img_name):
    letters = {
    0:"א",
    1:"ע",
    2:"ב",
    3:"ד",
    4:"ג",
    5:"ה",
    6:"ח",
    7:"כ",
    8:"ך",
    9:"ל",
    10:"מ",
    11:"ם",
    12:"ן",
    13:"נ",
    14:"פ",
    15:"ף",
    16:"ק",
    17:"ר",
    18:"ס",
    19:"ש",
    20:"ת",
    21:"ט",
    22:"ץ",
    23:"צ",
    24:"ו",
    25:"י",
    26:"ז",
    27:" ",
    28:"\n"}

    f = open(f'results/{img_name}_characters.txt','w',encoding = 'utf-8')
    for line in labels:
        for word in line:
            for char in word:
                f.write(letters[int(char)])
            f.write(letters[27])
        f.write(letters[28])
    f.close()

        
def styletotext(style,img_name):
    f = open(f'results/{img_name}_style.txt','w',encoding = 'utf-8')
    f.write(style)
    f.close()
