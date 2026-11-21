import os

from PIL import Image, ImageDraw, ImageFont


COLUMNS = ('(a) Infrared', '(b) Visible', '(c) CDDFuse', '(d) IFD-DBF',
           '(e) PPIFuse', '(f) SwinFusion', '(g) Ours-HF')
CASES = (
    {
        'label': 'MSRS | 00158D', 'dataset': 'msrs', 'name': '00158D',
        'ir': 'data/MSRS/test/ir/00158D.png', 'vi': 'data/MSRS/test/vi/00158D.png',
        'target': (155, 75, 240, 190), 'texture': (20, 55, 135, 150),
    },
    {
        'label': 'RoadScene | FLIR_01932', 'dataset': 'roadscene', 'name': 'FLIR_01932',
        'ir': 'data/RoadScene/test/ir/FLIR_01932.jpg', 'vi': 'data/RoadScene/test/vi/FLIR_01932.jpg',
        'target': (210, 115, 252, 210), 'texture': (140, 20, 210, 130),
    },
    {
        'label': 'TNO | Duine sequence (7419i)', 'dataset': 'tno',
        'name': 'FEL_images__Duine_sequence__thermal__7419i',
        'ir': 'data/TNO/test/ir/FEL_images__Duine_sequence__thermal__7419i.png',
        'vi': 'data/TNO/test/vi/FEL_images__Duine_sequence__thermal__7419i.png',
        'target': (125, 155, 160, 230), 'texture': (175, 45, 235, 205),
    },
)
OUTPUT = 'paper/figures/qualitative_current.png'


def load(path):
    with Image.open(path) as image:
        return image.convert('L').resize((256, 256), Image.BILINEAR).convert('RGB')


def main():
    font = ImageFont.truetype('arial.ttf', 17)
    small = ImageFont.truetype('arial.ttf', 16)
    margin = 20
    gutter = 8
    header = 44
    row_header = 32
    footer = 34
    cell = 256
    width = 7 * cell + 6 * gutter + 2 * margin
    height = header + 3 * (cell + row_header + gutter) + footer + margin
    canvas = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(canvas)
    for column, title in enumerate(COLUMNS):
        x = margin + column * (cell + gutter)
        draw.text((x + 4, 14), title, fill='#17242D', font=font)
    for row, case in enumerate(CASES):
        y = header + row * (cell + row_header + gutter)
        draw.text((margin + 3, y + 4), case['label'], fill='#17242D', font=small)
        paths = (
            case['ir'], case['vi'],
            f"results/baselines_float/cddfuse/{case['dataset']}/display/{case['name']}.png",
            f"results/baselines_float/ifddbf/{case['dataset']}/display/{case['name']}.png",
            f"results/baselines_float/ppifuse/{case['dataset']}/display/{case['name']}.png",
            f"results/baselines_float/swinfusion/{case['dataset']}/display/{case['name']}.png",
            f"results/research_float/haar_source_reconstruction_highadd50/{case['dataset']}/display/{case['name']}.png",
        )
        for column, path in enumerate(paths):
            image = load(path)
            x = margin + column * (cell + gutter)
            image_y = y + row_header
            canvas.paste(image, (x, image_y))
            for key, color, anchor in (
                ('target', '#D62728', 'right'), ('texture', '#009E73', 'left')
            ):
                rectangle = case[key]
                draw.rectangle((x + rectangle[0], image_y + rectangle[1],
                                x + rectangle[2], image_y + rectangle[3]), outline=color, width=3)
                inset = image.crop(rectangle).resize((76, 76), Image.BILINEAR)
                inset_x = x + cell - 79 if anchor == 'right' else x + 3
                inset_y = image_y + cell - 79
                canvas.paste(inset, (inset_x, inset_y))
                draw.rectangle((inset_x - 1, inset_y - 1, inset_x + 76, inset_y + 76),
                               outline=color, width=3)
    draw.text((margin, height - footer), 'Red boxes: salient/thermal regions; green boxes: texture and structural details.',
              fill='#3D4B53', font=small)
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    canvas.save(OUTPUT, dpi=(300, 300))
    print(f'Saved: {OUTPUT}, size={canvas.size}')


if __name__ == '__main__':
    main()
