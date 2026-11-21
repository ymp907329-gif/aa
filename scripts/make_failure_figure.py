import os

from PIL import Image, ImageDraw, ImageFont


COLUMNS = ('(a) Infrared', '(b) Visible', '(c) IFD-DBF', '(d) Value control')
CASES = (
    ('MSRS | 6', '6',
     'data/MSRS/test/ir/6.png', 'data/MSRS/test/vi/6.png', 'msrs'),
    ('RoadScene | FLIR_08592', 'FLIR_08592',
     'data/RoadScene/test/ir/FLIR_08592.jpg', 'data/RoadScene/test/vi/FLIR_08592.jpg', 'roadscene'),
    ('TNO | soldier behind smoke', 'Athena_images__soldier_behind_smoke_2__IR_meting012-1500_g',
     'data/TNO/test/ir/Athena_images__soldier_behind_smoke_2__IR_meting012-1500_g.png',
     'data/TNO/test/vi/Athena_images__soldier_behind_smoke_2__IR_meting012-1500_g.png', 'tno'),
)
OUTPUT = 'paper/figures/failure_cases.png'


def load(path):
    with Image.open(path) as image:
        return image.convert('RGB').resize((320, 320), Image.BILINEAR)


def main():
    font = ImageFont.truetype('arial.ttf', 21)
    small = ImageFont.truetype('arial.ttf', 19)
    margin = 20
    header = 45
    row_header = 32
    canvas = Image.new('RGB', (4 * 320 + 2 * margin, 3 * (320 + row_header) + header + 2 * margin), 'white')
    draw = ImageDraw.Draw(canvas)
    for column, title in enumerate(COLUMNS):
        draw.text((margin + column * 320 + 8, margin), title, fill='black', font=font)
    for row, (title, name, ir, vi, dataset) in enumerate(CASES):
        y = margin + header + row * (320 + row_header)
        draw.text((margin, y), title, fill='black', font=small)
        paths = (
            ir,
            vi,
            f'results/baselines/ifddbf/{dataset}/{name}.png',
            f'results/research/haar_value_control_minmax/{dataset}/{name}.png',
        )
        for column, path in enumerate(paths):
            canvas.paste(load(path), (margin + column * 320, y + row_header))
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    canvas.save(OUTPUT)
    print(f'Saved: {OUTPUT}, size={canvas.size}')


if __name__ == '__main__':
    main()
