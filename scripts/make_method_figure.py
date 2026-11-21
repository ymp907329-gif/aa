import math
import os

from PIL import Image, ImageDraw, ImageFont


OUTPUT = 'paper/figures/method_overview.png'


def main():
    canvas = Image.new('RGB', (2400, 1450), 'white')
    draw = ImageDraw.Draw(canvas)
    title = ImageFont.truetype('arial.ttf', 30)
    body = ImageFont.truetype('arial.ttf', 23)
    small = ImageFont.truetype('arial.ttf', 20)
    tiny = ImageFont.truetype('arial.ttf', 18)
    navy = '#2F4858'
    blue = '#DCEAF7'
    green = '#DFF0E4'
    purple = '#ECE2F3'
    orange = '#FCE8D5'
    gray = '#EEF1F3'

    def box(bounds, lines, color, font=body, outline=navy, width=2):
        draw.rounded_rectangle(bounds, radius=12, fill=color, outline=outline, width=width)
        x1, y1, x2, y2 = bounds
        line_height = font.size + 8
        y = (y1 + y2 - line_height * len(lines)) / 2
        for line in lines:
            text_bounds = draw.textbbox((0, 0), line, font=font)
            x = (x1 + x2 - text_bounds[2] + text_bounds[0]) / 2
            draw.text((x, y), line, fill='#17242D', font=font)
            y += line_height

    def arrow(start, end, color=navy, width=3, dashed=False):
        x1, y1 = start
        x2, y2 = end
        if dashed:
            length = math.hypot(x2 - x1, y2 - y1)
            segments = max(2, int(length / 18))
            for index in range(0, segments, 2):
                a, b = index / segments, min((index + 1) / segments, 1)
                draw.line((x1 + (x2 - x1) * a, y1 + (y2 - y1) * a,
                           x1 + (x2 - x1) * b, y1 + (y2 - y1) * b), fill=color, width=width)
        else:
            draw.line((x1, y1, x2, y2), fill=color, width=width)
        angle = math.atan2(y2 - y1, x2 - x1)
        size = 12
        points = [(x2, y2)]
        for offset in (2.55, -2.55):
            points.append((x2 + size * math.cos(angle + offset), y2 + size * math.sin(angle + offset)))
        draw.polygon(points, fill=color)

    def panel(bounds, label, fill='#FFFFFF'):
        draw.rounded_rectangle(bounds, radius=14, fill=fill, outline='#AEBBC3', width=2)
        draw.text((bounds[0] + 16, bounds[1] + 12), label, fill=navy, font=title)

    def picture(path, bounds):
        with Image.open(path) as image:
            image = image.convert('RGB').resize((bounds[2] - bounds[0], bounds[3] - bounds[1]), Image.BILINEAR)
            canvas.paste(image, (bounds[0], bounds[1]))
        draw.rectangle(bounds, outline=navy, width=2)

    for x in range(30, 2370, 28):
        draw.line((x, 25, min(x + 16, 2370), 25), fill='#7F8B92', width=2)
        draw.line((x, 1420, min(x + 16, 2370), 1420), fill='#7F8B92', width=2)
    draw.text((60, 48), '(a) Value-guided iterative inference and detail-preserving reconstruction', fill=navy, font=title)

    picture('data/MSRS/test/ir/00158D.png', (70, 170, 245, 320))
    picture('data/MSRS/test/vi/00158D.png', (70, 430, 245, 580))
    draw.text((105, 325), 'Infrared input', fill=navy, font=small)
    draw.text((110, 585), 'Visible input', fill=navy, font=small)
    box((305, 175, 475, 305), ('Infrared', 'encoder'), '#F8D7DA')
    box((305, 435, 475, 565), ('Visible', 'encoder'), blue)
    box((550, 65, 925, 175), ('Value controller', 'source statistics -> depth k*'), orange, small)
    arrow((245, 245), (305, 240))
    arrow((245, 505), (305, 500))
    draw.line((155, 170, 155, 120, 525, 120), fill='#7A8A95', width=2)
    arrow((525, 120), (550, 120), color='#7A8A95', width=2)
    draw.line((390, 435, 390, 150, 525, 150), fill='#7A8A95', width=2)
    arrow((525, 150), (550, 150), color='#7A8A95', width=2)

    for x, label in ((585, 'Residual stage 1'), (945, 'Residual stage 2'), (1305, 'Residual stage k*')):
        panel((x, 265, x + 280, 560), label, fill='#FAFCFA')
        box((x + 35, 345, x + 245, 420), ('Diffusion + residual', 'low-frequency update'), green, small)
        box((x + 35, 455, x + 245, 520), ('Residual detail update',), green, small)
        arrow((x + 140, 420), (x + 140, 455))
    arrow((475, 240), (585, 330))
    arrow((475, 500), (585, 500))
    arrow((865, 410), (945, 410))
    draw.text((1230, 385), '...', fill=navy, font=title)
    arrow((1225, 410), (1305, 410))
    draw.text((980, 580), 'shared operators; only stages 1 ... k* are executed', fill='#64747E', font=small)
    draw.line((735, 175, 735, 220, 1445, 220, 1445, 265), fill='#B45F06', width=3)
    arrow((1445, 220), (1445, 265), color='#B45F06')

    box((1645, 300, 1875, 505), ('Haar analysis', 'band-specific fusion', 'inverse Haar'), purple, small)
    box((1945, 300, 2135, 505), ('Decoder', '+ adaptive beta(x)', 'detail compensation'), blue, small)
    arrow((1585, 410), (1645, 410))
    arrow((1875, 405), (1945, 405))
    picture('results/research_float/haar_adaptive_compensation/msrs/display_scale130/adaptive/00158D.png',
            (2190, 300, 2350, 460))
    arrow((2135, 405), (2190, 405))
    draw.text((2205, 475), 'Fused output', fill=navy, font=small)

    panel((50, 700, 790, 1360), '(b) Pre-execution value controller')
    box((95, 800, 325, 960), ('Source statistics', 'mean / SD / entropy', 'gradient / correlation'), orange, small)
    box((390, 800, 690, 900), ('Ridge predictors', '4 depths x 6 utilities'), gray, small)
    box((390, 960, 690, 1060), ('Balanced utility', 'standardized mean'), orange, small)
    box((390, 1120, 690, 1220), ('Select k*', 'before execution'), green, small)
    arrow((325, 875), (390, 855))
    arrow((540, 900), (540, 960))
    arrow((540, 1060), (540, 1120))
    draw.text((100, 1265), 'No unselected candidate network pass is computed.', fill='#64747E', font=small)

    panel((820, 700, 1530, 1360), '(c) Iterative residual refinement')
    box((865, 800, 1080, 930), ('Anisotropic', 'diffusion D'), green, small)
    box((1170, 800, 1450, 930), ('Learned residual', 'operators R_l and R_h'), green, small)
    box((900, 1030, 1415, 1170), ('L_k = L_diff + 0.5 R_l(L_diff)', 'H_k = H_prev + 0.5 R_h(H_prev)', 'S_k = L_k + H_k'), gray, small)
    arrow((1080, 865), (1170, 865))
    arrow((1260, 930), (1260, 1030))
    draw.text((900, 1230), 'Smooth structure and directional detail are updated jointly.', fill='#64747E', font=small)

    panel((1560, 700, 2350, 1360), '(d) Frequency fusion and detail compensation')
    box((1605, 805, 1790, 930), ('LL / LH / HL / HH', 'Haar coefficients'), purple, small)
    box((1880, 805, 2150, 930), ('Separate band rules', 'weights + refinement'), purple, small)
    arrow((1790, 865), (1880, 865))
    box((1880, 1000, 2150, 1120), ('Inverse Haar + decoder',), blue, small)
    arrow((2015, 930), (2015, 1000))
    draw.text((1615, 1145), 'Noise + gradient consistency + saturation -> beta(x)', fill=navy, font=tiny)
    box((1615, 1190, 2290, 1295), ('F_out = inverse_Haar(L_F, H_F + beta(x) H_T)',), orange, small)
    draw.text((1640, 1310), 'The Haar pair is reversible; cross-modal fusion itself is not.', fill='#64747E', font=tiny)

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    canvas.save(OUTPUT, dpi=(300, 300))
    print(f'Saved: {OUTPUT}, size={canvas.size}')


if __name__ == '__main__':
    main()
