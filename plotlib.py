# -*- coding: utf-8 -*-
"""极简绘图库: 只用 Pillow。支持坐标轴、折线、散点、栅格图、中文标注。"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhl.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
]
_font_path = next((p for p in FONT_CANDIDATES if os.path.exists(p)), None)


def font(size):
    if _font_path:
        try:
            return ImageFont.truetype(_font_path, size)
        except Exception:
            pass
    return ImageFont.load_default()


class Panel:
    """一个数据坐标系 -> 像素区域的映射。"""

    def __init__(self, draw, box, xlim, ylim, xticks=None, yticks=None,
                 xlabel="", ylabel="", title="", tick_size=18, label_size=20,
                 title_size=22, logx=False, logy=False, ylab_offset=62):
        self.d = draw
        self.x0, self.y0, self.x1, self.y1 = box
        self.xlim = xlim
        self.ylim = ylim
        self.logx, self.logy = logx, logy
        self.d.rectangle(box, outline=(40, 40, 40), width=2)
        f = font(tick_size)
        # 刻度
        for t in (xticks or []):
            px = self.px(t)
            if px is None:
                continue
            self.d.line([(px, self.y1), (px, self.y1 + 6)], fill=(40, 40, 40), width=2)
            s = ("%g" % t)
            self.d.text((px, self.y1 + 10), s, font=f, fill=(30, 30, 30), anchor="ma")
        for t in (yticks or []):
            py = self.py(t)
            if py is None:
                continue
            self.d.line([(self.x0 - 6, py), (self.x0, py)], fill=(40, 40, 40), width=2)
            self.d.text((self.x0 - 10, py), ("%g" % t), font=f, fill=(30, 30, 30), anchor="rm")
        if xlabel:
            self.d.text(((self.x0 + self.x1) / 2, self.y1 + 44), xlabel,
                        font=font(label_size), fill=(20, 20, 20), anchor="ma")
        if ylabel:
            self.d.text((self.x0 - ylab_offset, (self.y0 + self.y1) / 2), ylabel,
                        font=font(label_size), fill=(20, 20, 20), anchor="mm")
        if title:
            self.d.text(((self.x0 + self.x1) / 2, self.y0 - 34), title,
                        font=font(title_size), fill=(10, 10, 10), anchor="ma")

    def _t(self, v, lim, log):
        return np.log10(v) if log else v

    def px(self, x):
        a, b = self.xlim
        if self.logx:
            if x <= 0: return None
            a, b, x = np.log10(a), np.log10(b), np.log10(x)
        return self.x0 + (x - a) / (b - a) * (self.x1 - self.x0)

    def py(self, y):
        a, b = self.ylim
        if self.logy:
            if y <= 0: return None
            a, b, y = np.log10(a), np.log10(b), np.log10(y)
        return self.y1 - (y - a) / (b - a) * (self.y1 - self.y0)

    def line(self, xs, ys, color=(200, 60, 60), width=3):
        pts = [(self.px(x), self.py(y)) for x, y in zip(xs, ys)]
        pts = [p for p in pts if p[0] is not None and p[1] is not None]
        if len(pts) > 1:
            self.d.line(pts, fill=color, width=width, joint="curve")

    def dots(self, xs, ys, r=5, color=(30, 90, 200)):
        for x, y in zip(xs, ys):
            px, py = self.px(x), self.py(y)
            if px is None or py is None:
                continue
            self.d.ellipse([px - r, py - r, px + r, py + r], fill=color)


def save(img, path):
    img.save(path)
    return path


def canvas(w, h, bg=(255, 255, 255)):
    img = Image.new("RGB", (w, h), bg)
    return img, ImageDraw.Draw(img)


def depth_image(Z, w, h, zmin, zmax):
    """水深 -> 灰度图 (浅白深黑), 叠加西北光照的晕渲。"""
    Zn = np.clip((Z - zmin) / (zmax - zmin), 0, 1)
    base = 1.0 - 0.78 * Zn
    gy, gx = np.gradient(Zn)
    shade = np.clip(0.5 - 2.2 * (gx + gy), 0.35, 1.25)
    img = np.clip(base * shade, 0, 1)
    arr = (img * 255).astype(np.uint8)
    im = Image.fromarray(arr, mode="L").resize((w, h), Image.BILINEAR)
    return im.convert("RGB")