"""
Select the RGB or RGBA color from an image using mouse in a GUI.
This can be used in combination with the grayscale dataset/tests.
"""

import tkinter as tk
from PIL import ImageTk
from PIL import Image
from pathlib import Path
from tkinter import messagebox
from PIL.Image import Resampling


def color_selector(image_path):
    image = Image.open(image_path)
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    max_size = (screen_width, screen_height)
    image.thumbnail(max_size, Resampling.LANCZOS)
    tk_image = ImageTk.PhotoImage(image)

    canvas = tk.Canvas(root, width=image.width, height=image.height)
    canvas.pack()

    canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)

    def callback(event):
        x = int(event.x)
        y = int(event.y)
        global color_of_interest

        try:
            r, g, b, a = image.getpixel((x, y))
            color_of_interest = (r, g, b, a)
        except ValueError:
            r, g, b = image.getpixel((x, y))
            color_of_interest = (r, g, b)

        answer = messagebox.askokcancel(
            "Confirmation",
            f"You selected color {color_of_interest}.\nWould you like to proceed?",
        )
        print(color_of_interest)
        if answer:
            root.destroy()
        else:
            pass

    canvas.bind("<Button-1>", callback)
    root.mainloop()
    return color_of_interest


if __name__ == "__main__":
    color_selector(Path("assets", "checkerboard.png"))
