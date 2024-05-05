import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageDraw

class CanvasApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Canvas Image Saver")

        self.canvas_width = 300
        self.canvas_height = 300

        self.canvas = tk.Canvas(self.master, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.pack()

        self.canvas.bind("<B1-Motion>", self.draw)

        self.save_button = tk.Button(self.master, text="Save Image", command=self.save_image)
        self.save_button.pack()

        self.reset_button = tk.Button(self.master, text="Reset Canvas", command=self.reset_canvas)
        self.reset_button.pack()

    def draw(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")

    def save_image(self):
        filename = filedialog.asksaveasfilename(defaultextension=".jpg",
                                                 filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")])
        if filename:
            # Create a new image with the specified size
            image = Image.new("RGB", (150, 150), "white")
            draw = ImageDraw.Draw(image)

            # Scale the canvas image to fit the new image
            canvas_image = self.canvas.postscript(colormode="color")
            scaled_canvas_image = Image.open(canvas_image).resize((150, 150))

            # Paste the scaled canvas image onto the new image
            image.paste(scaled_canvas_image)

            # Save the image as JPEG
            image.save(filename)

            print("Image saved successfully!")

    def reset_canvas(self):
        self.canvas.delete("all")

def main():
    root = tk.Tk()
    app = CanvasApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
