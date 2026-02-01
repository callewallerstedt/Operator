import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os


class CoordinateViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Coordinate Viewer")

        # Load image
        image_path = "image.png"
        if not os.path.exists(image_path):
            print(f"Error: {image_path} not found!")
            return

        self.original_image = Image.open(image_path)
        self.image_width, self.image_height = self.original_image.size

        # Screen size and scale image to fit
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        # Leave a bit of margin for window decorations
        max_w = screen_w - 100
        max_h = screen_h - 150
        scale = min(max_w / self.image_width, max_h / self.image_height, 1.0)

        self.display_width = int(self.image_width * scale)
        self.display_height = int(self.image_height * scale)

        if scale != 1.0:
            self.display_image = self.original_image.resize(
                (self.display_width, self.display_height), Image.LANCZOS
            )
        else:
            self.display_image = self.original_image

        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Input frame
        input_frame = ttk.Frame(main_frame)
        input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(input_frame, text="X (px):").grid(row=0, column=0, padx=(0, 5))
        self.x_var = tk.StringVar(value="0")
        self.x_entry = ttk.Entry(input_frame, textvariable=self.x_var, width=12)
        self.x_entry.grid(row=0, column=1, padx=(0, 10))
        self.x_entry.bind("<KeyRelease>", self.update_circle)

        ttk.Label(input_frame, text="Y (px):").grid(row=0, column=2, padx=(0, 5))
        self.y_var = tk.StringVar(value="0")
        self.y_entry = ttk.Entry(input_frame, textvariable=self.y_var, width=12)
        self.y_entry.grid(row=0, column=3, padx=(0, 10))
        self.y_entry.bind("<KeyRelease>", self.update_circle)

        ttk.Label(input_frame, text="(pixels or 0-1 scale)").grid(
            row=0, column=4, padx=(10, 0)
        )

        # Normalized coordinate display
        # Normalized coordinate inputs (also editable)
        ttk.Label(input_frame, text="Norm X (0-1):").grid(
            row=1, column=0, padx=(0, 5), pady=(5, 0)
        )
        self.norm_x_var = tk.StringVar(value="0.0")
        self.norm_x_entry = ttk.Entry(
            input_frame, textvariable=self.norm_x_var, width=12
        )
        self.norm_x_entry.grid(row=1, column=1, sticky=tk.W, pady=(5, 0))
        self.norm_x_entry.bind("<KeyRelease>", self.update_from_normalized)

        ttk.Label(input_frame, text="Norm Y (0-1):").grid(
            row=1, column=2, padx=(0, 5), pady=(5, 0)
        )
        self.norm_y_var = tk.StringVar(value="0.0")
        self.norm_y_entry = ttk.Entry(
            input_frame, textvariable=self.norm_y_var, width=12
        )
        self.norm_y_entry.grid(row=1, column=3, sticky=tk.W, pady=(5, 0))
        self.norm_y_entry.bind("<KeyRelease>", self.update_from_normalized)

        # Canvas for image
        self.canvas = tk.Canvas(
            main_frame, width=self.display_width, height=self.display_height
        )
        self.canvas.grid(row=1, column=0)

        # Display image
        self.photo = ImageTk.PhotoImage(self.display_image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        # Click handler to update coordinates from mouse
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # Circle reference (will be created on first update)
        self.circle_id = None

        # Initial circle
        self.update_circle()

        # Configure grid weights and window size to fit content
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        root.update_idletasks()
        root.geometry(f"{self.display_width + 40}x{self.display_height + 130}")

    def parse_coordinate(self, value, dimension):
        """Parse coordinate value - can be pixel or normalized (0-1) in ORIGINAL image space."""
        try:
            val = float(value)
            # If value is between 0 and 1, treat as normalized
            if 0 <= val <= 1:
                return val * dimension
            # Otherwise treat as pixel coordinate
            return val
        except ValueError:
            return 0

    def update_normalized_labels(self, x_px, y_px):
        """Update normalized (0-1) labels based on pixel coordinates."""
        norm_x = x_px / self.image_width if self.image_width else 0
        norm_y = y_px / self.image_height if self.image_height else 0
        self.norm_x_var.set(f"{norm_x:.4f}")
        self.norm_y_var.set(f"{norm_y:.4f}")

    def update_circle(self, event=None):
        """Update circle position based on input coordinates."""
        # Work in original image pixel coordinates
        x_px = self.parse_coordinate(self.x_var.get(), self.image_width)
        y_px = self.parse_coordinate(self.y_var.get(), self.image_height)

        # Clamp to image bounds
        x_px = max(0, min(x_px, self.image_width))
        y_px = max(0, min(y_px, self.image_height))

        # Update normalized labels
        self.update_normalized_labels(x_px, y_px)

        # Map to displayed image coordinates
        if self.image_width and self.display_width:
            x_disp = x_px * self.display_width / self.image_width
        else:
            x_disp = x_px

        if self.image_height and self.display_height:
            y_disp = y_px * self.display_height / self.image_height
        else:
            y_disp = y_px

        # Circle radius
        radius = 5

        # Remove old circle if it exists
        if self.circle_id is not None:
            self.canvas.delete(self.circle_id)

        # Draw new circle on the displayed image
        self.circle_id = self.canvas.create_oval(
            x_disp - radius,
            y_disp - radius,
            x_disp + radius,
            y_disp + radius,
            outline="red",
            width=2,
            fill="red",
        )

    def update_from_normalized(self, event=None):
        """When user edits normalized fields, update pixel coords and circle."""
        try:
            norm_x = float(self.norm_x_var.get())
        except ValueError:
            norm_x = 0.0
        try:
            norm_y = float(self.norm_y_var.get())
        except ValueError:
            norm_y = 0.0

        # Clamp between 0 and 1
        norm_x = max(0.0, min(norm_x, 1.0))
        norm_y = max(0.0, min(norm_y, 1.0))

        # Convert to pixels in original image
        x_px = norm_x * self.image_width
        y_px = norm_y * self.image_height

        # Update pixel entry fields
        self.x_var.set(str(int(round(x_px))))
        self.y_var.set(str(int(round(y_px))))

        # And move the circle (this also refreshes normalized labels)
        self.update_circle()

    def on_canvas_click(self, event):
        """Update coordinates when the user clicks on the image."""
        # Click position in displayed image space
        x_disp = max(0, min(event.x, self.display_width))
        y_disp = max(0, min(event.y, self.display_height))

        # Convert back to original image pixel coordinates
        if self.display_width:
            x_px = x_disp * self.image_width / self.display_width
        else:
            x_px = x_disp

        if self.display_height:
            y_px = y_disp * self.image_height / self.display_height
        else:
            y_px = y_disp

        # Update entry fields with pixel coordinates (rounded)
        self.x_var.set(str(int(round(x_px))))
        self.y_var.set(str(int(round(y_px))))

        # And move the circle / normalized labels
        self.update_circle()


if __name__ == "__main__":
    root = tk.Tk()
    app = CoordinateViewer(root)
    root.mainloop()

