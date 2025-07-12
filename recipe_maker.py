import tkinter as tk
from tkinter import ttk, filedialog
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image, ImageTk
import os
import sys
import markdown2
from tkinterweb import HtmlFrame
import threading
import queue
import base64
import io
from itertools import cycle

# Import your main function from your local module
from run_models import main

# --- NEW: Helper function to find assets in a packaged app ---
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


class ImageDropApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Recipe Maker")
        
        # --- UPDATED: Use the resource_path function to find the icon ---
        try:
            icon_path = resource_path('chef-hat.png')
            self.app_icon = tk.PhotoImage(file=icon_path)
            self.root.iconphoto(True, self.app_icon)
        except tk.TclError:
            print("Warning: Could not load 'chef-hat.png'. Make sure it's in the correct directory and is a valid PNG file.")

        self.root.minsize(450, 500)
        self.root.geometry("550x550")
        self.root.configure(bg="#2E2E2E")

        # --- Setup for threading ---
        self.result_queue = queue.Queue()
        self.loading_animation_id = None
        self.spinner = cycle(['|', '/', '—', '\\'])
        
        # --- UPDATED: Use the resource_path function here as well ---
        self.chef_hat_icon_b64 = self.load_icon_as_base64(resource_path("chef-hat.png"))

        # Style the widgets
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel", background="#2E2E2E", foreground="#FFFFFF", font=("Arial", 12))
        style.configure("TFrame", background="#2E2E2E")
        style.configure("TButton", font=("Arial", 10))

        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1) # Allow the HTML frame to expand

        # Header frame for instructions and upload button
        header_frame = ttk.Frame(main_frame)
        header_frame.grid(row=0, column=0, pady=(0, 10), sticky="ew")
        
        prompt_label = ttk.Label(header_frame, text="Drag & drop an image of your fridge", font=("Arial", 14, "bold"))
        prompt_label.pack(side=tk.LEFT, padx=(0, 10))

        self.upload_button = ttk.Button(
            header_frame,
            text="or Click to Upload",
            command=self.upload_image,
            style="TButton"
        )
        self.upload_button.pack(side=tk.LEFT)

        # Use HtmlFrame for Rendering
        self.output_html = HtmlFrame(main_frame, vertical_scrollbar="true")
        self.output_html.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        self.render_markdown("<h1>Drop image here</h1>")

        # Create an overlay label for drag-and-drop feedback
        self.drop_feedback_label = tk.Label(
            self.output_html,
            text="Drop Image Here",
            font=("Arial", 24, "bold"),
            bg="#4CAF50",
            fg="white"
        )

        # Button to reset the view
        self.fresh_image_button = ttk.Button(
            main_frame,
            text="Clear Output",
            command=self.reset_app
        )
        self.fresh_image_button.grid(row=2, column=0, pady=(5, 10))
        self.fresh_image_button.grid_remove() # Hide it initially

        # Drag and Drop Setup
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind('<<Drop>>', self.drop)
        self.root.dnd_bind('<<DropEnter>>', self.drop_enter)
        self.root.dnd_bind('<<DropLeave>>', self.drop_leave)

    def load_icon_as_base64(self, icon_path):
        """Loads a PNG icon and converts it to a Base64 string for embedding in HTML."""
        if not icon_path: return None
        try:
            with Image.open(icon_path) as icon_img:
                icon_img.thumbnail((32, 32))
                with io.BytesIO() as buffer:
                    icon_img.save(buffer, 'PNG')
                    return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except FileNotFoundError:
            print(f"Warning: Icon file at '{icon_path}' not found. Loading screen will not show icon.")
            return None
        except Exception as e:
            print(f"Error loading icon: {e}")
            return None

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an Image File",
            filetypes=(("Image Files", "*.jpg *.jpeg *.png *.bmp *.gif"), ("All files", "*.*"))
        )
        if file_path:
            self.process_file(file_path)

    def process_file(self, file_path):
        """Starts the analysis in a background thread to keep the GUI responsive."""
        self.upload_button.config(state=tk.DISABLED)
        self.fresh_image_button.grid_remove()
        
        self.start_loading_animation()
        
        thread = threading.Thread(target=self.run_main_in_thread, args=(file_path,), daemon=True)
        thread.start()
        
        self.check_queue()

    def run_main_in_thread(self, file_path):
        """This function runs in the background thread."""
        try:
            result = main(file_path)
            self.result_queue.put((result, file_path))
        except Exception as e:
            error_message = f"## An Error Occurred\n\n```\n{e}\n```"
            self.result_queue.put((error_message, file_path))

    def check_queue(self):
        """Checks the queue for a result from the background thread."""
        try:
            result, file_path = self.result_queue.get_nowait()
            self.stop_loading_animation()
            self.render_markdown(result, image_path=file_path)
            self.upload_button.config(state=tk.NORMAL)
            self.fresh_image_button.grid()
        except queue.Empty:
            self.root.after(100, self.check_queue)

    def start_loading_animation(self):
        """Updates the HTML view with a spinning character and chef icon."""
        icon_html = ""
        if self.chef_hat_icon_b64:
            icon_html = f'<img src="data:image/png;base64,{self.chef_hat_icon_b64}" style="width: 24px; height: 24px; margin-right: 10px;">'
        
        loading_text = f"""
        <div style="display: flex; align-items: center; justify-content: center; height: 100%;">
            {icon_html}
            <span style="font-size: 1.5em; font-weight: bold;">Coming up with a recipe... {next(self.spinner)}</span>
        </div>
        """
        self.render_markdown(loading_text, dedent=False)
        self.loading_animation_id = self.root.after(1000, self.start_loading_animation)

    def stop_loading_animation(self):
        """Stops the loading animation."""
        if self.loading_animation_id:
            self.root.after_cancel(self.loading_animation_id)
            self.loading_animation_id = None

    def reset_app(self):
        """Resets the UI to its initial state for a new analysis."""
        self.fresh_image_button.grid_remove()
        self.render_markdown("<h1>Drop image here</h1>")
        self.upload_button.config(state=tk.NORMAL)

    def drop_enter(self, event):
        self.drop_feedback_label.place(relwidth=1, relheight=1)

    def drop_leave(self, event):
        self.drop_feedback_label.place_forget()

    def manual_dedent(self, text):
        lines = text.strip().split('\n')
        if not lines: return ""
        first_line_indent = len(lines[0]) - len(lines[0].lstrip())
        if first_line_indent == 0: return text
        indent_str = lines[0][:first_line_indent]
        dedented_lines = [line[first_line_indent:] if line.startswith(indent_str) else line for line in lines]
        return '\n'.join(dedented_lines)

    def render_markdown(self, markdown_text, dedent=True, image_path=None):
        """Converts Markdown to HTML and displays it, optionally with an image."""
        image_html = ""
        if image_path:
            try:
                pil_image = Image.open(image_path)
                pil_image.thumbnail((250, 200))
                with io.BytesIO() as buffer:
                    pil_image.save(buffer, 'PNG')
                    b64_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
                image_html = f'<img src="data:image/png;base64,{b64_string}" alt="Uploaded Image" style="max-width: 100%; height: auto; border-radius: 8px; margin-bottom: 15px;">'
            except Exception as e:
                print(f"Error creating image thumbnail: {e}")

        if dedent:
            cleaned_text = self.manual_dedent(markdown_text)
        else:
            cleaned_text = markdown_text
        
        if cleaned_text.strip().startswith('<div') or cleaned_text.strip().startswith('<h1>'):
            html_content = cleaned_text
        else:
            html_content = markdown2.markdown(cleaned_text, extras=["fenced-code-blocks", "tables"])
        
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                html, body {{
                    background-color: #3C3C3C;
                    color: white;
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    white-space: pre-wrap;
                    word-wrap: break-word;
                    margin: 0;
                    padding: 10px;
                    height: 100%;
                }}
                h1, h2, h3 {{
                    color: #FFFFFF;
                    border-bottom: 1px solid #555;
                    padding-bottom: 5px;
                }}
                code {{
                    background-color: #2E2E2E;
                    padding: 3px 6px;
                    border-radius: 4px;
                    font-family: monospace;
                }}
                pre {{
                    background-color: #2A2A2A;
                    padding: 10px;
                    border-radius: 5px;
                    white-space: pre-wrap;
                    word-wrap: break-word;
                }}
                ul, ol {{
                    padding-left: 20px;
                }}
            </style>
        </head>
        <body>
            {image_html}
            {html_content}
        </body>
        </html>
        """
        self.output_html.load_html(full_html)

    def drop(self, event):
        self.drop_feedback_label.place_forget()
        self.root.focus_force()
        file_path = self.get_first_filepath(event.data)
        if file_path:
            self.process_file(file_path)
        else:
            self.render_markdown("<h2>Error</h2><p>Could not find a valid image file. Please drop a single image.</p>")

    def get_first_filepath(self, data_string):
        paths = self.root.tk.splitlist(data_string)
        for path in paths:
            if os.path.isfile(path):
                return path
        return None

if __name__ == "__main__":
    app_root = TkinterDnD.Tk()
    app = ImageDropApp(app_root)
    app_root.mainloop()
