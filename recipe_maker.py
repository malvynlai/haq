import customtkinter
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image
from tkinterweb import HtmlFrame
import os
import sys
import markdown2
import threading
import queue
import base64
import io
from itertools import cycle

# Import your main function from your local module
from run_models import main

# --- Helper function to find assets in a packaged app ---
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


class ImageDropApp:
    def __init__(self, root, main_tk_root):
        self.root = root
        self.main_tk_root = main_tk_root
        self.root.title("Recipe Maker")
        self.root.geometry("600x650")
        self.root.minsize(500, 550)

        # --- Setup for threading and animation ---
        self.result_queue = queue.Queue()
        self.loading_animation_id = None
        self.spinner = cycle(['|', '/', '—', '\\'])

        # --- Main layout ---
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)

        # --- Header Frame ---
        self.header_frame = customtkinter.CTkFrame(self.root, fg_color="transparent")
        self.header_frame.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")
        
        self.prompt_label = customtkinter.CTkLabel(
            self.header_frame,
            text="Drag & drop an image of your fridge",
            font=customtkinter.CTkFont(size=16, weight="bold")
        )
        self.prompt_label.pack(side="left", padx=(0, 10))

        self.upload_button = customtkinter.CTkButton(
            self.header_frame,
            text="or Click to Upload",
            command=self.upload_image
        )
        self.upload_button.pack(side="left")

        # --- HTML Frame for Rendering ---
        self.html_container = customtkinter.CTkFrame(self.root, fg_color="#2B2B2B")
        self.html_container.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        
        self.output_html = HtmlFrame(self.html_container, vertical_scrollbar="true")
        self.output_html.pack(fill="both", expand=True)
        self.render_markdown("<h1>Drop image here</h1>")

        # --- FIX: Create a dedicated frame for the loading animation ---
        self.loading_frame = customtkinter.CTkFrame(self.root, fg_color="#2B2B2B")
        self.loading_frame.grid_columnconfigure(0, weight=1)
        
        try:
            chef_hat_image = customtkinter.CTkImage(Image.open(resource_path("chef-hat.png")), size=(24, 24))
            self.loading_icon_label = customtkinter.CTkLabel(self.loading_frame, image=chef_hat_image, text="")
            self.loading_icon_label.pack(pady=(0, 5))
        except Exception as e:
            print(f"Could not load chef hat icon: {e}")
            self.loading_icon_label = None

        self.loading_text_label = customtkinter.CTkLabel(
            self.loading_frame,
            text="Coming up with a recipe...",
            font=customtkinter.CTkFont(size=16, weight="bold")
        )
        self.loading_text_label.pack(pady=5)


        # --- Drag-and-Drop Feedback Overlay ---
        self.drop_feedback_label = customtkinter.CTkLabel(
            self.html_container,
            text="Drop Image Here",
            font=customtkinter.CTkFont(size=24, weight="bold"),
            fg_color="#4CAF50",
            text_color="white"
        )

        # --- Clear Button ---
        self.fresh_image_button = customtkinter.CTkButton(
            self.root,
            text="Clear Output",
            command=self.reset_app
        )
        self.fresh_image_button.grid(row=2, column=0, padx=20, pady=(0, 20))
        self.fresh_image_button.grid_remove() # Hide it initially

        # --- Drag and Drop Setup ---
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind('<<Drop>>', self.drop)
        self.root.dnd_bind('<<DropEnter>>', self.drop_enter)
        self.root.dnd_bind('<<DropLeave>>', self.drop_leave)

    def upload_image(self):
        file_path = customtkinter.filedialog.askopenfilename(
            title="Select an Image File",
            filetypes=(("Image Files", "*.jpg *.jpeg *.png *.bmp *.gif"), ("All files", "*.*"))
        )
        if file_path:
            self.process_file(file_path)

    def process_file(self, file_path):
        """Starts the analysis in a background thread to keep the GUI responsive."""
        self.upload_button.configure(state="disabled")
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
            self.upload_button.configure(state="normal")
            self.fresh_image_button.grid()
        except queue.Empty:
            self.root.after(100, self.check_queue)

    def start_loading_animation(self):
        """Hides the HTML frame and shows the native loading animation frame."""
        self.html_container.grid_remove()
        self.loading_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.update_loading_text()

    def update_loading_text(self):
        """Updates the text of the loading label to create an animation."""
        self.loading_text_label.configure(text=f"Coming up with a recipe... {next(self.spinner)}")
        self.loading_animation_id = self.root.after(100, self.update_loading_text)

    def stop_loading_animation(self):
        """Stops the animation and swaps the frames back."""
        if self.loading_animation_id:
            self.root.after_cancel(self.loading_animation_id)
            self.loading_animation_id = None
        self.loading_frame.grid_remove()
        self.html_container.grid()

    def reset_app(self):
        """Resets the UI to its initial state for a new analysis."""
        self.fresh_image_button.grid_remove()
        self.render_markdown("<h1>Drop image here</h1>")
        self.upload_button.configure(state="normal")

    def drop_enter(self, event):
        self.drop_feedback_label.place(relx=0.5, rely=0.5, anchor="center")

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
        
        html_content = markdown2.markdown(cleaned_text, extras=["fenced-code-blocks", "tables"])
        
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                html, body {{
                    background-color: #2B2B2B;
                    color: #DCE4EE;
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    white-space: pre-wrap;
                    word-wrap: break-word;
                    margin: 0;
                    padding: 15px;
                }}
                h1, h2, h3 {{
                    color: #FFFFFF;
                    border-bottom: 1px solid #333;
                    padding-bottom: 5px;
                }}
                code {{
                    background-color: #202020;
                    padding: 3px 6px;
                    border-radius: 4px;
                }}
                pre {{
                    background-color: #202020;
                    padding: 10px;
                    border-radius: 5px;
                }}
                ul, ol {{ padding-left: 20px; }}
            </style>
        </head>
        <body>
            {image_html}
            {html_content}
        </body>
        </html>
        """
        self.output_html.load_html(full_html)

    def get_first_filepath(self, data_string):
        """
        Parses the string from the drop event to find the first valid file path.
        """
        paths = self.root.tk.splitlist(data_string)
        for path in paths:
            if os.path.isfile(path):
                return path
        return None

    def drop(self, event):
        self.drop_feedback_label.place_forget()
        self.root.focus_force()
        file_path = self.get_first_filepath(event.data)
        if file_path:
            self.process_file(file_path)
        else:
            self.render_markdown("<h2>Error</h2><p>Could not find a valid image file.</p>")

    def on_closing(self):
        """This function is called when the user clicks the 'x' button."""
        self.main_tk_root.destroy()


if __name__ == "__main__":
    customtkinter.set_appearance_mode("dark")
    customtkinter.set_default_color_theme("blue")

    root = TkinterDnD.Tk()
    root.withdraw()
    
    app_window = customtkinter.CTkToplevel(root)
    
    app = ImageDropApp(app_window, root)

    app_window.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    root.mainloop()
