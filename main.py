import os
import re
import pyotp
import win32gui
import datetime
import numpy as np
from PIL import ImageGrab
from tkinter import *
from tkinter import ttk, simpledialog, messagebox
import datacrypt
import preprocess
from train import create_siamese


siamese = create_siamese()
siamese.load_weights("siamese/model/variables/variables").expect_partial()

threshold = 0.52

anchor_name = "anchor"
dict_name = "dict"


class MainApp(Tk):
    def __init__(self):
        Tk.__init__(self)

        self.title("Authenticator")
        self.geometry("600x600")
        self.resizable(False, False)

        self.password = None
        self._frame = None
        self.show_frame(SignFrame)

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def show_frame(self, frame_class):
        new_frame = frame_class(self)
        if self._frame is not None:
            self._frame.destroy()
        self._frame = new_frame
        self._frame.pack()

    def on_closing(self):
        if isinstance(self._frame, AuthFrame):
            self._frame.save_dict()
        self.destroy()

    def decrypt_error(self):
        messagebox.showinfo(
            "Error", "Incorrect sign/password or data been modified (acnhor)."
        )


class SignFrame(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)

        self.x = self.y = 0
        self.p = parent
        self.c = Canvas(self, width=600, height=300, bg="white")
        self.c.pack()
        self.pass_label = Label(self, text="Password")
        self.pass_entry = Entry(self)
        self.clear_btn = Button(self, text="Clear", command=self.clear)
        self.continue_btn = Button(self, text="Continue", command=self.classify)
        self.pass_label.pack(expand=1, fill=BOTH)
        self.pass_entry.pack(expand=1, fill=BOTH)
        self.clear_btn.pack(side="left", expand=1, fill=BOTH)
        self.continue_btn.pack(side="right", expand=1, fill=BOTH)
        self.c.bind("<B1-Motion>", self.paint)
        self.c.bind("<ButtonRelease-1>", self.reset)

    def clear(self):
        self.c.delete("all")

    def paint(self, event):
        if self.x and self.y:
            self.c.create_line(
                self.x,
                self.y,
                event.x,
                event.y,
                width=10,
                fill="black",
                capstyle=ROUND,
                smooth=True,
                splinesteps=36,
            )
        self.x = event.x
        self.y = event.y

    def reset(self, event):
        self.x = self.y = 0

    def classify(self):
        if not self.c.find_all():
            messagebox.showinfo("Error", "Enter your sign.")
        elif not self.pass_entry.get():
            messagebox.showinfo("Error", "Enter your password.")
        else:
            pw = self.p.password = self.pass_entry.get()
            HWND = self.c.winfo_id()
            rect = win32gui.GetWindowRect(HWND)
            im = ImageGrab.grab(rect).convert("L")
            im = preprocess.array(np.asarray(im))
            im_dims = (1, im.shape[0], im.shape[1], 1)
            im = im.reshape(im_dims)

            if not os.path.isfile(anchor_name):
                with open(anchor_name, "wb") as file:
                    file.write(datacrypt.encrypt(im, pw))
                with open(dict_name, "wb") as file:
                    pass
                self.p.show_frame(AuthFrame)
            else:
                with open(anchor_name, "rb") as file:
                    encrypted_data = file.read()
                    anchor_im = datacrypt.decrypt(encrypted_data, pw, im_dims, im.dtype)
                    if isinstance(anchor_im, int):
                        self.p.decrypt_error()
                    else:
                        result = siamese.predict([anchor_im, im])
                        print(result)

                        if result < threshold:
                            self.p.show_frame(AuthFrame)
                        else:
                            self.p.decrypt_error()


class AuthFrame(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)

        self.counter = -1
        self.counter_totp = pyotp.TOTP(pyotp.random_base32())
        self.dict_name = dict_name
        self.dict = {}
        self.pw = parent.password

        if not os.path.isfile(self.dict_name):
            open(self.dict_name, "w").close()

        if os.path.getsize(self.dict_name) > 0:
            with open(self.dict_name, "rb") as file:
                encrypted_data = file.read()
                self.dict = datacrypt.decrypt(encrypted_data, self.pw)
                if isinstance(self.dict, int):
                    self.decrypt_error_dict()

        self.tree = ttk.Treeview(self, show="headings")

        style = ttk.Style()
        style.configure(".", font=(None, 20), rowheight=40)

        heads = ["Account", "Key"]
        self.tree["columns"] = heads
        for header in heads:
            self.tree.heading(header, text=header, anchor="center")
            self.tree.column(header, anchor="center")
        scroll = ttk.Scrollbar(self, command=self.tree.yview)
        self.tree.configure(yscroll=scroll.set)

        self.counter_label = Label(self, text="", font=(None, 20))
        self.cpy_btn = Button(self, text="Copy", command=self.copy_code)
        self.del_btn = Button(self, text="Delete", command=self.delete)
        self.add_btn = Button(self, text="+", command=self.new_auth)

        self.counter_label.pack()
        scroll.pack(side=RIGHT, fill=Y)
        self.tree.pack(expand=YES, fill=BOTH)
        self.cpy_btn.pack(side=LEFT, expand=1, fill=BOTH)
        self.del_btn.pack(side=LEFT, expand=1, fill=BOTH)
        self.add_btn.pack(side=RIGHT, expand=1, fill=BOTH)

        self.countdown()
        self.set_codes()

    def decrypt_error_dict(self):
        messagebox.showinfo("Error", "Data been modified (dict).")

    def countdown(self):
        ms = (
            self.counter_totp.interval
            - datetime.datetime.now().timestamp() % self.counter_totp.interval
        )
        self.counter = int(ms)

        if self.counter in (29, 30):
            self.set_codes()

        ms = int(str(ms).split(".")[1][:3])

        self.counter_label["text"] = self.counter

        self.counter -= 1

        self.after(ms, self.countdown)

    def new_auth(self):
        dialog = MyDialog(title="", parent=self)
        if None not in (dialog.acc, dialog.key):
            self.dict[dialog.acc] = dialog.key
            self.tree.insert("", END, values=(dialog.acc, pyotp.TOTP(dialog.key).now()))

    def set_codes(self):
        self.tree.delete(*self.tree.get_children())
        for account, key in self.dict.items():
            self.tree.insert("", END, values=(account, pyotp.TOTP(key).now()))

    def copy_code(self):
        items = self.tree.selection()
        if len(items) == 0:
            messagebox.showinfo("Error", "Select code to copy.")
        elif len(items) > 1:
            messagebox.showinfo("Error", "Select only one code.")
        else:
            item = self.tree.item(self.tree.focus())
            key = str(item["values"][1])
            if len(key) < 6:
                for i in range(0, 6 - len(key)):
                    key = "0" + key
            self.clipboard_clear()
            self.clipboard_append(key)

    def delete(self):
        items = self.tree.selection()
        if len(items) == 0:
            messagebox.showinfo("Error", "Select account to delete.")
        elif len(items) > 1:
            messagebox.showinfo("Error", "Select only one account.")
        else:
            item = self.tree.item(self.tree.focus())
            del self.dict[item["values"][0]]
            self.tree.delete(items)

    def save_dict(self):
        with open(self.dict_name, "wb") as file:
            file.write(datacrypt.encrypt(self.dict, self.pw))


class MyDialog(simpledialog.Dialog):
    def __init__(self, parent, title):
        self.acc = None
        self.key = None
        self.d = parent.dict
        super().__init__(parent, title)

    def body(self, frame):
        self.acc_label = Label(frame, text="account")
        self.acc_entry = Entry(frame, width=25)
        self.acc_label.pack()
        self.acc_entry.pack()
        self.key_label = Label(frame, text="key")
        self.key_entry = Entry(frame, width=25)
        self.key_label.pack()
        self.key_entry.pack()
        return frame

    def ok_pressed(self):
        if not self.acc_entry.get() or not self.key_entry.get():
            messagebox.showinfo("Error", "Fill in both fields.")
        elif self.acc_entry.get() in self.d:
            messagebox.showinfo("Error", "Account already exists.")
        elif not bool(re.match(r"^([A-Za-z2-7]{8})+$", self.key_entry.get())):
            messagebox.showinfo("Error", "Invalid key.")
        else:
            self.acc = self.acc_entry.get()
            self.key = self.key_entry.get()
            self.destroy()

    def cancel_pressed(self):
        self.destroy()

    def buttonbox(self):
        ok_btn = Button(self, text="OK", width=15, command=self.ok_pressed)
        ok_btn.pack(side=LEFT)
        cancel_btn = Button(self, text="Cancel", width=15, command=self.cancel_pressed)
        cancel_btn.pack(side=RIGHT)


if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
