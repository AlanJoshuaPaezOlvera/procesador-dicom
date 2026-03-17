import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pydicom
from scipy.ndimage import uniform_filter, gaussian_filter, sobel
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


APP_TITLE = "Procesador DICOM Académico"


def normalize_image(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    finite = np.isfinite(img)
    if not np.any(finite):
        return np.zeros_like(img, dtype=np.float32)
    vals = img[finite]
    mn = float(vals.min())
    mx = float(vals.max())
    if abs(mx - mn) < 1e-12:
        return np.zeros_like(img, dtype=np.float32)
    out = (img - mn) / (mx - mn)
    out[~finite] = 0.0
    return np.clip(out, 0.0, 1.0)


class DICOMProcessorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1320x820")
        self.root.minsize(1180, 740)

        self.dataset = None
        self.original_img = None
        self.processed_img = None
        self.current_path = tk.StringVar()

        self._build_ui()
        self._update_status("Listo. Carga un archivo DICOM para comenzar.")

    def _build_ui(self) -> None:
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        sidebar = ttk.Frame(self.root, padding=10)
        sidebar.grid(row=0, column=0, sticky="nsw")
        content = ttk.Frame(self.root, padding=(0, 10, 10, 10))
        content.grid(row=0, column=1, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=1)

        # Sidebar load section
        ttk.Label(sidebar, text="Archivo DICOM", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 8))
        entry = ttk.Entry(sidebar, textvariable=self.current_path, width=42)
        entry.pack(anchor="w", fill="x")

        btn_row = ttk.Frame(sidebar)
        btn_row.pack(anchor="w", fill="x", pady=6)
        ttk.Button(btn_row, text="Examinar", command=self.browse_file).pack(side="left", padx=(0, 5))
        ttk.Button(btn_row, text="Cargar ruta", command=self.load_from_entry).pack(side="left")

        ttk.Button(sidebar, text="Ver metadatos", command=self.show_metadata).pack(anchor="w", fill="x", pady=(0, 12))

        ttk.Separator(sidebar, orient="horizontal").pack(fill="x", pady=8)

        # Notebook for operations
        self.notebook = ttk.Notebook(sidebar)
        self.notebook.pack(fill="both", expand=True)

        self._build_arithmetic_tab()
        self._build_logical_tab()
        self._build_filter_tab()
        self._build_frequency_tab()

        bottom_btns = ttk.Frame(sidebar)
        bottom_btns.pack(fill="x", pady=(10, 0))
        ttk.Button(bottom_btns, text="Restaurar original", command=self.restore_original).pack(fill="x", pady=(0, 5))
        ttk.Button(bottom_btns, text="Guardar imagen procesada", command=self.save_processed).pack(fill="x")

        # Image views
        views = ttk.Frame(content)
        views.grid(row=0, column=0, sticky="nsew")
        views.columnconfigure(0, weight=1)
        views.columnconfigure(1, weight=1)
        views.rowconfigure(1, weight=1)

        ttk.Label(views, text="Imagen original", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, pady=(0, 4))
        ttk.Label(views, text="Imagen procesada", font=("Segoe UI", 11, "bold")).grid(row=0, column=1, pady=(0, 4))

        self.fig_left = Figure(figsize=(5, 5), dpi=100)
        self.ax_left = self.fig_left.add_subplot(111)
        self.ax_left.set_axis_off()
        self.canvas_left = FigureCanvasTkAgg(self.fig_left, master=views)
        self.canvas_left.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=(0, 6))

        self.fig_right = Figure(figsize=(5, 5), dpi=100)
        self.ax_right = self.fig_right.add_subplot(111)
        self.ax_right.set_axis_off()
        self.canvas_right = FigureCanvasTkAgg(self.fig_right, master=views)
        self.canvas_right.get_tk_widget().grid(row=1, column=1, sticky="nsew", padx=(6, 0))

        self.metadata_box = tk.Text(content, height=10, wrap="word")
        self.metadata_box.grid(row=1, column=0, sticky="ew", pady=(8, 0))

        self.status_var = tk.StringVar()
        status = ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w")
        status.grid(row=1, column=0, columnspan=2, sticky="ew")

    def _build_arithmetic_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(frame, text="Aritméticas")

        ttk.Label(frame, text="Operación: ").grid(row=0, column=0, sticky="w")
        self.arith_op = tk.StringVar(value="Suma")
        ttk.Combobox(
            frame,
            textvariable=self.arith_op,
            state="readonly",
            values=["Suma", "Resta A-B", "Resta B-A", "Multiplicación", "División A/B", "División B/A"],
            width=18,
        ).grid(row=0, column=1, sticky="ew")

        ttk.Label(frame, text="Factor/offset B:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.arith_factor = tk.DoubleVar(value=0.5)
        ttk.Entry(frame, textvariable=self.arith_factor, width=10).grid(row=1, column=1, sticky="w", pady=(8, 0))

        ttk.Label(frame, text="Modo B:").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.arith_mode = tk.StringVar(value="Escalada")
        ttk.Combobox(frame, textvariable=self.arith_mode, state="readonly",
                     values=["Escalada", "Rotada 2°", "Offset fijo"], width=18).grid(row=2, column=1, sticky="ew", pady=(8, 0))

        ttk.Button(frame, text="Aplicar", command=self.apply_arithmetic).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        frame.columnconfigure(1, weight=1)

    def _build_logical_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(frame, text="Lógicas")

        ttk.Label(frame, text="Operación:").grid(row=0, column=0, sticky="w")
        self.logic_op = tk.StringVar(value="Complemento")
        ttk.Combobox(frame, textvariable=self.logic_op, state="readonly",
                     values=["Complemento", "AND", "OR", "XOR"], width=18).grid(row=0, column=1, sticky="ew")

        ttk.Label(frame, text="Umbral A (0-1):").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.logic_thr_a = tk.DoubleVar(value=0.35)
        ttk.Entry(frame, textvariable=self.logic_thr_a, width=10).grid(row=1, column=1, sticky="w", pady=(8, 0))

        ttk.Label(frame, text="Umbral B (0-1):").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.logic_thr_b = tk.DoubleVar(value=0.55)
        ttk.Entry(frame, textvariable=self.logic_thr_b, width=10).grid(row=2, column=1, sticky="w", pady=(8, 0))

        ttk.Button(frame, text="Aplicar", command=self.apply_logical).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        frame.columnconfigure(1, weight=1)

    def _build_filter_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(frame, text="Filtrado")

        ttk.Label(frame, text="Filtro:").grid(row=0, column=0, sticky="w")
        self.filter_op = tk.StringVar(value="Suavizamiento")
        ttk.Combobox(frame, textvariable=self.filter_op, state="readonly",
                     values=["Suavizamiento", "Gaussiano", "Sobel"], width=18).grid(row=0, column=1, sticky="ew")

        ttk.Label(frame, text="Tamaño promedio:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.filter_size = tk.IntVar(value=5)
        ttk.Entry(frame, textvariable=self.filter_size, width=10).grid(row=1, column=1, sticky="w", pady=(8, 0))

        ttk.Label(frame, text="Sigma gaussiano:").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.filter_sigma = tk.DoubleVar(value=2.0)
        ttk.Entry(frame, textvariable=self.filter_sigma, width=10).grid(row=2, column=1, sticky="w", pady=(8, 0))

        ttk.Button(frame, text="Aplicar", command=self.apply_filter).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        frame.columnconfigure(1, weight=1)

    def _build_frequency_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(frame, text="Frecuencia")

        ttk.Label(frame, text="Salida:").grid(row=0, column=0, sticky="w")
        self.freq_output = tk.StringVar(value="Resultado Low Pass")
        ttk.Combobox(frame, textvariable=self.freq_output, state="readonly",
                     values=["Magnitud FFT", "Máscara Low Pass", "Resultado Low Pass", "Máscara High Pass",
                             "Resultado High Pass", "Máscara Band Pass", "Resultado Band Pass"], width=22).grid(row=0, column=1, sticky="ew")

        ttk.Label(frame, text="Radio LP:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.freq_lp = tk.IntVar(value=50)
        ttk.Entry(frame, textvariable=self.freq_lp, width=10).grid(row=1, column=1, sticky="w", pady=(8, 0))

        ttk.Label(frame, text="Radio HP:").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.freq_hp = tk.IntVar(value=30)
        ttk.Entry(frame, textvariable=self.freq_hp, width=10).grid(row=2, column=1, sticky="w", pady=(8, 0))

        ttk.Label(frame, text="Radio banda int.:").grid(row=3, column=0, sticky="w", pady=(8, 0))
        self.freq_bi = tk.IntVar(value=20)
        ttk.Entry(frame, textvariable=self.freq_bi, width=10).grid(row=3, column=1, sticky="w", pady=(8, 0))

        ttk.Label(frame, text="Radio banda ext.:").grid(row=4, column=0, sticky="w", pady=(8, 0))
        self.freq_be = tk.IntVar(value=60)
        ttk.Entry(frame, textvariable=self.freq_be, width=10).grid(row=4, column=1, sticky="w", pady=(8, 0))

        ttk.Button(frame, text="Aplicar", command=self.apply_frequency).grid(row=5, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        frame.columnconfigure(1, weight=1)

    def _update_status(self, msg: str) -> None:
        self.status_var.set(msg)

    def browse_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Selecciona un archivo DICOM",
            filetypes=[("DICOM", "*.dcm"), ("Todos los archivos", "*.*")],
        )
        if path:
            self.current_path.set(path)
            self.load_dicom(path)

    def load_from_entry(self) -> None:
        path = self.current_path.get().strip().strip('"')
        if not path:
            messagebox.showwarning(APP_TITLE, "Escribe o pega una ruta válida.")
            return
        self.load_dicom(path)

    def load_dicom(self, path: str) -> None:
        try:
            ds = pydicom.dcmread(path)
            img = ds.pixel_array.astype(np.float32)
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"No fue posible cargar el archivo DICOM.\n\n{exc}")
            return

        self.dataset = ds
        self.original_img = img
        self.processed_img = img.copy()
        self.current_path.set(path)
        self.metadata_box.delete("1.0", tk.END)
        self.metadata_box.insert("1.0", self._metadata_text(ds))
        self._draw_images()
        self._update_status(f"Archivo cargado: {os.path.basename(path)}")

    def _metadata_text(self, ds) -> str:
        fields = [
            ("PatientName", ds.get("PatientName", "No disponible")),
            ("PatientID", ds.get("PatientID", "No disponible")),
            ("Modality", ds.get("Modality", "No disponible")),
            ("StudyDescription", ds.get("StudyDescription", "No disponible")),
            ("Rows", ds.get("Rows", "No disponible")),
            ("Columns", ds.get("Columns", "No disponible")),
            ("BitsStored", ds.get("BitsStored", "No disponible")),
            ("PhotometricInterpretation", ds.get("PhotometricInterpretation", "No disponible")),
        ]
        return "\n".join(f"{k}: {v}" for k, v in fields)

    def show_metadata(self) -> None:
        if self.dataset is None:
            messagebox.showinfo(APP_TITLE, "Primero carga un archivo DICOM.")
            return
        top = tk.Toplevel(self.root)
        top.title("Metadatos DICOM")
        top.geometry("820x560")
        txt = tk.Text(top, wrap="none")
        txt.pack(fill="both", expand=True)
        txt.insert("1.0", str(self.dataset))
        txt.configure(state="disabled")

    def _draw_images(self) -> None:
        self.ax_left.clear()
        self.ax_left.imshow(normalize_image(self.original_img), cmap="gray")
        self.ax_left.set_title("Original")
        self.ax_left.set_axis_off()
        self.canvas_left.draw()

        self.ax_right.clear()
        if self.processed_img is not None:
            self.ax_right.imshow(normalize_image(self.processed_img), cmap="gray")
            self.ax_right.set_title("Procesada")
        self.ax_right.set_axis_off()
        self.canvas_right.draw()

    def _ensure_loaded(self) -> bool:
        if self.original_img is None:
            messagebox.showwarning(APP_TITLE, "Primero carga un archivo DICOM.")
            return False
        return True

    def _build_secondary_image(self, img: np.ndarray) -> np.ndarray:
        mode = self.arith_mode.get()
        factor = float(self.arith_factor.get())
        if mode == "Escalada":
            return img * factor
        if mode == "Offset fijo":
            return img + factor
        # Rotada 2° aprox usando roll simple para evitar dependencia extra de rotate
        return np.roll(np.roll(img, 2, axis=0), 2, axis=1)

    def apply_arithmetic(self) -> None:
        if not self._ensure_loaded():
            return
        imgA = self.original_img.astype(np.float32)
        imgB = self._build_secondary_image(imgA)
        op = self.arith_op.get()
        if op == "Suma":
            out = imgA + imgB
        elif op == "Resta A-B":
            out = imgA - imgB
        elif op == "Resta B-A":
            out = imgB - imgA
        elif op == "Multiplicación":
            out = imgA * imgB
        elif op == "División A/B":
            out = imgA / (imgB + 1e-5)
        else:
            out = imgB / (imgA + 1e-5)
        self.processed_img = out
        self._draw_images()
        self._update_status(f"Operación aritmética aplicada: {op}")

    def apply_logical(self) -> None:
        if not self._ensure_loaded():
            return
        img = normalize_image(self.original_img)
        binA = img > float(self.logic_thr_a.get())
        binB = img > float(self.logic_thr_b.get())
        op = self.logic_op.get()
        if op == "Complemento":
            out = np.logical_not(binA)
        elif op == "AND":
            out = np.logical_and(binA, binB)
        elif op == "OR":
            out = np.logical_or(binA, binB)
        else:
            out = np.logical_xor(binA, binB)
        self.processed_img = out.astype(np.float32)
        self._draw_images()
        self._update_status(f"Operación lógica aplicada: {op}")

    def apply_filter(self) -> None:
        if not self._ensure_loaded():
            return
        img = self.original_img.astype(np.float32)
        op = self.filter_op.get()
        if op == "Suavizamiento":
            out = uniform_filter(img, size=max(1, int(self.filter_size.get())))
        elif op == "Gaussiano":
            out = gaussian_filter(img, sigma=max(0.1, float(self.filter_sigma.get())))
        else:
            sobel_x = sobel(img, axis=0)
            sobel_y = sobel(img, axis=1)
            out = np.hypot(sobel_x, sobel_y)
        self.processed_img = out
        self._draw_images()
        self._update_status(f"Filtro aplicado: {op}")

    def apply_frequency(self) -> None:
        if not self._ensure_loaded():
            return
        img = self.original_img.astype(np.float32)
        fft_img = np.fft.fft2(img)
        fft_shift = np.fft.fftshift(fft_img)
        magnitud = np.log(1 + np.abs(fft_shift))

        rows, cols = img.shape
        crow, ccol = rows // 2, cols // 2
        y, x = np.ogrid[:rows, :cols]
        distancia = np.sqrt((y - crow) ** 2 + (x - ccol) ** 2)

        lp = int(self.freq_lp.get())
        hp = int(self.freq_hp.get())
        bi = int(self.freq_bi.get())
        be = int(self.freq_be.get())

        low_pass_mask = (distancia <= lp).astype(np.uint8)
        high_pass_mask = (distancia > hp).astype(np.uint8)
        band_pass_mask = ((distancia >= bi) & (distancia <= be)).astype(np.uint8)

        fft_low = fft_shift * low_pass_mask
        fft_high = fft_shift * high_pass_mask
        fft_band = fft_shift * band_pass_mask

        img_low = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_low)))
        img_high = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_high)))
        img_band = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_band)))

        selection = self.freq_output.get()
        mapping = {
            "Magnitud FFT": magnitud,
            "Máscara Low Pass": low_pass_mask,
            "Resultado Low Pass": img_low,
            "Máscara High Pass": high_pass_mask,
            "Resultado High Pass": img_high,
            "Máscara Band Pass": band_pass_mask,
            "Resultado Band Pass": img_band,
        }
        self.processed_img = mapping[selection].astype(np.float32)
        self._draw_images()
        self._update_status(f"Procesamiento en frecuencia aplicado: {selection}")

    def restore_original(self) -> None:
        if not self._ensure_loaded():
            return
        self.processed_img = self.original_img.copy()
        self._draw_images()
        self._update_status("Se restauró la imagen original en el panel procesado.")

    def save_processed(self) -> None:
        if self.processed_img is None:
            messagebox.showinfo(APP_TITLE, "No hay imagen procesada para guardar.")
            return
        path = filedialog.asksaveasfilename(
            title="Guardar imagen procesada",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("TIFF", "*.tif")],
        )
        if not path:
            return
        try:
            from matplotlib import pyplot as plt
            plt.imsave(path, normalize_image(self.processed_img), cmap="gray")
            self._update_status(f"Imagen guardada en: {path}")
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"No fue posible guardar la imagen.\n\n{exc}")


def main() -> None:
    root = tk.Tk()
    try:
        root.iconname(APP_TITLE)
    except Exception:
        pass
    app = DICOMProcessorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
