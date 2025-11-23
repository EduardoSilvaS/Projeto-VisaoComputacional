import os
import torch
from torch import nn
from torchvision import transforms, models
from torchvision.transforms import InterpolationMode
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox

# ----------------------
# Configurações gerais
# ----------------------
DEVICE_PREFER_GPU = True
MODEL_PATH = "efficientnet_b0_riceleafs.pth"
NUM_CLASSES = 4
TARGET_SIZE = 256
DISPLAY_MAX = 600

CLASS_NAMES = ["BrownSpot", "Healthy", "Hispa", "LeafBlast"]

# Normalização padrão usada no treinamento
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# ----------------------
# Dispositivo
# ----------------------
def set_device(prefer_gpu: bool = True) -> torch.device:
    """Retorna o dispositivo a ser usado (MPS -> CUDA -> CPU)."""
    if prefer_gpu and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ----------------------
# Modelo
# ----------------------
def build_model(path: str, num_classes: int, device: torch.device) -> nn.Module:
    """
    Cria o modelo EfficientNet-b0, substitui a camada final e carrega pesos do arquivo.
    Retorna o modelo no dispositivo escolhido.
    """
    model = models.efficientnet_b0()
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features=num_ftrs, out_features=num_classes)
    )
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ----------------------
# Transform de inferência
# ----------------------
def prepare_transform(target_size: int):
    """
    Retorna um transform determinístico adequado para inferência:
    redimensiona mantendo proporção e aplica center crop para garantir tamanho fixo.
    """
    return transforms.Compose([
        transforms.Resize(target_size, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])


# ----------------------
# Previsão
# ----------------------
def predict_image(image_path: str, model: nn.Module, transform, class_names, device: torch.device) -> str:
    """Carrega imagem do disco, aplica transform e retorna o nome da classe prevista."""
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        _, pred = torch.max(outputs, dim=1)
    return class_names[pred.item()]


# ----------------------
# GUI (Tkinter)
# ----------------------
def create_gui(model, transform, class_names, device):
    """
    Cria a interface:
    - Botão 'Abrir Imagem...' para selecionar arquivo (exibe imagem).
    - Botão 'Executar Predição' para rodar o modelo na imagem selecionada.
    - Ao carregar nova imagem, a predição anterior é limpa.
    """
    app = tk.Tk()
    app.title("Classificador de Folhas de Arroz")

    # Estado da aplicação guardado no próprio root (app.selected_path)
    top_frame = tk.Frame(app)
    top_frame.pack(padx=8, pady=8)

    status_label = tk.Label(app, text="Nenhuma imagem selecionada")
    status_label.pack(pady=(4, 0))

    img_label = tk.Label(app)
    img_label.pack(padx=8, pady=8)

    result_label = tk.Label(app, text="Predição: -", font=("Arial", 14))
    result_label.pack(pady=(0, 8))

    def load_and_show(path: str):
        """Exibe imagem e limpa texto de predição anterior."""
        try:
            pil = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Erro", f"Não foi possível abrir a imagem:\n{e}")
            return
        w, h = pil.size
        scale = min(DISPLAY_MAX / w, DISPLAY_MAX / h, 1)
        display = pil.resize((int(w * scale), int(h * scale)), Image.Resampling.BICUBIC)
        tkimg = ImageTk.PhotoImage(display)
        img_label.config(image=tkimg)
        img_label.image = tkimg
        app.selected_path = path
        status_label.config(text=f"Arquivo: {os.path.basename(path)}")
        result_label.config(text="Predição: -")

    def open_file():
        """Abre diálogo para o usuário escolher uma imagem e chama load_and_show."""
        path = filedialog.askopenfilename(filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if path:
            load_and_show(path)

    def run_predict():
        """Executa predição na imagem atualmente selecionada."""
        path = getattr(app, "selected_path", None)
        if not path:
            messagebox.showwarning("Aviso", "Nenhuma imagem selecionada.")
            return
        try:
            pred = predict_image(path, model, transform, class_names, device)
            result_label.config(text=f"Predição: {pred}")
        except Exception as e:
            messagebox.showerror("Erro na predição", str(e))

    btn_open = tk.Button(top_frame, text="Abrir Imagem...", command=open_file)
    btn_open.grid(row=0, column=0, padx=4)

    btn_predict = tk.Button(top_frame, text="Executar Predição", command=run_predict)
    btn_predict.grid(row=0, column=1, padx=4)

    app.mainloop()


if __name__ == "__main__":
    device = set_device(DEVICE_PREFER_GPU)
    try:
        model = build_model(MODEL_PATH, NUM_CLASSES, device)
    except Exception as e:
        messagebox.showerror("Erro ao carregar modelo", f"Não foi possível carregar o modelo:\n{e}")
        raise

    transform = prepare_transform(TARGET_SIZE)
    create_gui(model, transform, CLASS_NAMES, device)