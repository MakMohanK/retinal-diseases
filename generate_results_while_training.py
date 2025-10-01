import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_auc_score, f1_score
import torch

# Create results directory
os.makedirs("results_img", exist_ok=True)

# ----------------------------
# TRACK METRICS DURING TRAINING
# ----------------------------
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

def calculate_accuracy(outputs, labels):
    """Multi-label accuracy: threshold sigmoid predictions at 0.5"""
    preds = (torch.sigmoid(outputs) > 0.5).int()
    correct = (preds == labels.int()).sum().item()
    total = torch.numel(labels)
    return correct / total

# (In training loop, append metrics)
# Example:
# acc = calculate_accuracy(outputs, labels)
# batch_accs.append(acc)

# ----------------------------
# AFTER TRAINING - GENERATE PLOTS
# ----------------------------
def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(8,6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.savefig("results_img/loss_curve.png")
    plt.close()

def plot_accuracy(train_accs, val_accs):
    plt.figure(figsize=(8,6))
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Val Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.savefig("results_img/accuracy_curve.png")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Multi-label confusion matrix per class"""
    cms = multilabel_confusion_matrix(y_true, y_pred)
    for i, cm in enumerate(cms):
        plt.figure(figsize=(4,4))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix - {class_names[i]}")
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ["0","1"])
        plt.yticks(tick_marks, ["0","1"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(f"results_img/conf_matrix_class_{class_names[i]}.png")
        plt.close()

# ----------------------------
# FEATURE IMPORTANCE WITH GRADCAM
# ----------------------------
from torchvision.transforms.functional import to_pil_image
import cv2

def generate_gradcam(model, image_tensor, target_class=None, layer_name="features.6.3"):
    """Generate Grad-CAM heatmap for one image"""
    model.eval()
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    # Register hooks
    layer = dict([*model.named_modules()])[layer_name]
    fh = layer.register_forward_hook(forward_hook)
    bh = layer.register_backward_hook(backward_hook)

    # Forward
    output = model(image_tensor.unsqueeze(0))
    if target_class is None:
        target_class = output.argmax(1).item()
    loss = output[0, target_class]

    # Backward
    model.zero_grad()
    loss.backward()

    # Grad-CAM calculation
    grads = gradients[0].mean(dim=[2,3], keepdim=True)
    cam = (grads * activations[0]).sum(dim=1).squeeze()
    cam = torch.relu(cam)
    cam = cam / cam.max()

    # Resize to image size
    cam = cam.cpu().numpy()
    cam = cv2.resize(cam, (image_tensor.shape[2], image_tensor.shape[1]))
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay
    img = to_pil_image(image_tensor.cpu())
    heatmap_img = cv2.addWeighted(np.array(img), 0.6, heatmap, 0.4, 0)
    
    fh.remove()
    bh.remove()

    return heatmap_img

# Example after training:
# plot_loss(train_losses, val_losses)
# plot_accuracy(train_accuracies, val_accuracies)
# plot_confusion_matrix(y_true, y_pred, class_names)
# heatmap = generate_gradcam(model, sample_image, target_class=0)
# plt.imsave("results_img/gradcam_example.png", heatmap)
