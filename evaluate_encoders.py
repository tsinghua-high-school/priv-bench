import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import os
from tqdm import tqdm
from PIL import Image
from util import modified_resnet50
from opacus.validators import ModuleValidator
from sklearn.model_selection import train_test_split
from white_box2 import MetaEncoder, MembershipDataset, register_last_n_hooks
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, precision_score, recall_score
import numpy as np

def evaluate_encoder(encoder, target_model, data_loader, device):
    encoder.eval()
    target_model.eval()
    
    all_labels = []
    all_probs = []
    all_preds = []

    #with torch.no_grad():
    for x, y in tqdm(data_loader, desc="Evaluating"):
        x = x.to(device)
        y = y.to(device).float().unsqueeze(1)
        batch_size = x.size(0)

        activations, gradients, handles = register_last_n_hooks(target_model)

        dummy_label = F.one_hot(torch.zeros(batch_size, dtype=torch.long), num_classes=12).float().to(device)
        logits = target_model(x)
        loss = F.binary_cross_entropy_with_logits(logits, dummy_label)
        target_model.zero_grad()
        loss.backward()

        pred_logits = encoder(activations, gradients, dummy_label, loss.item())
        probs = torch.sigmoid(pred_logits)

        all_labels.extend(y.cpu().numpy())
        all_probs.extend(probs.cpu().detach().numpy())
        all_preds.extend((probs > 0.5).int().cpu().detach().numpy())

        for h in handles:
            h.remove()

    # Flatten to 1D arrays
    y_true = np.array(all_labels).flatten()
    y_prob = np.array(all_probs).flatten()
    y_pred = np.array(all_preds).flatten()

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    ll = log_loss(y_true, y_prob)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)

    return acc, auc, ll, prec, rec


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset = MembershipDataset(
        member_dir='./results+data/Protest-Membership/UCLA',
        non_member_dir='./results+data/Protest-Membership/VGKG',
        transform=transform
    )
    labels = [label for _, label in dataset.samples]
    train_indices, test_indices = train_test_split(
        range(len(dataset)),
        test_size=0.2,
        stratify=labels,
        random_state=42  # fixed seed for reproducibility
    )
    test_dataset = Subset(dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model_ckpts = {
        "ucla": "./results+data/downstream/model_best_ucla.pth.tar",
        "cond": "./results+data/downstream/model_best_cond.pth.tar",
        "dpsgd1": "./results+data/downstream/model_best_dpsgd1.pth.tar",
        "dpsgd10": "./results+data/downstream/model_best_dpsgd10.pth.tar"
    }

    for model_name, ckpt_path in model_ckpts.items():
        print(f"\n=== Evaluating encoder for: {model_name} ===")
        
        # Load target model
        target_model = modified_resnet50().to(device)
        if model_name.startswith("dpsgd"):
            target_model = ModuleValidator.fix(target_model)

        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint['state_dict']
        new_state_dict = {k.replace('_module.', ''): v for k, v in state_dict.items()}
        target_model.load_state_dict(new_state_dict)

        # Load encoder
        encoder_path = f"trained_encoders/encoder_{model_name}.pth"
        encoder = MetaEncoder().to(device)
        encoder.load_state_dict(torch.load(encoder_path, map_location=device))

        # Evaluate
        acc, auc, ll, prec, rec = evaluate_encoder(encoder, target_model, test_loader, device)

        print(f"[{model_name}] Accuracy: {acc:.4f} | AUC: {auc:.4f} | LogLoss: {ll:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")