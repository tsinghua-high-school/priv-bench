import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader,Subset, random_split
from torchvision import transforms
import os
from tqdm import tqdm
from util import modified_resnet50
from PIL import Image
from sklearn.model_selection import train_test_split
from opacus.validators import ModuleValidator

class MetaEncoder(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()
        
        # For activation maps: FC encoder after flattening
        self.activation_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(C * 7 * 7, 128),  # You can use adaptive avg pool if needed
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64), 
            )
            for C in [2048, 512, 512, 512, 512, 2048, 2048, 2048]  # Match activation channels
        ])
        
        # For gradient maps: CNN-based encoder
        self.gradient_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(C, 64, kernel_size=1, stride=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            for C in [2048, 2048, 2048, 512, 512, 512, 512, 2048]
        ])

        # Logits FCNs
        self.logits_acts_fc = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.logits_grads_fc = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Label encoder: one-hot (12D) → FC
        self.label_encoder = nn.Sequential(
            nn.Linear(num_classes, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Loss encoder: scalar → FC
        self.loss_encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Final classifier
        total_dim = (64 + 64) * 8 + 64 + 64 + 12 + 12  # act+grad + label + loss + 2x logits
        self.final_fc = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, activations, gradients, label, loss):
        act_feats = [encoder(act) for encoder, act in zip(self.activation_encoders, activations[:-2])]
        grad_feats = [encoder(grad) for encoder, grad in zip(self.gradient_encoders, gradients[2:])]

        logits_acts = activations[-2]
        logits_grads = gradients[0]

        logits_acts_encoded = self.logits_acts_fc(logits_acts)
        logits_grads_encoded = self.logits_grads_fc(logits_grads)

        # One-hot encode label if it's not already
        if label.dim() == 0:
            label = F.one_hot(label, num_classes=12).float()
        label_encoded = self.label_encoder(label)

        # Loss: ensure it's a tensor of shape [1]
        if not isinstance(loss, torch.Tensor):
            loss = torch.tensor([loss], dtype=torch.float32, device=label.device)
        elif loss.dim() == 0:
            loss = loss.unsqueeze(0)

        batch_size = label.size(0)
        loss_expanded = loss.repeat(batch_size,1)
        loss_encoded = self.loss_encoder(loss_expanded)

        act_concat = torch.cat(act_feats, dim=1)   # shape: [32, 512] (8*64)
        grad_concat = torch.cat(grad_feats, dim=1) # shape: [32, 512] (8*64)

        combined_feats = torch.cat([act_concat, grad_concat, label_encoded, loss_encoded, logits_acts, logits_grads], dim=1)
        return self.final_fc(combined_feats)

class MembershipDataset(Dataset):
    def __init__(self, member_dir, non_member_dir, transform=None):
        self.member_dir = member_dir
        self.non_member_dir = non_member_dir
        self.transform = transform

        self.samples = []

        # Load member images with label 1
        for fname in os.listdir(member_dir):
            fpath = os.path.join(member_dir, fname)
            if os.path.isfile(fpath) and self._is_image(fname):
                self.samples.append((fpath, 1))

        # Load non-member images with label 0
        for fname in os.listdir(non_member_dir):
            fpath = os.path.join(non_member_dir, fname)
            if os.path.isfile(fpath) and self._is_image(fname):
                self.samples.append((fpath, 0))

    def _is_image(self, filename):
        return any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif'])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def register_last_n_hooks(model, n=10):
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    # Get all leaf modules (with parameters)
    layers_with_params = [
        m for m in model.modules()
        if any(p.requires_grad for p in m.parameters())
    ]

    last_layers = layers_with_params[-n:]

    handles = []
    for layer in last_layers:
        handles.append(layer.register_forward_hook(forward_hook))
        handles.append(layer.register_backward_hook(backward_hook))

    return activations, gradients, handles

def train_encoder(encoder, target_model, train_loader, device, epochs=10):
    encoder.train()
    target_model.eval()  # Freeze behavior like dropout, BN, etc.
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    batch_size = 32  
    dummy_label = F.one_hot(torch.zeros(batch_size, dtype=torch.long), num_classes=12).float().to(device)

    for epoch in range(epochs):
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device).float().unsqueeze(1)
            activations, gradients, handles = register_last_n_hooks(target_model)

            logits = target_model(x)
            loss = F.binary_cross_entropy_with_logits(logits, dummy_label)
            target_model.zero_grad()
            loss.backward()

            #pass the activations and gradients to the encoder
            pred_logits = encoder(activations, gradients, dummy_label, loss.item())
            encoder_loss = F.binary_cross_entropy_with_logits(pred_logits, y)

            # Backprop and update encoder
            optimizer.zero_grad()
            encoder_loss.backward()
            optimizer.step()
            for h in handles:
                h.remove()

        print(f"[Epoch {epoch}] Loss: {loss.item():.4f}")
        
    return

def get_activation_shapes(model, device):
    model.eval() 
    # Dummy batch
    x_dummy = torch.randn(1, 3, 224, 224).to(device)
    y_dummy = torch.tensor([0]).to(device)

    # Collectors
    activations = []
    gradients = []
    activations, gradients, handles = register_last_n_hooks(target_model)

    # Forward and backward
    x_dummy.requires_grad = True
    logits = model(x_dummy)
    loss = F.cross_entropy(logits, y_dummy)
    model.zero_grad()
    loss.backward()
    for h in handles:
        h.remove()
    return activations, gradients


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ])

    #load dataset
    dataset = MembershipDataset(
        member_dir='./results+data/Protest-Membership/UCLA',
        non_member_dir='./results+data/Protest-Membership/VGKG',
        transform=transform
    )
    # Extract labels for stratification
    labels = [label for _, label in dataset.samples]

    # Perform stratified split
    train_indices, test_indices = train_test_split(
        range(len(dataset)),
        test_size=0.2,
        stratify=labels,
        random_state=42  # fixed seed for reproducibility
    )

    # Create subset datasets
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
        
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # Define 4 target model
    model_ckpts = {
        "ucla": "./results+data/downstream/model_best_ucla.pth.tar", # Real data model
        "cond": "./results+data/downstream/model_best_cond.pth.tar",
        "dpsgd1": "./results+data/downstream/model_best_dpsgd1.pth.tar",
        "dpsgd10": "./results+data/downstream/model_best_dpsgd10.pth.tar"
    }

    os.makedirs("trained_encoders", exist_ok=True)

    for model_name, ckpt_path in model_ckpts.items():
        print(f"\n=== Training encoder for: {model_name} ===")
        
        # Load model and weights
        target_model = modified_resnet50().to(device)
        if model_name == "dpsgd1" or model_name == "dpsgd10":
            target_model = ModuleValidator.fix(target_model)
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint['state_dict']
        new_state_dict = {k.replace('_module.', ''): v for k, v in state_dict.items()} # Fix for models trained with Opacus
        target_model.load_state_dict(new_state_dict)
        print(f"Loaded weights from {ckpt_path}")

        # Get activation shapes (optional, for validation/logging)
        activations, gradients = get_activation_shapes(target_model, device)

        # Initialize encoder
        encoder = MetaEncoder().to(device)

        # Train encoder
        train_encoder(encoder, target_model, train_loader, device, epochs=25)

        # Save encoder
        encoder_save_path = f"trained_encoders/encoder_{model_name}.pth"
        torch.save(encoder.state_dict(), encoder_save_path)
        print(f"Saved trained encoder to {encoder_save_path}")