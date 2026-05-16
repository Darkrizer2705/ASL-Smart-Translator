# src/llm/gan_augment.py
"""
GAN-inspired data augmentation for hand landmarks.
Uses a Generator network to create synthetic landmark samples
and a Discriminator to ensure they look realistic.
This augments your training data for better generalization.
"""
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import os

FEATURE_SIZE = 63   # 21 landmarks × 3 coords

# ── Generator ──────────────────────────────────────
class LandmarkGenerator(nn.Module):
    """
    Generates synthetic hand landmark vectors.
    Takes random noise + class label → realistic landmarks.
    """
    def __init__(self, noise_dim=32, num_classes=25,
                 output_dim=FEATURE_SIZE):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, 16)
        self.model = nn.Sequential(
            nn.Linear(noise_dim + 16, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_emb   = self.label_embed(labels)
        x           = torch.cat([noise, label_emb], dim=1)
        return self.model(x)


# ── Discriminator ──────────────────────────────────
class LandmarkDiscriminator(nn.Module):
    """
    Distinguishes real landmarks from generated ones.
    Also conditioned on class label.
    """
    def __init__(self, num_classes=25,
                 input_dim=FEATURE_SIZE):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, 16)
        self.model = nn.Sequential(
            nn.Linear(input_dim + 16, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, landmarks, labels):
        label_emb = self.label_embed(labels)
        x         = torch.cat([landmarks, label_emb], dim=1)
        return self.model(x)


# ── Training ───────────────────────────────────────
def train_gan(csv_path: str, epochs: int = 100,
              batch_size: int = 64, noise_dim: int = 32):
    """
    Trains the conditional GAN on your landmark dataset.
    """
    # Load data
    df      = pd.read_csv(csv_path)
    df["label"] = df["label"].str.lower().str.strip()
    labels  = sorted(df["label"].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    num_classes = len(labels)

    X = df.drop("label", axis=1).values.astype(np.float32)
    y = np.array([label2id[l] for l in df["label"]])

    # Normalize to [-1, 1] for Tanh output
    X_min, X_max = X.min(), X.max()
    X_norm = 2 * (X - X_min) / (X_max - X_min) - 1

    # Models
    G = LandmarkGenerator(noise_dim, num_classes)
    D = LandmarkDiscriminator(num_classes)

    g_opt = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    loss_fn = nn.BCELoss()

    print(f"Training GAN - {num_classes} classes, {len(X)} samples")

    for epoch in range(epochs):
        # Random batch
        idx   = np.random.randint(0, len(X), batch_size)
        real  = torch.tensor(X_norm[idx])
        lbls  = torch.tensor(y[idx], dtype=torch.long)

        # ── Train Discriminator ──────────────────
        noise   = torch.randn(batch_size, noise_dim)
        fake    = G(noise, lbls).detach()

        real_loss = loss_fn(D(real, lbls), torch.ones(batch_size, 1))
        fake_loss = loss_fn(D(fake, lbls), torch.zeros(batch_size, 1))
        d_loss    = (real_loss + fake_loss) / 2

        d_opt.zero_grad()
        d_loss.backward()
        d_opt.step()

        # ── Train Generator ──────────────────────
        noise  = torch.randn(batch_size, noise_dim)
        fake   = G(noise, lbls)
        g_loss = loss_fn(D(fake, lbls), torch.ones(batch_size, 1))

        g_opt.zero_grad()
        g_loss.backward()
        g_opt.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"D Loss: {d_loss.item():.4f} | "
                  f"G Loss: {g_loss.item():.4f}")

    # Save models
    os.makedirs("models", exist_ok=True)
    torch.save({
        "G": G.state_dict(),
        "D": D.state_dict(),
        "label2id": label2id,
        "X_min": X_min,
        "X_max": X_max,
        "noise_dim": noise_dim
    }, "models/landmark_gan.pt")
    print("GAN saved: models/landmark_gan.pt")
    return G, label2id, X_min, X_max


def generate_samples(label: str, n_samples: int = 50) -> np.ndarray:
    """
    Generates synthetic landmark samples for a given phrase.
    Use this to augment your training data.
    """
    checkpoint = torch.load("models/landmark_gan.pt",
                            weights_only=False)
    label2id   = checkpoint["label2id"]
    X_min      = checkpoint["X_min"]
    X_max      = checkpoint["X_max"]
    noise_dim  = checkpoint["noise_dim"]
    num_classes = len(label2id)

    G = LandmarkGenerator(noise_dim, num_classes)
    G.load_state_dict(checkpoint["G"])
    G.eval()

    label_id = label2id[label.lower()]
    noise    = torch.randn(n_samples, noise_dim)
    labels   = torch.tensor([label_id] * n_samples, dtype=torch.long)

    with torch.no_grad():
        fake_norm = G(noise, labels).numpy()

    # Denormalize
    fake = (fake_norm + 1) / 2 * (X_max - X_min) + X_min
    return fake


def augment_dataset(csv_path: str, output_path: str,
                    samples_per_class: int = 100):
    """
    Augments your existing dataset with GAN-generated samples.
    """
    df      = pd.read_csv(csv_path)
    df["label"] = df["label"].str.lower().str.strip()
    labels  = sorted(df["label"].unique())
    cols    = [c for c in df.columns if c != "label"]

    augmented_rows = []
    print(f"Augmenting {len(labels)} phrases x {samples_per_class} samples")

    for label in labels:
        fake = generate_samples(label, samples_per_class)
        for row in fake:
            augmented_rows.append(
                {**dict(zip(cols, row)), "label": label}
            )

    aug_df   = pd.DataFrame(augmented_rows)
    combined = pd.concat([df, aug_df], ignore_index=True)
    combined.to_csv(output_path, index=False)
    print(f"Augmented dataset saved: {output_path}")
    print(f"   Original: {len(df)} rows")
    print(f"   Added:    {len(aug_df)} rows")
    print(f"   Total:    {len(combined)} rows")
    return combined


if __name__ == "__main__":
    CSV = "datasets/gesture_dataset.csv"

    print("Step 1: Training GAN on your landmark data...")
    train_gan(CSV, epochs=100)

    print("\nStep 2: Generating augmented dataset...")
    augment_dataset(
        CSV,
        "datasets/gesture_dataset_augmented.csv",
        samples_per_class=100
    )
