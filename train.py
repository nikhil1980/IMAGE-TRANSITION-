#!/bin/python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tqdm import tqdm
from alive_progress import alive_bar
import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image, ImageEnhance
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import BertTokenizer, BertModel
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
from transformers import SwinForImageClassification
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
import warnings
import transformers


warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()

'''
Code to Train Contextual Emotion Detection Solution
'''

# For LR scheduling
# Implement learning rate scheduling to reduce the learning rate as the paligemma-weights starts converging.
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Set up logging
logging.basicConfig(filename='training_log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

# Give path to DS
DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DATASET')
MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MODEL')

csv_filename = 'labels.csv'

num_epochs = 20

# Define Ekman's basic 7 emotions + neutral
emotions2idx = {'anger': 0,
                'contempt': 1,
                'disgust': 2,
                'fear': 3,
                'happy': 4,
                'neutral': 5,
                'sad': 6,
                'surprise': 7}

idx2emotions = {0: 'anger',
                1: 'contempt',
                2: 'disgust',
                3: 'fear',
                4: 'happy',
                5: 'neutral',
                6: 'sad',
                7: 'surprise'}

# Define Transition
transitions2idx = {'crossfade': 0,
                   'fade_out': 1,
                   'fast_crossfade': 2,
                   'fast_washout': 3,
                   'fast_zoom_in': 4,
                   'fast_zoom_out': 5,
                   'quick_zoom_out': 6,
                   'slide_left': 7,
                   'slide_right': 8,
                   'slow_fade_out': 9,
                   'simple_crossfade': 10,
                   'washout': 11,
                   'zoom_in': 12,
                   'zoom_out': 13}

idx2transitions = {0: 'crossfade',
                   1: 'fade_out',
                   2: 'fast_crossfade',
                   3: 'fast_washout',
                   4: 'fast_zoom_in',
                   5: 'fast_zoom_out',
                   6: 'quick_zoom_out',
                   7: 'slide_left',
                   8: 'slide_right',
                   9: 'slow_fade_out',
                   10: 'simple_crossfade',
                   11: 'washout',
                   12: 'zoom_in',
                   13: 'zoom_out'}

# Intensity threshold for transitions
INTENSITY_THRESHOLD = 0.6

prompt = "Describe the emotional context and key elements of the image in 100 words"


# Function to map emotion and intensity to a transition type
def map_emotion_to_transition(emotion, intensity):
    if emotion == 'happy':
        return 'fast_zoom_in' if intensity > INTENSITY_THRESHOLD else 'zoom_in'
    elif emotion == 'sad':
        return 'slow_fade_out' if intensity < INTENSITY_THRESHOLD else 'fade_out'
    elif emotion == 'fear':
        return 'fast_zoom_out' if intensity > INTENSITY_THRESHOLD else 'zoom_out'
    elif emotion == 'disgust':
        return 'slide_left' if intensity > INTENSITY_THRESHOLD else 'slide_right'
    elif emotion == 'anger':
        return 'quick_zoom_out' if intensity > INTENSITY_THRESHOLD else 'zoom_out'
    elif emotion == 'surprise':
        return 'fast_crossfade' if intensity > INTENSITY_THRESHOLD else 'crossfade'
    elif emotion == 'contempt':
        return 'fast_washout' if intensity > INTENSITY_THRESHOLD else 'washout'
    elif emotion == 'neutral':
        return 'simple_crossfade'

    return 'crossfade'  # Default transition


def add_transition():
    """
    Function to add transition to AffectNet DS
    :return:
    """
    # Load AffectNet dataset (assuming it's in CSV format)
    file_path = os.path.join(DATASET, csv_filename)
    affectnet_df = pd.read_csv(file_path)

    # Add a new column 'transition' to store the transition labels
    transitions = []

    for index, row in affectnet_df.iterrows():
        emotion = row['emotion']  # Assuming 'emotion' column contains emotion label
        intensity = row['intensity']  # Assuming 'intensity' column contains intensity value (0-1)

        # Map emotion and intensity to transition
        transition = map_emotion_to_transition(emotion, intensity)

        # Append the transition to the list
        transitions.append(transition)

    # Add the new 'transition' column to the DataFrame
    affectnet_df['transition'] = transitions

    # Save the updated DataFrame to a new CSV file
    affectnet_df.to_csv(os.path.join(DATASET, csv_filename), index=False)

    # Preview the first few rows to verify the changes
    # print(affectnet_df.head())


def get_train_augmentations():
    return A.Compose([
        A.Resize(height=224, width=224, p=1),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.ColorJitter(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def get_val_augmentations():
    return A.Compose([
        A.Resize(height=224, width=224, p=1),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


# Load BLIP processor and paligemma-weights
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

"""caption_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
caption_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    device_map="cpu",
)"""


# Function to generate caption for a given image
def generate_caption(image_path, prompt):
    image = Image.open(image_path).convert("RGB")
    inputs = caption_processor(images=image, return_tensors="pt")

    caption_model.eval()

    # Dynamic quantization (weights only)
    quantized_model = torch.quantization.quantize_dynamic(caption_model, {torch.nn.Linear}, dtype=torch.qint8)

    caption_ids = quantized_model.generate(**inputs,
                                           num_beams=3,  # Set 5 paths ast each stage
                                           repetition_penalty=1.2,  # Model get stuck so use [1.2-1.5]
                                           max_length=100,
                                           top_k=50,
                                           #top_p=0.9,  # Most probable tokens whose cumulative probability is less than or equal to top_p
                                           temperature=0.7,  # Set low to make it more deterministic
                                           length_penalty=1.1,  # Penalty to the length of the generated sequence with positive values encourage longer sequences
                                           early_stopping=False,  # Let paligemma-weights generate best caption
                                           max_new_tokens=100,
                                           no_repeat_ngram_size=5,  # Stop generating n-grams (sequences of n words) already given
                                           min_length=35)
    caption = caption_processor.decode(caption_ids[0], skip_special_tokens=True).strip()
    print(caption)
    # Remove all words that start with "stock" or contain metadata like characters.
    import re
    # Remove non-alphanumeric characters except common punctuation
    caption = re.sub(r'[^\w\s.,!?]', '', caption)
    # Remove repeated "stock" phrases
    caption = re.sub(r'\s*stock.*$', '', caption, flags=re.IGNORECASE)
    # Remove multiple spaces
    caption = re.sub(r'\s+', ' ', caption).strip()

    # Combine the prompt with the generated caption
    detailed_caption = f"{prompt}: {caption}"
    print(detailed_caption)

    return caption


class AffectNetDataset(Dataset):
    def __init__(self, df, tokenizer, transform=None, generate_captions=False,
                 prompt="Describe the emotional context and key elements of the image."):
        self.df = df
        self.tokenizer = tokenizer
        self.transform = transform
        self.generate_captions = generate_captions
        self.prompt = prompt

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_name = os.path.join(DATASET, row['image_path'])
        image = Image.open(file_name).convert("RGB")
        caption = row.get('caption', None)

        # Generate caption if not available and generate_captions is True
        if caption is None and self.generate_captions:
            caption = generate_caption(file_name, self.prompt)
            print(file_name, " :: ", caption)
            row['caption'] = caption

        emotion_label = row['emotion']
        intensity = row['intensity']
        transition = row['transition']

        if self.transform:
            import numpy as np
            image = self.transform(image=np.array(image))['image']

        tokens = self.tokenizer(caption,
                                return_tensors="pt",
                                padding='max_length',
                                max_length=128,
                                truncation=True)
        return {
            'image': image,
            'caption_tokens': tokens,
            'emotion_label': torch.tensor(emotion_label, dtype=torch.long),
            'intensity': torch.tensor(intensity, dtype=torch.float),
            'transition': torch.tensor(transition, dtype=torch.long)
        }


def preprocess_image(image_path, size=(224, 224)):
    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0)  # Add batch dimension


class MultimodalModel(nn.Module):
    def __init__(self, image_encoder, text_encoder, num_transitions):
        super(MultimodalModel, self).__init__()
        self.image_encoder = image_encoder

        # Modify the classifier to match the desired number of classes
        self.image_encoder.classifier = torch.nn.Linear(self.image_encoder.config.hidden_size,
                                                        len(emotions2idx))

        self.text_encoder = text_encoder
        self.transition_classifier = nn.Linear(len(emotions2idx) + 768,
                                               num_transitions)  # Adjust this based on feature sizes

    def forward(self, image, caption_tokens):
        img_features = self.image_encoder(image).logits
        image_features = img_features.mean(dim=1)  # Pooling over sequence dimension
        print("Pooled image features shape:", image_features.shape)

        ''' Text '''
        # Extract the input_ids and attention_mask from caption_tokens
        input_ids = caption_tokens["input_ids"].squeeze(1)
        attention_mask = caption_tokens["attention_mask"].squeeze(1)

        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state.mean(dim=1)
        print("Pooled Text features shape:", text_features.shape)

        combined_features = torch.cat((img_features, text_features), dim=1)
        print("Combined features shape:", combined_features.shape)

        # Classify into transitions
        transition_logits = self.transition_classifier(combined_features)
        return transition_logits


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Ensure inputs are in the correct shape [batch_size, num_classes]
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # Flatten the last dimensions
            inputs = inputs.transpose(1, 2)  # Transpose to [batch_size, -1, num_classes]
            inputs = inputs.contiguous().view(-1, inputs.size(-1))

        # Flatten the targets as well
        targets = targets.view(-1)

        # Check if inputs and targets have matching shapes
        assert inputs.size(0) == targets.size(0), "Input and target size mismatch"

        # Ensure targets are valid
        num_classes = inputs.size(1)
        if targets.min().item() < 0 or targets.max().item() >= num_classes:
            raise ValueError(f"Target values are out of range. Expected values between 0 and {num_classes - 1}, "
                             f"but got min={targets.min().item()}, max={targets.max().item()}")

        # Compute log probabilities
        log_p = F.log_softmax(inputs, dim=-1)
        p = torch.exp(log_p)

        # Gather the log probabilities corresponding to the targets
        log_p = log_p.gather(1, targets.unsqueeze(1))
        log_p = log_p.view(-1)

        # Apply the focal loss formula
        loss = -self.alpha * ((1 - p.gather(1, targets.unsqueeze(1)).view(-1)) ** self.gamma) * log_p

        # Apply reduction method
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def log_metrics(epoch, train_loss, val_accuracy, val_f1, val_cm):
    logging.info(f'Epoch {epoch + 1} - Train Loss: {train_loss:.4f}')
    logging.info(f'Epoch {epoch + 1} - Validation Accuracy: {val_accuracy:.4f}')
    logging.info(f'Epoch {epoch + 1} - Validation F1 Score: {val_f1:.4f}')
    logging.info(f'Epoch {epoch + 1} - Validation Confusion Matrix:\n{val_cm}')


scaler = GradScaler()


# @torch.compile
def train_one_epoch(model, dataloader, optimizer, loss_fn, scheduler, device):
    model.train()
    total_loss = 0
    loop = tqdm(enumerate(dataloader), total=len(dataloader))

    for idx, batch in loop:
        optimizer.zero_grad()
        image, caption_tokens, emotion_label, intensity, transition = (
            batch['image'].to(device),
            {k: v.to(device) for k, v in batch['caption_tokens'].items()},
            batch['emotion_label'].to(device),
            batch['intensity'].to(device),
            batch['transition'].to(device),
        )

        print(f"For batch: {idx}, unique labels: {torch.unique(transition)}")

        with autocast():
            predictions = model(image, caption_tokens)
            print("Predictions shape:", predictions.shape)
            print("Labels shape:", transition.shape)

            num_classes = predictions.size(1)
            print("Targets min:", transition.min().item())
            print("Targets max:", transition.max().item())
            assert transition.min().item() >= 0, "Found negative class index in targets"
            assert transition.max().item() < num_classes, f"Found class index in targets that exceeds the number of classes ({num_classes})"
            loss = loss_fn(predictions, transition)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()
        total_loss += loss.item()

        # Progress the bar
        loop.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)


def predict(model, tokenizer, image_path):
    model.eval()
    with torch.no_grad():
        # Preprocess the image
        image_tensor = preprocess_image(image_path)

        # Get caption
        caption = generate_caption(image_path, prompt=prompt)

        # Tokenize the caption
        caption_tokens = tokenizer(caption,
                                   return_tensors="pt",
                                   padding=True,
                                   truncation=True,
                                   max_length=128)
        transition_logits = model(image_tensor, caption_tokens)

    # Get predicted transition and emotion
    predicted_transition_id = transition_logits.argmax(dim=1).item()
    print(predicted_transition_id, idx2transitions.get(predicted_transition_id))
    return idx2transitions.get(predicted_transition_id)


def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            image, caption_tokens, transition = (
                batch['image'].to(device),
                {k: v.to(device) for k, v in batch['caption_tokens'].items()},
                batch['transition'].to(device)
            )
            preds = model(image, caption_tokens).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(transition.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    return accuracy, f1, cm


def main():
    # Add transition
    add_transition()

    # Load dataset
    df = pd.read_csv(os.path.join(DATASET, csv_filename))

    # Fit and transform the emotion labels to numeric values
    # df['emotion'] = emotion_label_encoder.fit_transform(df['emotion'])
    df['emotion'] = df['emotion'].map(emotions2idx)

    # Fit and transform the transition labels to numeric values
    # df['transition'] = transition_encoder.fit_transform(df['transition'])
    df['transition'] = df['transition'].map(transitions2idx)

    print(df.head())

    # Split the dataset into training and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                              clean_up_tokenization_spaces=True)

    # Example: Enable caption generation when initializing the dataset
    train_dataset = AffectNetDataset(train_df, tokenizer,
                                     transform=get_train_augmentations(),
                                     generate_captions=True,
                                     prompt=prompt
                                     )

    train_loader = DataLoader(train_dataset,
                              batch_size=32,
                              shuffle=True,
                              num_workers=0)

    # Create Val datasets and dataloaders
    val_dataset = AffectNetDataset(val_df,
                                   tokenizer,
                                   transform=get_val_augmentations(),
                                   generate_captions=True,
                                   prompt=prompt)

    val_loader = DataLoader(val_dataset,
                            batch_size=16,
                            shuffle=False,
                            num_workers=0)

    image_encoder = SwinForImageClassification.from_pretrained("microsoft/swin-base-patch4-window7-224")
    # Use a pretrained BERT paligemma-weights as the text encoder.
    text_encoder = BertModel.from_pretrained('bert-base-uncased')

    # Define the paligemma-weights, loss function, and optimizer
    model = MultimodalModel(image_encoder=image_encoder,
                            text_encoder=text_encoder,
                            num_transitions=len(transitions2idx)).to(device)

    loss_fn = FocalLoss(alpha=1, gamma=2)

    # Use cosine annealing with warm restarts for LR scheduling
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)

    print(f"--->...Started training...<---\n")
    with alive_bar(num_epochs, bar='bubbles', spinner='notes2', force_tty=True) as bar:
        for epoch in range(num_epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, scheduler, device)
            val_accuracy, val_f1, val_cm = evaluate(model, val_loader, device)
            log_metrics(epoch, train_loss, val_accuracy, val_f1, val_cm)

            # Print progress to console
            print(f'Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}')
            print(f'Epoch {epoch + 1}/{num_epochs} - Validation Accuracy: {val_accuracy:.4f}')
            print(f'Epoch {epoch + 1}/{num_epochs} - Validation F1 Score: {val_f1:.4f}')
            print(f'Epoch {epoch + 1}/{num_epochs} - Validation Confusion Matrix:\n{val_cm}')
            bar()

    # Save the training log
    print("Training completed. Logs are saved to 'training_log.txt'.")

    # Save the trained paligemma-weights
    model_save_path = os.path.join(MODEL, "multimodal_transition_model.pth")
    torch.save(model.state_dict(), model_save_path)
    # scripted_model = torch.jit.script(paligemma-weights)
    # scripted_model.save("multimodal_transition_model.pt")

    # Load the paligemma-weights
    model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))

    # Inference: Call Predict() on sample image
    model = model.to('cpu')
    # Put the paligemma-weights in evaluation mode (if you're doing inference)
    model.eval()

    # Inference: Call Predict() on sample image
    image_path = '/home/nikhil/source_code/IMAGE_TRANSITION/DATASET/anger/image0000006.jpg'
    predict(model, tokenizer, image_path)

    print('\n...Image Transition Model training over...\n')


# Utility function to save a sequence of images as GIF
def save_gif(images, duration, output_path):
    images[0].save(output_path, save_all=True, append_images=images[1:], duration=duration, loop=0)


# Crossfade (slower fade with gradual blend)
def crossfade(image, duration=500):
    images = []
    fade_image = Image.new("RGB", image.size, (255, 255, 255))
    for i in range(10):
        blend = Image.blend(image, fade_image, i / 10)
        images.append(blend)
    save_gif(images, duration // len(images), "crossfade.gif")


# Fade out effect
def fade_out(image, duration=500):
    images = []
    for i in range(10):
        enhancer = ImageEnhance.Brightness(image)
        faded_image = enhancer.enhance(1 - i * 0.1)
        images.append(faded_image)
    save_gif(images, duration // len(images), "fade_out.gif")


# Fast crossfade (same as crossfade but faster)
def fast_crossfade(image, duration=300):
    images = []
    fade_image = Image.new("RGB", image.size, (255, 255, 255))
    for i in range(5):  # Fewer frames for faster transition
        blend = Image.blend(image, fade_image, i / 5)
        images.append(blend)
    save_gif(images, duration // len(images), "fast_crossfade.gif")


# Washout (quick fade to white)
def washout(image, duration=500):
    images = []
    for i in range(10):
        fade_image = Image.new("RGB", image.size, (255, 255, 255))
        blend = Image.blend(image, fade_image, i / 10)
        images.append(blend)
    save_gif(images, duration // len(images), "washout.gif")


# Zoom in effect (gradually zooms in)
def zoom_in(image, duration=500):
    images = []
    w, h = image.size
    for i in range(10):
        scale = 1 + i * 0.05
        resized = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        crop = resized.crop(((resized.width - w) // 2, (resized.height - h) // 2,
                             (resized.width + w) // 2, (resized.height + h) // 2))
        images.append(crop)
    save_gif(images, duration // len(images), "zoom_in.gif")


# Zoom out effect (gradually zooms out)
def zoom_out(image, duration=500):
    images = []
    w, h = image.size
    for i in range(10):
        scale = 1 - i * 0.05
        resized = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        background = Image.new("RGB", (w, h), (0, 0, 0))
        background.paste(resized, ((w - resized.width) // 2, (h - resized.height) // 2))
        images.append(background)
    save_gif(images, duration // len(images), "zoom_out.gif")


# Slide left effect
def slide_left(image, duration=500):
    images = []
    w, h = image.size
    for i in range(10):
        shift = int(w * i * 0.1)
        background = Image.new("RGB", (w, h), (0, 0, 0))
        background.paste(image, (-shift, 0))
        images.append(background)
    save_gif(images, duration // len(images), "slide_left.gif")


# Slide right effect
def slide_right(image, duration=500):
    images = []
    w, h = image.size
    for i in range(10):
        shift = int(w * i * 0.1)
        background = Image.new("RGB", (w, h), (0, 0, 0))
        background.paste(image, (shift, 0))
        images.append(background)
    save_gif(images, duration // len(images), "slide_right.gif")


# Slow fade out (longer fade duration)
def slow_fade_out(image, duration=700):
    images = []
    for i in range(15):  # More frames for slower fade
        enhancer = ImageEnhance.Brightness(image)
        faded_image = enhancer.enhance(1 - i * 0.07)
        images.append(faded_image)
    save_gif(images, duration // len(images), "slow_fade_out.gif")


# Fast zoom in (zoom in effect but faster)
def fast_zoom_in(image, duration=300):
    images = []
    w, h = image.size
    for i in range(5):  # Fewer frames for faster zoom
        scale = 1 + i * 0.1
        resized = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        crop = resized.crop(((resized.width - w) // 2, (resized.height - h) // 2,
                             (resized.width + w) // 2, (resized.height + h) // 2))
        images.append(crop)
    save_gif(images, duration // len(images), "fast_zoom_in.gif")


# Fast washout (similar to washout but quicker)
def fast_washout(image, duration=300):
    images = []
    for i in range(5):  # Fewer frames for faster washout
        fade_image = Image.new("RGB", image.size, (255, 255, 255))
        blend = Image.blend(image, fade_image, i / 5)
        images.append(blend)
    save_gif(images, duration // len(images), "fast_washout.gif")


# Fast zoom out effect (quick zoom out)
def fast_zoom_out(image, duration=300):
    images = []
    w, h = image.size
    for i in range(5):  # Fewer frames for faster zoom
        scale = 1 - i * 0.1
        resized = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        background = Image.new("RGB", (w, h), (0, 0, 0))
        background.paste(resized, ((w - resized.width) // 2, (h - resized.height) // 2))
        images.append(background)
    save_gif(images, duration // len(images), "fast_zoom_out.gif")


# Simple crossfade effect (like crossfade but with fewer frames for a quicker effect)
def simple_crossfade(image, duration=300):
    images = []
    fade_image = Image.new("RGB", image.size, (255, 255, 255))
    for i in range(5):  # Fewer frames for simpler crossfade
        blend = Image.blend(image, fade_image, i / 5)
        images.append(blend)
    save_gif(images, duration // len(images), "simple_crossfade.gif")


# Quick zoom out (similar to zoom out but quicker)
def quick_zoom_out(image, duration=300):
    images = []
    w, h = image.size
    for i in range(5):  # Fewer frames for quicker zoom
        scale = 1 - i * 0.1
        resized = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        background = Image.new("RGB", (w, h), (0, 0, 0))
        background.paste(resized, ((w - resized.width) // 2, (h - resized.height) // 2))
        images.append(background)
    save_gif(images, duration // len(images), "quick_zoom_out.gif")


if __name__ == "__main__":
    # main()

    image_encoder = SwinForImageClassification.from_pretrained("microsoft/swin-base-patch4-window7-224")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', clean_up_tokenization_spaces=True)

    # Use a pretrained BERT paligemma-weights as the text encoder.
    text_encoder = BertModel.from_pretrained('bert-base-uncased')

    # Define the paligemma-weights, loss function, and optimizer
    model = MultimodalModel(image_encoder=image_encoder,
                            text_encoder=text_encoder,
                            num_transitions=len(transitions2idx)).to(device)

    model_save_path = os.path.join(MODEL, "multimodal_transition_model.pth")

    # Load the paligemma-weights
    model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
    model = model.to('cpu')
    # Inference: Call Predict() on sample image
    image_path = '/Users/nikhil1.bhargava/PycharmProjects/ai-offerings/object-detection/input_files/f0efe7b70f4b4a79894fd411e422652fc26d4677406347938ebcbe94a9f5243c/fbf4740fd90e407ab19f81e1c42ba33c.jpg'
    transition = predict(model, tokenizer, image_path)

    # OpenCV implementation
    image = Image.open(image_path)

    # Apply each transition effect and store the result
    if transition == 'crossfade':
        crossfade(image)
    if transition == 'fade_out':
        fade_out(image)
    if transition == 'fast_crossfade':
        fast_crossfade(image)
    if transition == 'fast_washout':
        fast_washout(image)
    if transition == 'fast_zoom_out':
        fast_zoom_out(image)
    if transition == 'fast_zoom_in':
        fast_zoom_in(image)
    if transition == 'quick_zoom_out':
        quick_zoom_out(image)
    if transition == 'slide_left':
        slide_left(image)
    if transition == 'slide_right':
        slide_right(image)
    if transition == 'slow_fade_out':
        slow_fade_out(image)
    if transition == 'simple_crossfade':
        simple_crossfade(image)
    if transition == 'washout':
        washout(image)
    if transition == 'zoom_in':
        zoom_in(image)
    if transition == 'zoom_out':
        zoom_out(image)
