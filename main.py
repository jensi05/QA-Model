import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import random

# Module required for mixed precision
from torch.cuda.amp import autocast, GradScaler

from config import Config
from data import QADataset, load_json_data, create_mask
from model import Seq2SeqTransformer
from tokenizer import WordTokenizerJSON

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def train_model():
    # Load settings from config
    config = Config()

    # Create directory if it doesn't exist
    if not os.path.exists(config.DATA_DIR):
        os.makedirs(config.DATA_DIR)
    if not os.path.exists(config.CHECKPOINTS_DIR):
        os.makedirs(config.CHECKPOINTS_DIR)

    # Load data (now passing directory path)
    all_data = load_json_data(config.DATA_DIR)
    if not all_data:
        print("Data loading failed. Exiting.")
        return
    print(f"\nTotal data loaded: {len(all_data)} question-answer pairs.")

    # Shuffle the data
    random.shuffle(all_data)

    # Split the data into training and validation sets
    train_split = int(len(all_data) * config.TRAIN_SPLIT_RATE)
    train_data = all_data[:train_split]
    val_data = all_data[train_split:]
    print(f"Training data size: {len(train_data)}")
    print(f"Validation data size: {len(val_data)}")

    # Load and update the tokenizer
    tokenizer = WordTokenizerJSON(
        vocab_size=config.SRC_VOCAB_SIZE, 
        min_word_freq=config.MIN_WORD_FREQ, 
        special_tokens=config.SPECIAL_TOKENS
    )
    
    if os.path.exists(config.VOCAB_FILE):
        print(f"\n--- Loading existing tokenizer from {config.VOCAB_FILE} ---")
        tokenizer.load_from_json(config.VOCAB_FILE)
    else:
        print(f"\n--- {config.VOCAB_FILE} not found. Creating a new vocabulary ---")
        
    print("\n--- Updating vocabulary with new data ---")
    all_texts = [item['Question'] + " " + item['Answer'] for item in all_data]
    
    new_words_count = 0
    for text in tqdm(all_texts, desc="Processing new words", unit="text"):
        words = tokenizer._tokenize_words(text)
        for word in words:
            if word not in tokenizer.token_to_id:
                new_id = len(tokenizer.token_to_id)
                tokenizer.token_to_id[word] = new_id
                tokenizer.id_to_token[new_id] = word
                tokenizer.word_freq[word] = 1
                new_words_count += 1
    
    print(f"Total new words added: {new_words_count}")
    print(f"Final vocabulary size: {len(tokenizer.token_to_id)}")
    
    print(f"\n--- Saving updated vocabulary to {config.VOCAB_FILE} ---")
    tokenizer.save_to_json(config.VOCAB_FILE)
    
    # Initialize the model
    actual_vocab_size = len(tokenizer.token_to_id)
    
    transformer_model = Seq2SeqTransformer(
        num_encoder_layers=config.NUM_ENCODER_LAYERS, 
        num_decoder_layers=config.NUM_DECODER_LAYERS, 
        emb_size=config.EMB_SIZE, 
        nhead=config.NHEAD, 
        src_vocab_size=actual_vocab_size, 
        tgt_vocab_size=actual_vocab_size, 
        dim_feedforward=config.FF_DIM, 
        dropout=config.DROPOUT
    )
    
    for p in transformer_model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    transformer_model.to(device)

    # Prepare training and validation data loaders
    train_dataset = QADataset(train_data, tokenizer, max_len=config.MAX_LEN)
    val_dataset = QADataset(val_data, tokenizer, max_len=config.MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id['<pad>'])
    optimizer = torch.optim.Adam(transformer_model.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)
    
    # GradScaler ને ઇનિશિયલાઇઝ કરો
    scaler = GradScaler()
    
    # મોડેલને તાલીમ આપો
    print("\n--- Training Model ---")
    for epoch in range(config.EPOCHS):
        # ટ્રેનિંગ લૂપ
        transformer_model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Training]", unit="batch")
        for src, tgt in progress_bar:
            src = src.T.to(device)
            tgt = tgt.T.to(device)
            
            tgt_input = tgt[:-1, :]
            
            # autocast નો ઉપયોગ કરીને ગણતરી FP16 માં કરો
            with autocast():
                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, tokenizer.token_to_id['<pad>'])
                
                logits = transformer_model(src, tgt_input, src_mask, tgt_mask, 
                                           src_padding_mask, tgt_padding_mask, src_padding_mask)
                
                loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt[1:, :].reshape(-1))
            
            # scaler નો ઉપયોગ કરીને loss ને સ્કેલ કરો અને બેકવર્ડ પાસ કરો
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            optimizer.zero_grad()
            
            progress_bar.set_postfix(loss=loss.item())

        # વેલિડેશન લૂપ
        transformer_model.eval()
        total_val_loss = 0
        with torch.no_grad():
            progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Validation]", unit="batch")
            for src, tgt in progress_bar_val:
                src = src.T.to(device)
                tgt = tgt.T.to(device)

                tgt_input = tgt[:-1, :]
                
                # autocast નો ઉપયોગ કરીને ગણતરી FP16 માં કરો
                with autocast():
                    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, tokenizer.token_to_id['<pad>'])

                    logits = transformer_model(src, tgt_input, src_mask, tgt_mask,
                                               src_padding_mask, tgt_padding_mask, src_padding_mask)

                    loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt[1:, :].reshape(-1))
                    total_val_loss += loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")

        # મોડેલને દરેક epoch પછી સેવ કરો
        checkpoint_path = os.path.join(config.CHECKPOINTS_DIR, f"transformer_model_epoch_{epoch+1}.pth")
        torch.save(transformer_model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved successfully to {checkpoint_path}")

    print("\n--- Model Training Complete ---")
    
    # અંતિમ મોડેલને પણ સેવ કરો
    torch.save(transformer_model.state_dict(), config.MODEL_FILE)
    print(f"Final model saved successfully to {config.MODEL_FILE}")

if __name__ == "__main__":
    train_model()