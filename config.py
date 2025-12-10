import os

class Config:
    # Data and file paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # directories
    DATA_DIR = os.path.join(BASE_DIR, 'old_data')
    CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'checkpoints')
    
    # Files
    VOCAB_FILE = os.path.join(BASE_DIR, 'vocab.json') 
    MODEL_FILE = os.path.join(CHECKPOINTS_DIR, 'transformer_model.pth')

    # Model hyperparameters
    SRC_VOCAB_SIZE = 200000
    TGT_VOCAB_SIZE = 200000
    EMB_SIZE = 128
    NHEAD = 4
    FF_DIM = 1024
    NUM_ENCODER_LAYERS = 2   
    NUM_DECODER_LAYERS = 2  
    DROPOUT = 0.05
    
    # Training parameters
    EPOCHS = 1
    BATCH_SIZE = 16   
    LEARNING_RATE = 0.00005
    
    # Tokenizer parameters
    MIN_WORD_FREQ = 1
    SPECIAL_TOKENS = ['<pad>', '<unk>', '<bos>', '<eos>', '<mask>', '<sep>']
    
    # Other
    MAX_LEN = 128
    TRAIN_SPLIT_RATE = 0.8