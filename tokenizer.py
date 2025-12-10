import json
import re
from collections import Counter
from typing import List, Dict, Optional, Any
import unicodedata
from tqdm.auto import tqdm

# Here, you can copy your entire WordTokenizerJSON class code.
class WordTokenizerJSON:
    """
    Custom Word-based Tokenizer with JSON output
    """
    
    def __init__(self, vocab_size: int = 150000, min_word_freq: int = 1, special_tokens: Optional[List[str]] = None):
        self.vocab_size = vocab_size
        self.min_word_freq = min_word_freq
        self.special_tokens = special_tokens or [
            '<pad>', '<unk>', '<bos>', '<eos>', '<mask>', '<sep>'
        ]
        self.token_to_id = {}
        self.id_to_token = {}
        self.word_freq = {}
        for i, token in enumerate(self.special_tokens):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
    
    def _preprocess_text(self, text: str) -> str:
        text = unicodedata.normalize('NFKC', text)
        text = re.sub(r'\s+', ' ', text.strip())
        return text
    
    def _tokenize_words(self, text: str) -> List[str]:
        text = self._preprocess_text(text)
        pattern = r'[\u0A80-\u0AFF]+|[a-zA-Z]+|[0-9]+|[^\s\u0A80-\u0AFFa-zA-Z0-9]+'
        words = re.findall(pattern, text.lower())
        words = [word.strip() for word in words if word.strip()]
        return words
    
    def _build_vocabulary(self, texts: List[str]) -> Dict[str, Any]:
        print("Building vocabulary...")
        word_counter = Counter()
        total_words = 0
        for text in texts:
            words = self._tokenize_words(text)
            word_counter.update(words)
            total_words += len(words)
        filtered_words = {word: count for word, count in word_counter.items() if count >= self.min_word_freq}
        sorted_words = sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)
        max_words = self.vocab_size - len(self.special_tokens)
        top_words = sorted_words[:max_words]
        current_id = len(self.special_tokens)
        for word, freq in top_words:
            self.token_to_id[word] = current_id
            self.id_to_token[current_id] = word
            self.word_freq[word] = freq
            current_id += 1
        vocab_stats = {
            "total_unique_words": len(word_counter),
            "total_words_processed": total_words,
            "words_after_filtering": len(filtered_words),
            "final_vocab_words": len(top_words),
            "final_vocab_size": len(self.token_to_id),
            "coverage_percentage": round((sum(freq for _, freq in top_words) / total_words) * 100, 2),
            "most_frequent_words": top_words[:10],
            "min_frequency_threshold": self.min_word_freq
        }
        return vocab_stats
    
    def train(self, texts: List[str]) -> str:
        print("Starting word tokenizer training...")
        training_stats = {
            "training_config": {
                "vocab_size": self.vocab_size,
                "min_word_freq": self.min_word_freq,
                "special_tokens": self.special_tokens,
                "total_training_texts": len(texts)
            },
            "corpus_stats": {
                "total_characters": sum(len(text) for text in texts),
                "average_text_length": round(sum(len(text) for text in texts) / len(texts), 2)
            }
        }
        
        # Show loop progress using tqdm
        texts_with_progress = tqdm(texts, desc="Processing texts for vocab", unit="text")
        vocab_stats = self._build_vocabulary(texts_with_progress)
        
        training_stats["vocabulary_stats"] = vocab_stats
        gujarati_words, english_words, other_words = 0, 0, 0
        for word in self.token_to_id.keys():
            if word in self.special_tokens: continue
            elif re.match(r'^[\u0A80-\u0AFF]+$', word): gujarati_words += 1
            elif re.match(r'^[a-zA-Z]+$', word): english_words += 1
            else: other_words += 1
        training_stats["language_stats"] = {
            "gujarati_words": gujarati_words, "english_words": english_words, "other_tokens": other_words,
            "gujarati_percentage": round((gujarati_words / (gujarati_words + english_words + other_words)) * 100, 2),
            "english_percentage": round((english_words / (gujarati_words + english_words + other_words)) * 100, 2)
        }
        print("Training complete!")
        print(f"Final vocabulary size: {len(self.token_to_id)}")
        return json.dumps(training_stats, ensure_ascii=False, indent=2)
    
    def encode(self, text: str, add_special_tokens: bool = False, return_json: bool = True) -> Any:
        if not text.strip():
            result = {"input_text": text, "token_ids": [], "tokens": [], "encoding_stats": {"total_tokens": 0, "unknown_tokens": 0, "known_tokens": 0, "oov_rate": 0.0}}
            return json.dumps(result, ensure_ascii=False, indent=2) if return_json else result
        words = self._tokenize_words(text)
        token_ids, tokens, unknown_count = [], [], 0
        if add_special_tokens:
            token_ids.append(self.token_to_id['<bos>'])
            tokens.append('<bos>')
        for word in words:
            if word in self.token_to_id:
                token_ids.append(self.token_to_id[word])
                tokens.append(word)
            else:
                token_ids.append(self.token_to_id['<unk>'])
                tokens.append('<unk>')
                unknown_count += 1
        if add_special_tokens:
            token_ids.append(self.token_to_id['<eos>'])
            tokens.append('<eos>')
        known_tokens = len(words) - unknown_count
        oov_rate = (unknown_count / len(words)) * 100 if len(words) > 0 else 0
        result = {
            "input_text": text, "token_ids": token_ids, "tokens": tokens,
            "encoding_stats": {"total_tokens": len(token_ids), "original_words": len(words), "unknown_tokens": unknown_count, "known_tokens": known_tokens, "oov_rate": round(oov_rate, 2), "special_tokens_added": add_special_tokens}
        }
        return json.dumps(result, ensure_ascii=False, indent=2) if return_json else result
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True, return_json: bool = True) -> Any:
        if not token_ids:
            result = {"input_token_ids": token_ids, "decoded_text": "", "decoding_stats": {"total_tokens_processed": 0, "special_tokens_skipped": 0, "unknown_tokens_found": 0, "words_recovered": 0}}
            return json.dumps(result, ensure_ascii=False, indent=2) if return_json else result
        tokens, special_tokens_count, unknown_tokens_count = [], 0, 0
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if token in self.special_tokens:
                    special_tokens_count += 1
                    if not skip_special_tokens: tokens.append(token)
                    elif token not in ['<pad>', '<bos>', '<eos>']: tokens.append(token)
                else: tokens.append(token)
                if token == '<unk>': unknown_tokens_count += 1
        decoded_text = ' '.join(tokens)
        decoded_text = re.sub(r'\s+', ' ', decoded_text).strip()
        result = {
            "input_token_ids": token_ids, "decoded_text": decoded_text,
            "decoding_stats": {"total_tokens_processed": len(token_ids), "special_tokens_skipped": special_tokens_count if skip_special_tokens else 0, "unknown_tokens_found": unknown_tokens_count, "words_recovered": len(tokens), "final_text_length": len(decoded_text)}
        }
        return json.dumps(result, ensure_ascii=False, indent=2) if return_json else result
    
    def get_word_info(self, word: str) -> str:
        word = word.lower()
        word_info = {
            "word": word, "is_in_vocab": word in self.token_to_id, "token_id": self.token_to_id.get(word, None),
            "frequency": self.word_freq.get(word, 0), "word_type": self._detect_word_type(word),
            "alternatives": self._get_similar_words(word)
        }
        return json.dumps(word_info, ensure_ascii=False, indent=2)
    
    def _detect_word_type(self, word: str) -> str:
        if word in self.special_tokens: return "special_token"
        elif re.match(r'^[\u0A80-\u0AFF]+$', word): return "gujarati"
        elif re.match(r'^[a-zA-Z]+$', word): return "english"
        elif re.match(r'^[0-9]+$', word): return "number"
        else: return "other"
    
    def _get_similar_words(self, word: str, limit: int = 5) -> List[str]:
        similar_words, word_type = [], self._detect_word_type(word)
        for vocab_word in list(self.token_to_id.keys())[:100]:
            if vocab_word in self.special_tokens: continue
            if self._detect_word_type(vocab_word) == word_type:
                if vocab_word.startswith(word[:2]) or word.startswith(vocab_word[:2]):
                    similar_words.append(vocab_word)
            if len(similar_words) >= limit: break
        return similar_words
    
    def get_vocabulary_stats(self) -> str:
        type_counts = {"gujarati": 0, "english": 0, "numbers": 0, "other": 0, "special": 0}
        for word in self.token_to_id.keys():
            word_type = self._detect_word_type(word)
            if word_type == "special_token": type_counts["special"] += 1
            elif word_type == "gujarati": type_counts["gujarati"] += 1
            elif word_type == "english": type_counts["english"] += 1
            elif word_type == "number": type_counts["numbers"] += 1
            else: type_counts["other"] += 1
        gujarati_words = [(w, f) for w, f in self.word_freq.items() if self._detect_word_type(w) == "gujarati"]
        english_words = [(w, f) for w, f in self.word_freq.items() if self._detect_word_type(w) == "english"]
        gujarati_words.sort(key=lambda x: x[1], reverse=True)
        english_words.sort(key=lambda x: x[1], reverse=True)
        stats = {
            "vocabulary_overview": {"total_vocabulary_size": len(self.token_to_id), "word_type_distribution": type_counts,
                "coverage_stats": {"total_word_frequencies": sum(self.word_freq.values()), "average_word_frequency": round(sum(self.word_freq.values()) / len(self.word_freq), 2) if self.word_freq else 0}
            },
            "language_breakdown": {"top_gujarati_words": gujarati_words[:10], "top_english_words": english_words[:10]},
            "special_tokens": self.special_tokens
        }
        return json.dumps(stats, ensure_ascii=False, indent=2)
    
    def save_to_json(self, filepath: str) -> str:
        tokenizer_data = {
            "metadata": {"tokenizer_type": "WordBasedTokenizer", "vocab_size": self.vocab_size, "min_word_freq": self.min_word_freq, "current_vocab_size": len(self.token_to_id), "special_tokens": self.special_tokens},
            "vocabulary": {"token_to_id": self.token_to_id, "id_to_token": {str(k): v for k, v in self.id_to_token.items()}, "word_frequencies": self.word_freq}
        }
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
            save_result = {"status": "success", "message": f"Word tokenizer successfully saved to {filepath}", "file_size_kb": round(len(json.dumps(tokenizer_data)) / 1024, 2), "vocab_size": len(self.token_to_id)}
        except Exception as e:
            save_result = {"status": "error", "message": f"Failed to save tokenizer: {str(e)}"}
        return json.dumps(save_result, ensure_ascii=False, indent=2)
    
    def load_from_json(self, filepath: str) -> str:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                tokenizer_data = json.load(f)
            metadata = tokenizer_data['metadata']
            self.vocab_size, self.min_word_freq, self.special_tokens = metadata['vocab_size'], metadata['min_word_freq'], metadata['special_tokens']
            vocabulary = tokenizer_data['vocabulary']
            self.token_to_id, self.id_to_token, self.word_freq = vocabulary['token_to_id'], {int(k): v for k, v in vocabulary['id_to_token'].items()}, vocabulary['word_frequencies']
            load_result = {"status": "success", "message": f"Word tokenizer successfully loaded from {filepath}", "loaded_vocab_size": len(self.token_to_id), "loaded_word_frequencies": len(self.word_freq)}
        except Exception as e:
            load_result = {"status": "error", "message": f"Failed to load tokenizer: {str(e)}"}
        return json.dumps(load_result, ensure_ascii=False, indent=2)