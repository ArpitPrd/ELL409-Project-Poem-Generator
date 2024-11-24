import torch
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, AdamW
from transformers import LogitsProcessor, LogitsProcessorList
import re
from torch.utils.data import Dataset, DataLoader
class TextDataset(Dataset):
    def __init__(self, input_ids):
        self.input_ids = input_ids
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.input_ids[idx].clone()
    

class GPT2Handler:
    def __init__(self, model_checkpoint_path, file_path = "updated_raw_text.txt"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2').to(self.device)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.file_path = file_path
        self.maxi = None
        
        #Load the saved checkpoint
        checkpoint = torch.load(model_checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        print(f"Model state loaded from {model_checkpoint_path}")
        
    def clean_and_structure_text(self, text):
        """Clean and structure text to maintain poetic format"""
        text = re.sub(r'(?i)\b(gonna|gotta|wanna|huh|hey)\b', '', text)
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)
        
        lines = text.split('\n')
        structured_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                line = ' '.join(line.split())
                structured_lines.append(line)
        
        return '\n'.join(structured_lines)
    def process_data(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Clean and structure the text
        text = self.clean_and_structure_text(text)
        
        # Create sequences that respect line and stanza breaks
        sequences = []
        lines = text.split('\n')
        
        # Group lines into potential stanza-sized chunks
        chunk_size = 4  # Typical size for a Frost stanza
        for i in range(0, len(lines), chunk_size):
            chunk = '\n'.join(lines[i:i + chunk_size])
            if len(chunk.strip()) > 0:
                tokens = self.tokenizer(
                    chunk,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )
                sequences.append(tokens.input_ids[0])
        
        # Pad sequences to the same length
        max_len = max(seq.size(0) for seq in sequences)
        padded_sequences = []
        for seq in sequences:
            if seq.size(0) < max_len:
                padding = torch.full((max_len - seq.size(0),), self.tokenizer.pad_token_id)
                seq = torch.cat([seq, padding])
            padded_sequences.append(seq)
        
        return torch.stack(padded_sequences)
    
    
    def train(self, epochs=20, batch_size=4):
        """Train the model on poetry data"""
        # Process the input data
        input_ids = self.process_data()
        dataset = TextDataset(input_ids)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize optimizer with a lower learning rate for fine-tuning
        optimizer = AdamW(self.model.parameters(), lr=1e-5, weight_decay=0.01)
        
        # Learning rate scheduler for stable training
        from transformers import get_linear_schedule_with_warmup
        num_training_steps = len(dataloader) * epochs
        num_warmup_steps = num_training_steps // 10
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        print(f"Starting training for {epochs} epochs...")
        self.model.train()
        
        total_samples = 0
        running_loss = 0.0
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_idx, (inputs, labels) in enumerate(dataloader):
                # Move batch to device
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Clear gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs, labels=labels)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Update weights
                optimizer.step()
                scheduler.step()
                
                # Update statistics
                batch_loss = loss.item()
                epoch_loss += batch_loss
                running_loss += batch_loss
                total_samples += 1
                
                # Print batch statistics
                if batch_idx % 5 == 0:
                    avg_loss = running_loss / total_samples
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(dataloader)}, "
                        f"Loss: {batch_loss:.4f}, Average Loss: {avg_loss:.4f}")
            
            # Print epoch statistics
            avg_epoch_loss = epoch_loss / len(dataloader)
            print(f"\nEpoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.4f}\n")
            
            # Save checkpoint after each epoch
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }
            torch.save(checkpoint, f'poetry_model_epoch_{epoch+1}.pt')
        
        print("Training completed!")
    def generate(self, prompt, num_words=30):
        self.model.eval()
        
        # Clean and structure the prompt
        prompt = self.clean_and_structure_text(prompt)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_attention_mask=True
        ).to(self.device)
        
        # Nature-related words common in Frost's poetry
        frost_style_words = [
            "woods", "snow", "path", "trees", "leaves", "dark",
            "winter", "frost", "ice", "wind", "road", "mountain",
            "forest", "night", "cold", "silence", "dawn", "birch",
            "maple", "stone", "wall", "field", "farm", "earth"
        ]
        
        class VocabularyBiasLogitsProcessor(LogitsProcessor):
            def __init__(self, frost_words, tokenizer, bias_amount=5.0):
                self.frost_word_ids = set()
                for word in frost_words:
                    ids = tokenizer.encode(word, add_special_tokens=False)
                    self.frost_word_ids.update(ids)
                self.bias_amount = bias_amount

            def __call__(self, input_ids, scores):
                for token_id in self.frost_word_ids:
                    if token_id < scores.shape[-1]:
                        scores[:, token_id] += self.bias_amount
                return scores

        logits_processor = LogitsProcessorList([
            VocabularyBiasLogitsProcessor(frost_style_words, self.tokenizer)
        ])
        
        outputs = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=100,
            min_length=60,
            num_return_sequences=4,
            temperature=0.5,
            do_sample=True,
            top_k=0,
            top_p=0.95,
            no_repeat_ngram_size=2,
            repetition_penalty=1.2,
            pad_token_id=self.tokenizer.eos_token_id,
            logits_processor=logits_processor
        )
        
        generated_texts = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        
        best_text = max(generated_texts, key=self._score_frost_style)
        formatted_text = self._format_as_poem(best_text, num_words)
        return formatted_text
    
    def _score_frost_style(self, text):
        """Score text based on Frost-like characteristics"""
        score = 0
        
        nature_words = ["wood", "tree", "leaf", "snow", "frost", "ice", "wind", 
                       "path", "road", "mountain", "bird", "flower"]
        for word in nature_words:
            if word in text.lower():
                score += 2
        
        lines = text.split('\n')
        for line in lines:
            words = line.split()
            if 6 <= len(words) <= 10:
                score += 1
        
        ending_words = ["sleep", "keep", "deep", "way", "day", "night", "sight"]
        for word in ending_words:
            if text.lower().endswith(word):
                score += 3
        
        return score
    
    def _format_as_poem(self, text, num_words):
        """Format text as a poem with appropriate line breaks"""
        words = text.split()[:num_words]
        lines = []
        current_line = []
        
        for word in words:
            current_line.append(word)
            if len(current_line) >= 8 or word.endswith(('.', '?', '!')):
                lines.append(' '.join(current_line))
                current_line = []
        
        if current_line:
            lines.append(' '.join(current_line))
            
        return '\n'.join(lines)


if __name__ == "__main__":
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    file_path = 'updated_raw_text.txt'
    #gpt2_handler_1 = GPT2Handler(file_path)
    
    # Train the model
    # gpt2_handler_1.train(epochs=20)
    # torch.save(gpt2_handler_1.model.state_dict(), "fine_tuned_model.pt")
    gpt2_handler_1 = GPT2Handler('fine_tuned_model.pt')
    prompt = "The road not taken"
    generated_text = gpt2_handler_1.generate(prompt, num_words=120)
    print(f"\nGenerated text from prompt '{prompt}':\n{generated_text}")
    
    