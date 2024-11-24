import torch
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, AdamW
from transformers import LogitsProcessor, LogitsProcessorList
import re
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk


class TextDataset(Dataset):
    def __init__(self, input_ids):
        self.input_ids = input_ids
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.input_ids[idx].clone()

class GPT2Handler:
    def __init__(self, model_checkpoint_path=None, file_path="updated_raw_text.txt"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2').to(self.device)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.file_path = file_path
        try:
            nltk.download('punkt')
        except:
            pass
        #Load checkpoint if provided
        # if model_checkpoint_path:
        #     checkpoint = torch.load(model_checkpoint_path, map_location=self.device)
        #     self.model.load_state_dict(checkpoint)
        #     print(f"Model state loaded from {model_checkpoint_path}")

    
    def plot_attention_scores(self, text):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)  # Ensure the model is on the correct device
        
        inputs = self.tokenizer(text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        
        attention = outputs.attentions[-1][0]  # Use the last layer's attention scores
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().cpu().tolist())
        
        attention = attention.cpu().numpy()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        cax = ax.matshow(attention, cmap='viridis')
        fig.colorbar(cax)
        
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90, fontsize=10)
        ax.set_yticklabels(tokens, fontsize=10)
        
        plt.show()
    import matplotlib.pyplot as plt

    def plot_attention_scores(self, text):
        """
        Plot attention scores for a given input text.
        """
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        
        attention = outputs.attentions[-1]  # Shape: (batch_size, num_heads, seq_len, seq_len)
        attention = attention.mean(dim=1).squeeze()  # Average over heads, reduce batch dimension
    
        attention = attention.cpu().numpy()
    
        # Convert input IDs to tokens
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())
    
        # Plot attention scores
        fig, ax = plt.subplots(figsize=(10, 8))
        cax = ax.matshow(attention, cmap='viridis')
        fig.colorbar(cax)
    
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90)
        ax.set_yticklabels(tokens)
    
        plt.show()




    def clean_and_structure_text(self, text):
        """Clean and structure text to maintain poetic format."""
        text = re.sub(r'(?i)\b(gonna|gotta|wanna|huh|hey)\b', ' ', text)
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)
        lines = text.split('\n')
        structured_lines = [line.strip() for line in lines if line.strip()]
        return '\n'.join(structured_lines)

    def process_data(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        text = self.clean_and_structure_text(text)
        sequences = []
        lines = text.split('\n')
        chunk_size = 4

        for i in range(0, len(lines), chunk_size):
            chunk = '\n'.join(lines[i:i + chunk_size])
            if chunk.strip():
                tokens = self.tokenizer(
                    chunk,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )
                sequences.append(tokens.input_ids[0])
        
        max_len = max(seq.size(0) for seq in sequences)
        padded_sequences = [
            torch.cat([seq, torch.full((max_len - seq.size(0),), self.tokenizer.pad_token_id)])
            for seq in sequences
        ]
        words = ['the', 'road', 'not', 'taken']
        tokens = {word: self.tokenizer.encode(word, add_special_tokens=False) for word in words}
        print("Tokenization details:")
        for word, token_list in tokens.items():
            print(f"Word: '{word}' => Token IDs: {token_list} => Tokenized words: {self.tokenizer.convert_ids_to_tokens(token_list)}")
        
        # Print the maximum token length for the GPT-2 model
        max_length = self.tokenizer.model_max_length
        print(f"\nMaximum token length for GPT-2: {max_length}")
        
        # Check tokenization of a longer sentence and print its length
        print(f"vocab_size:{len(self.tokenizer)}")
        return torch.stack(padded_sequences)
    
    def calculate_bleu_score(self, reference, candidate):
        """
        Calculate BLEU score between reference and candidate texts.
        
        Args:
            reference (str): The reference text
            candidate (str): The generated text
            
        Returns:
            float: BLEU score
        """
        # Tokenize sentences into words
        reference_tokens = nltk.word_tokenize(reference.lower())
        candidate_tokens = nltk.word_tokenize(candidate.lower())
        
        # Create reference list in required format [[ref_tokens]]
        references = [reference_tokens]
        
        # Apply smoothing function to handle zero counts
        smoothie = SmoothingFunction().method1
        
        # Calculate cumulative 4-gram BLEU score with weights
        weights = (0.25, 0.25, 0.25, 0.25)
        
        try:
            bleu_score = sentence_bleu(references, candidate_tokens, 
                                     weights=weights,
                                     smoothing_function=smoothie)
        except Exception as e:
            print(f"BLEU calculation error: {e}")
            print(f"Reference: {reference_tokens}")
            print(f"Candidate: {candidate_tokens}")
            bleu_score = 0.0
            
        return bleu_score

    def calculate_corpus_bleu(self, dataloader):
        """
        Calculate BLEU score for the entire validation set using generated text.
        
        Args:
            dataloader: DataLoader containing validation data
            
        Returns:
            float: Average BLEU score across the dataset
        """
        self.model.eval()
        total_bleu = 0
        count = 0
        
        with torch.no_grad():
            for batch_inputs, batch_targets in dataloader:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                # Use only the first few tokens as input context
                context_length = batch_inputs.size(1) // 2
                input_context = batch_inputs[:, :context_length]
                
                # Generate continuation
                outputs = self.model.generate(
                    input_context,
                    max_length=batch_inputs.size(1),
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.7,
                    no_repeat_ngram_size=2
                )
                
                # Calculate BLEU for each sample in batch
                for i in range(len(outputs)):
                    # Get the full reference text
                    reference = self.tokenizer.decode(batch_targets[i], skip_special_tokens=True)
                    
                    # Get the generated text (excluding the context)
                    generated = self.tokenizer.decode(outputs[i][context_length:], skip_special_tokens=True)
                    
                    if len(reference.strip()) > 0 and len(generated.strip()) > 0:
                        bleu = self.calculate_bleu_score(reference, generated)
                        total_bleu += bleu
                        count += 1
                        
                    if count % 100 == 0:
                        print(f"\nSample BLEU calculation:")
                        print(f"Reference: {reference[:100]}...")
                        print(f"Generated: {generated[:100]}...")
                        print(f"BLEU score: {bleu:.4f}")
        
        avg_bleu = total_bleu / count if count > 0 else 0.0
        print(f"\nAverage BLEU score: {avg_bleu:.4f} (calculated over {count} samples)")
        return avg_bleu
        
    def train(self, epochs=5, batch_size=8):
        """Train the model on poetry data, with training and validation metrics."""
        input_ids = self.process_data()
        train_ids, val_ids = train_test_split(input_ids, test_size=0.01, random_state=42)
        
        train_dataset = TextDataset(train_ids)
        val_dataset = TextDataset(val_ids)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        
        optimizer = AdamW(self.model.parameters(), lr=1e-5, weight_decay=0.01)
        from transformers import get_linear_schedule_with_warmup
        num_training_steps = len(train_loader) * epochs
        num_warmup_steps = num_training_steps // 10
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Initialize metric trackers
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []
        train_perplexities, val_perplexities = [], []
        bleu_scores = []
        
        print(f"Starting training for {epochs} epochs...")
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs, labels=labels)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()
            
            train_losses.append(epoch_loss / len(train_loader.dataset))
            
            # Evaluate model
            train_loss, train_acc, train_ppl = self.evaluate_model(train_loader)
            val_loss, val_acc, val_ppl = self.evaluate_model(val_loader)
            
            # Calculate BLEU score on validation set
            val_bleu = self.calculate_corpus_bleu(val_loader)
            bleu_scores.append(val_bleu)
            
            # Store metrics
            train_accuracies.append(train_acc)
            train_perplexities.append(train_ppl)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            val_perplexities.append(val_ppl)
            
            print(
                f"Epoch {epoch+1}: "
                f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Train PPL={train_ppl:.4f} | "
                f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, Val PPL={val_ppl:.4f} | "
                f"BLEU={val_bleu:.4f}"
            )
        
        # Save the trained model
        torch.save(self.model.state_dict(), 'poetry_model_final.pt')
        print("Training completed and model saved!")
        
        # Plot metrics including BLEU scores
        self.plot_metrics(epochs, train_losses, val_losses, train_accuracies, 
                         val_accuracies, train_perplexities, val_perplexities, bleu_scores)
    # Helper function to evaluate the model
    def evaluate_model(self, data_loader):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
    
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs, labels=labels)
                loss = outputs.loss
                total_loss += loss.item() * inputs.size(0)
                
                predictions = outputs.logits.argmax(dim=-1)
                total_correct += (predictions == labels).sum().item()
                total_tokens += labels.numel()
    
        avg_loss = total_loss / len(data_loader.dataset)
        accuracy = total_correct / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss))
        return avg_loss, accuracy, perplexity
    
    # Helper function to plot metrics
    def plot_metrics(self, epochs, train_losses, val_losses, train_accuracies, val_accuracies, train_perplexities, val_perplexities):
        import matplotlib.pyplot as plt
    
        epochs_range = range(1, epochs + 1)
    
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, train_losses, label="Training Loss")
        plt.plot(epochs_range, val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training and Validation Loss")
    
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, train_accuracies, label="Training Accuracy")
        plt.plot(epochs_range, val_accuracies, label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Training and Validation Accuracy")

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, train_perplexities, label="Training Perplexities")
        plt.plot(epochs_range, val_perplexities, label="Validation Perplexities")
        plt.xlabel("Epochs")
        plt.ylabel("Perplexity")
        plt.legend()
        plt.title("Training and Validation Perplexity")
    
        plt.tight_layout()
        plt.show()


    def generate(self, prompt, num_words=30):
        self.model.eval()
        prompt = self.clean_and_structure_text(prompt)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_attention_mask=True
        ).to(self.device)
        
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
            max_length=200,
            min_length=60,
            num_return_sequences=1,
            temperature=0.7,
            top_k=100,
            top_p=0.95,
            no_repeat_ngram_size=2,
            repetition_penalty=1.2,
            pad_token_id=self.tokenizer.eos_token_id,
            logits_processor=logits_processor
        )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._format_as_poem(generated_text, num_words)
    
    def _format_as_poem(self, text, num_words):
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
    
    file_path = '/kaggle/input/robert-frost-poem/updated_raw_text.txt'
    model_checkpoint = 'pin_model.pt'  # Add path if resuming training
    
    gpt2_handler = GPT2Handler(model_checkpoint, file_path)
    
    # Train the model
    gpt2_handler.train(epochs=15, batch_size=8)
    
    # Generate text
    prompt = "I was not sure"
    generated_text = gpt2_handler.generate(prompt, num_words=120)
    print(f"\nGenerated text from prompt '{prompt}':\n{generated_text}")
    import matplotlib.pyplot as plt
    sample_text = "Roses are red, violets are blue."
    gpt2_handler.plot_attention_scores(sample_text)
