from model import MLPClassifier, BanhMiBert, MoEClassifier, MoEBanhMiBert

# init tokenizer and model
def init_model_and_tokenizer(
    config,
    arch_type = "mlp",
    freeze = False,
    n_freezes = 2,
    mode = "encoder",
    verbose = False,
):
    """
    Initializes the Tokenizer and Model (supporting both MLP and MoE heads) 
    along with layer freezing logic.
    
    :param config: Configuration object containing hyperparameters (Config)
    :param arch_type: Architecture type for the classification head ('mlp' or 'moe')
    :param freeze: Whether to activate layer freezing (True/False)
    :param n_freezes: Number of bottom encoder layers to freeze (up to 12)
    :param mode: Freezing strategy ('embeddings', 'encoder', 'all_bert')
    :param verbose: If True, suppresses the model architecture report printout
    """
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    # Init type of Model
    if arch_type.lower() == "moe":
        print("Init Architechture: MoEBanhMiBert")
        model = MoEBanhMiBert(config)
    else:
        print("Init Architechture: BanhMiBert (MLP)")
        model = BanhMiBert(config)
        
    if freeze:
        print(f"\n[Freeze Active] Mode: {mode} | freezed: {n_freezes} hidden layers")

        # Freeze embeddings
        if mode in ['embeddings', 'encoder', 'all_bert']:
            for param in model.bamibert.embeddings.parameters():
                param.requires_grad = False
            
        # freeze encoders
        if mode in ['encoder', 'all_bert']:
            num_layers_to_freeze = min(n_freezes or 0, model.bamibert.config.num_hidden_layers)
            for layer_idx in range(num_layers_to_freeze):
                for param in model.bamibert.encoder.layer[layer_idx].parameters():
                    param.requires_grad = False
    else:
        print("[Full Training] Finetune all weights in BamiBert")

    if verbose:
        print("\n" + "="*50)
        print(f"REPORT MODEL: ({arch_type.upper()})")
        print("="*50)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"-> All Params (BERT + Head) : {total_params:,}")
        print(f"-> Trainable Params         : {trainable_params:,}")
        print(f"-> Frozen Params            : {frozen_params:,}")
        print("-"*50)
        
        cls_params = sum(p.numel() for p in model.classifier.parameters())
        print(f"{'classifier (' + arch_type.upper() + ')':<10} | {cls_params:<8} | Training")
        print("="*50 + "\n")
        
    return model, tokenizer

# training
def train_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    scheduler, 
    device,
):
    model.train()
    total_loss = 0

    progress_bar = tqdm(dataloader, desc = "Training")

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # gradients
        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)
        # backward
        loss.backward()

        # gradient clipping to avoid explore and vanishing gradients
        clip_grad_norm(model.parameters(), max_norm = 1.0)
        
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    return total_loss / len(dataloader)

# Evaluations
def evaluate(model, dataloader, compute_metrics, device):
    model.eval()
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            
    # Concatenate all batches together to compute global metrics
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Standard format wrapper to reuse your compute_metrics function
    class EvalPred:
        def __init__(self, predictions, label_ids):
            self.predictions = predictions.numpy()
            self.label_ids = label_ids.numpy()
            
    metrics = compute_metrics(EvalPred(all_logits, all_labels))
    return metrics
    
def main():
    from kaggle_secrets import UserSecretsClient
    import wandb
    user_secrets = UserSecretsClient()
    secret_value_0 = user_secrets.get_secret("wandb_key")
    wandb.login(key=secret_value_0)
    
    #=== SET UP===#
    print("Set up training")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Use: {device}")
    
    
    # Init Data Preparation
    train_df = pd.read_csv(config.train_path)
    test_df = pd.read_csv(config.test_path)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    vnp = VnCoreNLP(config.vncorenlp_path,annotators="wseg")
    
    full_encoded = preprocess_data(train_df, "comment", "label", tokenizer, vnp, config.max_len)

    X_ids = full_encoded["input_ids"]
    X_mask = full_encoded["attention_mask"]
    y_labels = full_encoded["labels"]

    # folds
    skf = StratifiedKFold(
        n_splits = config.n_splits,
        shuffle = True,
        random_state = config.seed
    )

    print(f"\nTRAINING {config.n_splits}-FOLD CROSS VALIDATION")
    fold_f1_score = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y_labels)), y_labels.numpy())):
        print(f"\n" + "="*25 + f" FOLD {fold + 1} / {config.n_splits} " + "="*25)
        
        # init WanDB
        wandb.init(
            project="BanhMiBert-MLP-KFOLD",
            name=f"Fold-{fold + 1}",
            group="Experiment-V2-MLP",
            config={
                "learning_rate": config.learning_rate,
                "epochs": config.epochs,
                "batch_size": config.train_batch_size,
                "architecture": "MLP"
            }
        )
        
        # training
        train_dataset = SentimentDataset(X_ids[train_idx], X_mask[train_idx], y_labels[train_idx])
        valid_dataset = SentimentDataset(X_ids[val_idx], X_mask[val_idx], y_labels[val_idx])
        train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=config.valid_batch_size, shuffle=False)

        model, _ = init_model_and_tokenizer(
            config,
            arch_type = "mlp",
            freeze = False,
            mode = None,
            n_freezes = None,
            verbose = True,
        )
    
        model.to(device)
        weights = torch.tensor([0.25, 0.75]).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay)
        
        # init parameters
        total_steps = len(train_loader) * config.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * 0.1), 
            num_training_steps=total_steps
        )

        best_fold_f1 = 0.0
        
        for epoch in range(config.epochs):
            avg_train_loss = train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scheduler,
                device,
            )
            
            val_metrics = evaluate(
                model,
                valid_loader,
                compute_metrics,
                device
            )
            
            print(
                f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f} | "
                f"Val Precision: {val_metrics['precision']:.4f} | "
                f"Val Recall: {val_metrics['recall']:.4f}"
            )

            # log for wandb
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_accuracy": val_metrics['accuracy'],
                "val_f1_macro": val_metrics['f1'],
                "val_precision": val_metrics['precision'],
                "val_recall": val_metrics['recall']
            }, step = epoch)
    
            if val_metrics["f1"] > best_fold_f1:
                best_fold_f1 = val_metrics["f1"]
                torch.save(model.state_dict(), f"bamibert_mlp_fold{fold+1}.pth")
        
        print(f"Ending Fold {fold + 1} | Best Val F1-Score: {best_fold_f1:.4f}")
        fold_f1_score.append(best_fold_f1)
        wandb.finish()
    
    print("\n" + "="*50)
    print(f"➔ MEAN F1-SCORE OVER ROUNDS: {np.mean(fold_f1_score):.4f}")
    print("="*50)