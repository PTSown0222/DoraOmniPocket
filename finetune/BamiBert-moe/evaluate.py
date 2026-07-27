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