def collate_fn(batch: list[dict]):
    """Custom collate function to handle variable-length sequences in dataset."""
    # always at max length: tokens, segment_ids; always singleton: is_random_next
    input_ids = torch.tensor([item["tokens"] for item in batch])
    token_type_ids = torch.tensor([item["segment_ids"] for item in batch]).abs()
    is_random_next = torch.tensor([item["is_random_next"] for item in batch]).to(int)
    # variable length: masked_positions, masked_labels
    masked_pos = [(idx, pos) for idx, item in enumerate(batch) for pos in item["masked_positions"]]
    masked_labels = torch.tensor([label for item in batch for label in item["masked_labels"]])
    return input_ids, token_type_ids, is_random_next, masked_pos, masked_labels